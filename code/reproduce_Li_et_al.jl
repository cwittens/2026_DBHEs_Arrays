# Function to extract 1D profile along x-axis at given depth (only x > xc)
# FIXME: use the one from package
function extract_x_profile(T, gridx, gridy, gridz, z_target, xc, yc, full_profile=false)
    # Find closest y index (at borehole center)
    y_idx = argmin(abs.(gridy .- yc))

    # Interpolate in z-direction
    if z_target <= gridz[1]
        @warn "Requested z_target $z_target is below the grid minimum $(gridz[1]). Using first layer."
        T_full = T[1, y_idx, :]
    elseif z_target >= gridz[end]
        @warn "Requested z_target $z_target is above the grid maximum $(gridz[end]). Using last layer."
        # Above grid, use last layer
        T_full = T[end, y_idx, :]
    else
        # Find bracketing indices
        z_idx_lower = findlast(gridz .<= z_target)
        z_idx_upper = findfirst(gridz .>= z_target)

        if z_idx_lower == z_idx_upper
            # Exactly on a grid point
            T_full = T[z_idx_lower, y_idx, :]
        else
            # Linear interpolation between layers
            z_lower = gridz[z_idx_lower]
            z_upper = gridz[z_idx_upper]
            weight_upper = (z_target - z_lower) / (z_upper - z_lower)
            weight_lower = 1.0 - weight_upper

            T_full = weight_lower * T[z_idx_lower, y_idx, :] + weight_upper * T[z_idx_upper, y_idx, :]
        end
    end

    # Filter for x > xc
    if full_profile
        return gridx, T_full
    else
        mask = gridx .> xc
        x_filtered = gridx[mask]
        T_profile = T_full[mask]

        # Compute radial distances (now just x - xc, not absolute value)
        r_vals = x_filtered .- xc

        return r_vals, T_profile
    end
end

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using GeothermalWells
using KernelAbstractions: CPU, adapt, @kernel, @index, zeros
using OrdinaryDiffEqStabilizedRK: ODEProblem, solve, ROCK2
using Statistics: mean
using JLD2: @save, @load
using CUDA: CUDABackend
using Plots
plots_dir() = joinpath(@__DIR__, "plots")
!isdir(plots_dir()) && mkdir(plots_dir()) # create plots directory if it doesn't exist
simulation_data_dir() = joinpath(@__DIR__, "simulation_data")
!isdir(simulation_data_dir()) && mkdir(simulation_data_dir()) # create simulation data directory if it doesn't exist

Float_used = Float64
backend = CUDABackend() # or backend= ROCBackend() or backend = CUDABackend() or backend = CPU()

materials_stratified = StratifiedMaterialProperties{4,Float_used}(
    (1.8, 2.6, 3.5, 5.3),                                # k_rock
    (1780 * 1379, 2030 * 1450, 1510 * 1300, 2600 * 878), # rho_c_rock
    (500.0, 1000.0, 1500.0, 2000.0),                     # depths of different layers
    0.618, 4.166e6,  # k and rho_c of water
    41.0, 7850 * 475,   # k and rho_c of  outer tube (rho_c is a guess / not given in paper)
    0.4, 1955000,    # k and rho_c of insulation (rho_c is a guess / not given in paper)
    1.5, 1.76e6      # k and rho_c of backfill (rho_c is a guess / not given in paper)
)


borehole = Borehole{Float_used}(
    0.0,              # xc
    0.0,              # yc
    2000,             # h (from Table 2)
    0.0511,           # r_inner (calculated from 125-2×11.4)/2
    0.0114,           # t_inner = 11.4 mm
    0.0885,           # r_outer (calculated from (193.7-2×8.33)/2)
    0.00833,          # t_outer = 8.33 mm
    0.115,            # r_backfill = 115 mm (ASSUMED - try to match their results)
    42 * 998.2 / 3600, # m³/h * kg/m³ / s/h = kg/s
    0  # no insulation of the outer pipe)
)

boreholes = (borehole,)

# the thermal gradient used in Li et al. is not 100% specified, which is why we linearly interpolate between the different regions were with know the temperature


r500, T500 = data_li(1)   # get 500m  data
r1000, T1000 = data_li(2)  # get 1000m data
r1500, T1500 = data_li(3)  # get 1500m data
r2000, T2000 = data_li(4)  # get 2000m data

# away from the borehole, the temperature after 120 days should be the same as the initial temperature
n = 5
T_at_500 = mean(T500[end-n:end]) # average last n values
T_at_1000 = mean(T1000[end-n:end]) # average last n values
T_at_1500 = mean(T1500[end-n:end]) # average last n values
T_at_2000 = mean(T2000[end-n:end]) # average last n values

@inline function initial_temperature_Li_open(z, T_at_500, T_at_1000, T_at_1500, T_at_2000)
    # Known depths and temperatures from Figure 5
    z_data = [500.0, 1000.0, 1500.0, 2000.0]
    T_data = [T_at_500, T_at_1000, T_at_1500, T_at_2000]

    # Find the appropriate interval for interpolation/extrapolation
    if z <= z_data[1]
        # Extrapolate upward using first two points
        slope = (T_data[2] - T_data[1]) / (z_data[2] - z_data[1])
        return T_data[1] + slope * (z - z_data[1])
    elseif z >= z_data[end]
        # Extrapolate downward using last two points
        slope = (T_data[end] - T_data[end-1]) / (z_data[end] - z_data[end-1])
        return T_data[end] + slope * (z - z_data[end])
    else
        # Interpolate between points
        for i in 1:length(z_data)-1
            if z >= z_data[i] && z <= z_data[i+1]
                slope = (T_data[i+1] - T_data[i]) / (z_data[i+1] - z_data[i])
                return T_data[i] + slope * (z - z_data[i])
            end
        end
    end
end

initial_temperature_Li(z) = Float_used(initial_temperature_Li_open(z, T_at_500, T_at_1000, T_at_1500, T_at_2000))


xmin, xmax = -100, 100
ymin, ymax = -100, 100
zmin, zmax = 0, 2200

domain = 0.12
max_dxy = 10
scaling = 1.3
dxdy = 0.0025
dz = 100
gridx_cpu = create_adaptive_grid_1d(xmin=xmin, xmax=xmax, dx_fine=dxdy, growth_factor=scaling, dx_max=max_dxy,
    boreholes=boreholes, backend=CPU(), Float_used=Float_used, direction=:x);
gridy_cpu = create_adaptive_grid_1d(xmin=ymin, xmax=ymax, dx_fine=dxdy, growth_factor=scaling, dx_max=max_dxy,
    boreholes=boreholes, backend=CPU(), Float_used=Float_used, direction=:y);
gridz_cpu = create_uniform_gridz_with_borehole_depths(zmin=zmin, zmax=zmax, dz=dz, boreholes=boreholes, backend=CPU())

# initial_temperature_Li is not GPU compatible, so we create the initial condition on the CPU and then adapt it to the GPU
ϕ = adapt(backend, [initial_temperature_Li(z) for z in gridz_cpu, y in gridy_cpu, x in gridx_cpu]);
# adapt grids to backend
gridx = adapt(backend, gridx_cpu)
gridy = adapt(backend, gridy_cpu)
gridz = adapt(backend, gridz_cpu)

# define inlet model
# T_inlet = T_outlet - Q / C
Q = 200 * 1e3  # W
C = borehole.ṁ * 4174   # J/(kg·K) for water
inlet_model = HeatExchangerInlet{Float_used}(Q / C)

# create cache
cache = create_cache(backend=backend, gridx=gridx, gridy=gridy, gridz=gridz, materials=materials_stratified, boreholes=boreholes, inlet_model=inlet_model);

tspan = (0, 3600 * 24 * 120); # 120 days
prob = ODEProblem(rhs_diffusion_z!, ϕ, tspan, cache);


n_saves = 2  # save initial and final state only
saveat = range(tspan..., n_saves)
callback, saved_values = get_callback(
    saveat=saveat,
    print_every_n=10000,
    write_to_jld=false # no checkpoints
)

Δt = 80
t_elapsed = @elapsed solve(
    prob,
    ROCK2(max_stages=100, eigen_est=eigen_estimator),
    save_everystep=false,
    callback=callback,
    adaptive=false,
    dt=Δt,
    maxiters=Int(1e10)
)

println("Simulation completed in $t_elapsed seconds.")

# create a cache which lives on the CPU for later analysis
cache_cpu = create_cache(backend=CPU(), gridx=gridx, gridy=gridy, gridz=gridz, materials=materials_stratified, boreholes=boreholes, inlet_model=inlet_model);

@save joinpath(simulation_data_dir(), "Li_et_al_simulation_data.jld2") saved_values Δt cache_cpu t_elapsed

# end of simulation part



# analyze and plot results


gridx_cpu = cache_cpu.gridx
gridy_cpu = cache_cpu.gridy
gridz_cpu = cache_cpu.gridz
xc = cache_cpu.boreholes[1].xc
yc = cache_cpu.boreholes[1].yc
T = saved_values.saveval[end];  # final temperature field
t = saved_values.t[end] / 3600 / 24 # final time
# Extract profiles at different depths
depths = [500.0, 1000.0, 1500.0, 2000.0]
colors =[1, 2, 3, 4]


p = plot(xlabel="Radial distance from borehole [m]",
    ylabel="Temperature of rock layers [°C]",
    # title="Temperature profiles around borehole after $(Int(round(t))) days of operation\n(Li et al. 2021 numerical data vs. our simulation)",
    xlims=(-2, 60),
    ylims=(20, 60),
    legend=:bottomright,
    box=:on,
     gridlinewidth=2,
      xtickfontsize=14, ytickfontsize=14,
    xguidefontsize=16, yguidefontsize=16,
    ztickfontsize=14, zguidefontsize=16,
    legendfontsize=12,
    size=(700, 500),
    dpi=300)
plot!(p, [], [], label="GeothermalWells.jl", color=:black, linewidth=2)
scatter!(p, [], [], label="Numerical data (Li et al.)", color=:black, markersize=3, markerstrokewidth=0, markershape = :diamond,)

# Plot simulation results
for (i, depth) in enumerate(depths)
    r, T_profile = extract_x_profile(T, gridx_cpu, gridy_cpu, gridz_cpu, depth, xc, yc)
    plot!(p, r, T_profile,
        label="",
        color=colors[i],
        linewidth=3)
end

# Add numerical data as scatter points
for i in 1:length(depths)
    r_exp, T_exp = data_li(i)
    scatter!(p, r_exp, T_exp,
        label="",
        color=colors[i],
        markersize=3,
        markerstrokewidth=0.1,
         markershape = :diamond,)
end

# Add depth annotations on the plot
annotate!(p, 50, 58.12403198166667 - 2, text("2000m", color=p.series_list[6][:linecolor], :left, 15))
annotate!(p, 50, 51.01548958 - 2, text("1500m", color=p.series_list[5][:linecolor], :left, 15))
annotate!(p, 50, 42.473182396666665 - 2, text("1000m", color=p.series_list[4][:linecolor], :left, 15))
annotate!(p, 50, 29.563582989999997  + 2, text("500m",  color=p.series_list[3][:linecolor], :left, 15))


@info savefig(p, joinpath(plots_dir(), "Li_et_al_temperature_radial_profile.pdf"))

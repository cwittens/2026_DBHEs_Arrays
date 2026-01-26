# Reproduction of Li et al. (2021) validation case
# Reference: Heat extraction model and characteristics of coaxial deep borehole heat exchanger
# https://doi.org/10.1016/j.renene.2021.01.036
# Compares simulation results with numerical data from Figure 5 at 500m, 1000m, 1500m, and 2000m depths

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using GeothermalWells
using OrdinaryDiffEqStabilizedRK: ODEProblem, solve, ROCK2
using KernelAbstractions: CPU, adapt
using Statistics: mean
using JLD2: @save, @load
using CUDA: CUDABackend
using Plots

# =============================================================================
# Setup directories
# =============================================================================
plots_dir() = joinpath(@__DIR__, "plots")
!isdir(plots_dir()) && mkdir(plots_dir())

simulation_data_dir() = joinpath(@__DIR__, "simulation_data")
!isdir(simulation_data_dir()) && mkdir(simulation_data_dir())

# =============================================================================
# Backend and precision
# =============================================================================
# Choose backend: CPU() for testing, or CUDABackend()/ROCBackend() for GPU
backend = CUDABackend()
Float_used = Float64

# =============================================================================
# Material properties (stratified rock)
# =============================================================================
# Four distinct rock layers with increasing thermal conductivity and varying heat capacity
materials_stratified = StratifiedMaterialProperties{4,Float_used}(
    (1.8, 2.6, 3.5, 5.3),                                # k_rock [W/(m·K)]
    (1780 * 1379, 2030 * 1450, 1510 * 1300, 2600 * 878), # rho_c_rock [J/(m³·K)]
    (500.0, 1000.0, 1500.0, 2000.0),                     # layer_depths [m]
    0.618,                   # k_water [W/(m·K)]
    4.166e6,                 # rho_c_water [J/(m³·K)]
    41.0,                    # k_steel [W/(m·K)]
    7850 * 475,              # rho_c_steel [J/(m³·K)] (estimated - not specified in paper)
    0.4,                     # k_insulating [W/(m·K)]
    1.955e6,                 # rho_c_insulating [J/(m³·K)] (estimated - not specified in paper)
    1.5,                     # k_backfill [W/(m·K)]
    1.76e6                   # rho_c_backfill [J/(m³·K)] (estimated - not specified in paper)
)

# =============================================================================
# Borehole geometry
# =============================================================================
# Deep coaxial borehole heat exchanger (2000m depth) based on Li et al. Table 2
# No insulation on outer pipe
borehole = Borehole{Float_used}(
    0.0,                      # xc [m]
    0.0,                      # yc [m]
    2000.0,                   # h - borehole depth [m]
    0.0511,                   # r_inner - inner pipe radius [m] (calculated from (125-2×11.4)/2)
    0.0114,                   # t_inner - inner pipe wall thickness [m] (11.4 mm)
    0.0885,                   # r_outer - outer pipe inner radius [m] (calculated from (193.7-2×8.33)/2)
    0.00833,                  # t_outer - outer pipe wall thickness [m] (8.33 mm)
    0.115,                    # r_backfill - outer radius [m] (115 mm - estimated - not specified in paper)
    42 * 998.2 / 3600,        # ṁ - mass flow rate [kg/s] (42 m³/h converted)
    0.0                       # insulation_depth [m] (no insulation)
)

boreholes = (borehole,)

# =============================================================================
# Initial temperature field
# =============================================================================
# The thermal gradient in Li et al. is not fully specified, so we reconstruct it
# by interpolating between temperatures extracted from their Figure 5 data

# Load reference data from Li et al. Figure 5
r500, T500 = data_li(1)      # 500m depth data
r1000, T1000 = data_li(2)    # 1000m depth data
r1500, T1500 = data_li(3)    # 1500m depth data
r2000, T2000 = data_li(4)    # 2000m depth data

# Extract undisturbed rock temperatures (far from borehole)
# After 120 days, temperature far from borehole equals initial temperature
n = 5  # average last n data points
T_at_500 = mean(T500[end-n:end])
T_at_1000 = mean(T1000[end-n:end])
T_at_1500 = mean(T1500[end-n:end])
T_at_2000 = mean(T2000[end-n:end])

# Create piecewise linear interpolation function for initial temperature profile
@inline function initial_temperature_Li_open(z, T_at_500, T_at_1000, T_at_1500, T_at_2000)
    # Known depths and temperatures from Figure 5
    z_data = [500.0, 1000.0, 1500.0, 2000.0]
    T_data = [T_at_500, T_at_1000, T_at_1500, T_at_2000]

    # Extrapolate upward using first two points
    if z <= z_data[1]
        slope = (T_data[2] - T_data[1]) / (z_data[2] - z_data[1])
        return T_data[1] + slope * (z - z_data[1])
    end

    # Extrapolate downward using last two points
    if z >= z_data[end]
        slope = (T_data[end] - T_data[end-1]) / (z_data[end] - z_data[end-1])
        return T_data[end] + slope * (z - z_data[end])
    end

    # Interpolate between known points
    for i in 1:length(z_data)-1
        if z >= z_data[i] && z <= z_data[i+1]
            slope = (T_data[i+1] - T_data[i]) / (z_data[i+1] - z_data[i])
            return T_data[i] + slope * (z - z_data[i])
        end
    end
end

# Closure for type stability and speed
initial_temperature_Li(z) = Float_used(initial_temperature_Li_open(z, T_at_500, T_at_1000, T_at_1500, T_at_2000))

# =============================================================================
# Grid setup
# =============================================================================
# Domain boundaries
xmin, xmax = -100.0, 100.0
ymin, ymax = -100.0, 100.0
zmin, zmax = 0.0, 2200.0

# Grid parameters
dx_fine = 0.0025      # fine spacing near borehole [m]
growth_factor = 1.3   # geometric growth rate
dx_max = 10.0         # maximum spacing far from borehole [m]
dz = 100.0            # vertical spacing [m]

# Create adaptive grids (fine near borehole, coarse far away)
gridx_cpu = create_adaptive_grid_1d(
    xmin=xmin, xmax=xmax,
    dx_fine=dx_fine, growth_factor=growth_factor, dx_max=dx_max,
    boreholes=boreholes, backend=CPU(), Float_used=Float_used, direction=:x
)

gridy_cpu = create_adaptive_grid_1d(
    xmin=ymin, xmax=ymax,
    dx_fine=dx_fine, growth_factor=growth_factor, dx_max=dx_max,
    boreholes=boreholes, backend=CPU(), Float_used=Float_used, direction=:y
)

gridz_cpu = create_uniform_gridz_with_borehole_depths(
    zmin=zmin, zmax=zmax, dz=dz,
    boreholes=boreholes, backend=CPU()
)


# =============================================================================
# Initial condition
# =============================================================================
# Hack: initial_temperature_Li function is not written to be GPU-compatible, so we
# create the  initial condition on CPU and then adapt to backend
ϕ = adapt(backend, [initial_temperature_Li(z) for z in gridz_cpu, y in gridy_cpu, x in gridx_cpu])

# Adapt grids to backend
gridx = adapt(backend, gridx_cpu)
gridy = adapt(backend, gridy_cpu)
gridz = adapt(backend, gridz_cpu)

# =============================================================================
# Inlet model
# =============================================================================
# Heat exchanger inlet: T_inlet = T_outlet - Q / C
# where Q is heat extraction rate and C is heat capacity flow rate
Q = 200e3                      # heat extraction rate [W]
C = borehole.ṁ * 4174          # heat capacity flow rate [J/(s·K)]
inlet_model = HeatExchangerInlet{Float_used}(Q / C)

# =============================================================================
# Create simulation cache
# =============================================================================
cache = create_cache(
    backend=backend,
    gridx=gridx,
    gridy=gridy,
    gridz=gridz,
    materials=materials_stratified,
    boreholes=boreholes,
    inlet_model=inlet_model
)

# =============================================================================
# Time integration
# =============================================================================
tspan = (0, 3600 * 24 * 120)  # 120 days [s]

prob = ODEProblem(rhs_diffusion_z!, ϕ, tspan, cache)

# Save solution at initial and final times only
n_saves = 2
saveat = range(tspan..., n_saves)
callback, saved_values = get_callback(
    saveat=saveat,
    print_every_n=10000,
    write_to_jld=false
)

# Time step and solver
Δt = 80 # [s]

t_elapsed = @elapsed solve(
    prob,
    ROCK2(max_stages=100, eigen_est=eigen_estimator),
    save_everystep=false,
    callback=callback,
    adaptive=false,
    dt=Δt,
    maxiters=Int(1e10)
)

println("Simulation completed in $(round(t_elapsed, digits=2))s")

# =============================================================================
# Save simulation data
# =============================================================================
# Create CPU cache for analysis (makes it easier to not have to deal with GPU arrays)
cache_cpu = create_cache(
    backend=CPU(),
    gridx=gridx,
    gridy=gridy,
    gridz=gridz,
    materials=materials_stratified,
    boreholes=boreholes,
    inlet_model=inlet_model
)

@save joinpath(simulation_data_dir(), "Li_et_al_simulation_data.jld2") saved_values Δt cache_cpu t_elapsed

println("Simulation data saved to $(simulation_data_dir())")

# =============================================================================
# Analysis and visualization
# =============================================================================

# If you dont want to un the simulation yourself, load previously saved simulation
# @load joinpath(simulation_data_dir(), "Li_et_al_simulation_data.jld2") saved_values Δt cache_cpu t_elapsed

# Extract grids and final temperature field
gridx_cpu = cache_cpu.gridx
gridy_cpu = cache_cpu.gridy
gridz_cpu = cache_cpu.gridz
xc = cache_cpu.boreholes[1].xc
yc = cache_cpu.boreholes[1].yc

T = saved_values.saveval[end]           # final temperature field [°C]
t = saved_values.t[end] / 3600 / 24    # final time [days]

# Depths at which to compare with Li et al. data (matches their Figure 5)
depths = [500.0, 1000.0, 1500.0, 2000.0]  # [m]
colors = [1, 2, 3, 4]

# =============================================================================
# Create comparison plot
# =============================================================================
p = plot(
    xlabel="Radial distance from borehole [m]",
    ylabel="Temperature of rock layers [°C]",
    xlims=(-2, 60),
    ylims=(20, 60),
    legend=:bottomright,
    box=:on,
    gridlinewidth=2,
    xtickfontsize=14, ytickfontsize=14,
    xguidefontsize=16, yguidefontsize=16,
    legendfontsize=12,
    size=(700, 500),
    dpi=300
)

# Add legend entries (plotted first with empty data)
plot!(p, [], [], label="GeothermalWells.jl", color=:black, linewidth=2)
scatter!(p, [], [], label="Numerical data (Li et al.)",
    color=:black, markersize=3, markerstrokewidth=0, markershape=:diamond)

# Plot simulation results for each depth
for (i, depth) in enumerate(depths)
    # Extract radial temperature profile at this depth
    r, T_profile = GeothermalWells.extract_x_profile(T, gridx_cpu, gridy_cpu, gridz_cpu, depth, xc, yc)
    plot!(p, r, T_profile,
        label="",
        color=colors[i],
        linewidth=3)
end

# Add Li et al. numerical data as scatter points
for i in 1:length(depths)
    r_exp, T_exp = data_li(i)
    scatter!(p, r_exp, T_exp,
        label="",
        color=colors[i],
        markersize=3,
        markerstrokewidth=0.1,
        markershape=:diamond)
end

# Add depth annotations to identify each curve
# Positions chosen to avoid overlap with curves
annotate!(p, 50, 58.12403198166667 - 2, text("2000m", color=p.series_list[6][:linecolor], :left, 15))
annotate!(p, 50, 51.01548958 - 2, text("1500m", color=p.series_list[5][:linecolor], :left, 15))
annotate!(p, 50, 42.473182396666665 - 2, text("1000m", color=p.series_list[4][:linecolor], :left, 15))
annotate!(p, 50, 29.563582989999997 + 2, text("500m", color=p.series_list[3][:linecolor], :left, 15))

# Save figure
savefig(p, joinpath(plots_dir(), "Li_et_al_temperature_radial_profile.pdf"))
println("Plot saved to $(plots_dir())")
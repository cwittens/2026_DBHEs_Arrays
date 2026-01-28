# Reproduction of Brown et al. (2023)
# Reference: Investigating scalability of deep borehole heat exchangers: 
#            Numerical modelling of arrays with varied modes of operation
# https://doi.org/10.1016/j.renene.2022.11.100
# Compares simulation results with numerical data from Figures 4b and 4d

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using GeothermalWells
using OrdinaryDiffEqStabilizedRK: ODEProblem, solve, ROCK2
using KernelAbstractions: CPU, adapt
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
# Material properties (homogeneous rock)
# =============================================================================
# Material properties from Brown et al. Table 1
# Using single homogeneous layer since paper uses weighted average
materials = HomogenousMaterialProperties{Float_used}(
    2.55,                    # k_rock - thermal conductivity of rock [W/(m·K)]
    2.356e6,                 # rho_c_rock - volumetric heat capacity [J/(m³·K)]
    0.59,                    # k_water [W/(m·K)]
    998 * 4179,              # rho_c_water [J/(m³·K)]
    52.7,                    # k_steel (outer pipe) [W/(m·K)]
    7850 * 475,              # rho_c_steel [J/(m³·K)] (estimated - not specified in paper)
    0.45,                    # k_insulating (polyethylene inner pipe) [W/(m·K)]
    941 * 1800,              # rho_c_insulating [J/(m³·K)] (estimated - not specified in paper)
    1.05,                    # k_backfill (grout) [W/(m·K)]
    995 * 1200               # rho_c_backfill (grout) [J/(m³·K)]
)

# =============================================================================
# Borehole geometry
# =============================================================================
# Newcastle borehole geometry from Brown et al. Table 1
# Geometry calculations:
# - Inner pipe outer diameter = 0.1005 m → outer radius = 0.05025 m
# - Inner pipe thickness = 0.00688 m → inner radius = 0.05025 - 0.00688 = 0.04337 m
# - Borehole diameter = 0.216 m → borehole radius = 0.108 m
# - Grout thickness = 0.01905 m → r_outer + t_outer = 0.108 - 0.01905 = 0.08895 m
# - Outer pipe thickness = 0.0081 m → r_outer = 0.08895 - 0.0081 = 0.08085 m

borehole = Borehole{Float_used}(
    0.0,                     # xc [m]
    0.0,                     # yc [m]
    922.0,                   # h - borehole depth [m]
    0.04337,                 # r_inner - inner radius of central pipe [m]
    0.00688,                 # t_inner - thickness of inner pipe wall [m]
    0.08085,                 # r_outer - inner radius of outer pipe [m]
    0.0081,                  # t_outer - thickness of outer pipe wall [m]
    0.108,                   # r_backfill - borehole radius [m]
    998.0 * 0.005,           # ṁ - mass flow rate [kg/s] (5 L/s converted)
    0.0                      # insulation_depth [m] (no insulation mentioned in paper)
)

boreholes = (borehole,)

# =============================================================================
# Grid setup
# =============================================================================
# Domain boundaries
xmin, xmax = -100.0, 100.0
ymin, ymax = -100.0, 100.0
zmin, zmax = 0.0, 1100.0

# Grid parameters
dx_fine = 0.003       # fine spacing near borehole [m]
growth_factor = 1.3   # geometric growth rate
dx_max = 10           # maximum spacing far from borehole [m]
dz = 30               # vertical spacing [m]

# Create adaptive grids (fine near borehole, coarse far away)
gridx = create_adaptive_grid_1d(
    xmin=xmin, xmax=xmax,
    dx_fine=dx_fine, growth_factor=growth_factor, dx_max=dx_max,
    boreholes=boreholes, backend=backend, Float_used=Float_used, direction=:x
)

gridy = create_adaptive_grid_1d(
    xmin=ymin, xmax=ymax,
    dx_fine=dx_fine, growth_factor=growth_factor, dx_max=dx_max,
    boreholes=boreholes, backend=backend, Float_used=Float_used, direction=:y
)

gridz = create_uniform_gridz_with_borehole_depths(
    zmin=zmin, zmax=zmax, dz=dz,
    boreholes=boreholes, backend=backend
)

# =============================================================================
# Initial condition
# =============================================================================
# Linear thermal gradient from Brown et al.
# T(z) = T_surface + gradient * z
ϕ = initial_condition_thermal_gradient(
    backend, Float_used, gridx, gridy, gridz;
    T_surface=9.0,      # surface temperature [°C]
    gradient=0.0334     # thermal gradient [K/m]
)


# =============================================================================
# Inlet model
# =============================================================================
# Heat exchanger inlet from Brown et al.
# P_DBHE = 50 kW, ṁ = 4.99 kg/s, c_water = 4179 J/(kg·K)
# ΔT = P / (ṁ * c) = 50000 / (4.99 * 4179) ≈ 2.398 K
Q = 50e3                             # heat extraction rate [W]
c_water = 4179.0                     # specific heat of water [J/(kg·K)]
inlet_model = HeatExchangerInlet{Float_used}(Q / (borehole.ṁ * c_water))

# =============================================================================
# Create simulation cache
# =============================================================================
cache = create_cache(
    backend=backend,
    gridx=gridx,
    gridy=gridy,
    gridz=gridz,
    materials=materials,
    boreholes=boreholes,
    inlet_model=inlet_model
)

# =============================================================================
# Time integration
# =============================================================================
tspan = (0, 3600 * 24 * 365 * 20)  # 20 years [s]

prob = ODEProblem(rhs_diffusion_z!, ϕ, tspan, cache)

# Save solution at initial and final times only
n_saves = 2
saveat = range(tspan..., n_saves)
callback, saved_values = get_simulation_callback(
    saveat=saveat,
    print_every_n=100_000,
    write_to_jld=false
)

# Time step and solver
Δt = 80  # [s]

t_elapsed = @elapsed solve(
    prob,
    ROCK2(max_stages=100, eigen_est=eigen_estimator),
    save_everystep=false,
    callback=callback,
    adaptive=false,
    dt=Δt,
    maxiters=Int(1e10)
)

println("Simulation completed in $(round(t_elapsed / 3600, digits=2)) hours")

# =============================================================================
# Save simulation data
# =============================================================================
# Create CPU cache for analysis (makes it easier to not have to deal with GPU arrays)
cache_cpu = create_cache(
    backend=CPU(),
    gridx=gridx,
    gridy=gridy,
    gridz=gridz,
    materials=materials,
    boreholes=boreholes,
    inlet_model=inlet_model
)

@save joinpath(simulation_data_dir(), "Brown_et_al_simulation_data.jld2") saved_values Δt cache_cpu t_elapsed

println("Simulation data saved to $(simulation_data_dir())")


# =============================================================================
# Analysis and visualization
# =============================================================================

# If you dont want to un the simulation yourself, you can load previously saved simulation
# @load joinpath(simulation_data_dir(), "Brown_et_al_simulation_data.jld2") saved_values Δt cache_cpu t_elapsed


# =============================================================================
# Visualization: Figure 4b - Temperature profiles along borehole depth
# =============================================================================

# Load reference data from Brown et al. Figure 4b
(z_beier, T_beier), (z_brown, T_brown) = data_brown_single_well_b()

# Extract temperatures along borehole depth from simulation
T_inner, T_outer, gridz_adjusted = GeothermalWells.get_temperatures_along_z_single_well(saved_values.saveval[end], cache_cpu)

# Create plot comparing inlet/outlet temperatures
p1 = scatter(
    T_inner, -gridz_adjusted,
    label="GeothermalWells.jl",
    color=1,
    legendfontsize=8,
    ylabel="Depth [m]",
    xlabel="Temperature [°C]",
    markersize=4,
    xtickfontsize=14, ytickfontsize=14,
    xguidefontsize=16, yguidefontsize=16,
    ztickfontsize=14, zguidefontsize=16,
    box=:on,
    gridlinewidth=2,
    size=(600, 400),
    dpi=300,
)
scatter!(p1, T_outer, -gridz_adjusted, label="", color=1, markersize=4,)

# Add Brown et al. numerical data
scatter!(p1, T_brown, -z_brown,
    label="Numerical data (Brown et al.)",
    color=2,
    markershape=:diamond,
    markersize=4,
)

# Add annotations to identify inlet and outlet curves
color = :black
annotate!(p1, [(2.85, -300, text("Inlets", color, 14)),
    (4.7, -260, text("Outlets", color, 14))])

# Add arrows pointing to the curves
plot!(p1, [2.4, 3.2], [-300, -500], arrow=true, color=color, linewidth=2, label="")
plot!(p1, [4.32, 4.12], [-300, -80], arrow=true, color=color, linewidth=2, label="")

# Manually set y-axis ticks (since yflip affects arrow directions)
yticks!(p1, -0:-200:-800, string.(0:200:800))

# Save figure
savefig(p1, joinpath(plots_dir(), "Brown_et_al_temperature_in_outlet.pdf"))
println("Figure 4b saved to $(plots_dir())")

# =============================================================================
# Visualization: Figure 4d - Radial temperature profiles at different depths
# =============================================================================

# Extract grids and final temperature field
gridx_cpu = cache_cpu.gridx
gridy_cpu = cache_cpu.gridy
gridz_cpu = cache_cpu.gridz
xc = cache_cpu.boreholes[1].xc
yc = cache_cpu.boreholes[1].yc

T = saved_values.saveval[end];             # final temperature field [°C]
t = saved_values.t[end] / 3600 / 24 / 365  # final time [years]

println("Plotting temperature profiles after $(round(t, digits=1)) years")

# Depths at which to compare with Brown et al. data (matches their Figure 4d)
depths = [300.0, 600.0, 920.0]  # [m]
colors = [1, 2, 3]

# Create comparison plot
p2 = plot(
    xlabel="Radial distance from borehole [m]",
    ylabel="Temperature of rock layers [°C]",
    xlims=(-100, 100),
    ylims=(-3, 42),
    box=:on,
    gridlinewidth=2,
    legend=:bottomleft,
    legendfontsize=10,
    markersize=4,
    xtickfontsize=14, ytickfontsize=14,
    xguidefontsize=16, yguidefontsize=16,
    ztickfontsize=14, zguidefontsize=16,
    left_margin=1Plots.mm,
    right_margin=3Plots.mm,
    size=(700, 500),
    dpi=300
)

# Add legend entries (plotted first with empty data)
plot!(p2, [], [], label="GeothermalWells.jl", color=:black, linewidth=2)
scatter!(p2, [], [], label="Numerical data (Brown et al.)",
    color=:black, markersize=3, markerstrokewidth=0, markershape=:diamond)

# Plot simulation results for each depth
for (i, depth) in enumerate(depths)
    # Extract radial temperature profile at this depth
    # Note: Using full_profile=true to get both sides of borehole
    r, T_profile = GeothermalWells.extract_x_profile(T, gridx_cpu, gridy_cpu, gridz_cpu, depth, xc, yc, true)
    plot!(p2, r, T_profile,
        label="",
        color=colors[i],
        linewidth=3)
end

# Add Brown et al. numerical data as scatter points
for i in 1:length(depths)
    r_num, T_num = data_brown_single_well_c(i)
    scatter!(p2, r_num, T_num,
        label="",
        color=colors[i],
        markersize=3,
        markershape=:diamond)
end

# Add depth annotations to identify each curve
# Positions chosen to avoid overlap with curves
annotate!(p2, 65, 37.2, text("920m", color=p2.series_list[5][:linecolor], :left, 15))
annotate!(p2, 65, 26.5, text("600m", color=p2.series_list[4][:linecolor], :left, 15))
annotate!(p2, 65, 16.5, text("300m", color=p2.series_list[3][:linecolor], :left, 15))

# Save figure
savefig(p2, joinpath(plots_dir(), "Brown_et_al_temperature_rock.pdf"))
println("Figure 4d saved to $(plots_dir())")

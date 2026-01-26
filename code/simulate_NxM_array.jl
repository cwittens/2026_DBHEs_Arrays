# Example: N×M borehole array simulation
# General template for simulating multiple deep borehole heat exchangers arranged in a grid
# This example uses custom parameters (not from literature) and can be adapted for various configurations

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
# Borehole array configuration
# =============================================================================
# Define array layout by specifying x and y coordinates for borehole centers
# Example configurations (uncomment desired configuration):

# 5×5 array with 30m spacing
XC = [-60.0, -30.0, 0.0, 30.0, 60.0]
YC = [-60.0, -30.0, 0.0, 30.0, 60.0]

# Alternative configurations:
# 3×3 array with 60m spacing
# XC = [-60.0, 0.0, 60.0]
# YC = [-60.0, 0.0, 60.0]

# Single borehole
# XC = [0.0]
# YC = [0.0]

println("Configuring $(length(XC))×$(length(YC)) borehole array")


# =============================================================================
# Borehole geometry
# =============================================================================
# Create array of boreholes at specified coordinates
# All boreholes have identical geometry (can be customized if needed)
boreholes = tuple(
    (Borehole{Float_used}(
        xc,                      # xc [m]
        yc,                      # yc [m]
        2000.0,                  # h - borehole depth [m]
        0.0511,                  # r_inner - inner pipe radius [m]
        0.0114,                  # t_inner - inner pipe wall thickness [m]
        0.0885,                  # r_outer - outer pipe inner radius [m]
        0.00833,                 # t_outer - outer pipe wall thickness [m]
        0.115,                   # r_backfill - outer radius [m]
        42 * 998.2 / 3600,       # ṁ - mass flow rate [kg/s] (42 m³/h converted)
        0.0                      # insulation_depth [m]
    ) for xc in XC, yc in YC)...
)

# =============================================================================
# Material properties (stratified rock)
# =============================================================================
# Four distinct rock layers with increasing thermal conductivity
# Similar setup to Li et al. (2021) but with custom thermal gradient
materials = StratifiedMaterialProperties{4,Float_used}(
    (1.8, 2.6, 3.5, 5.3),                                # k_rock [W/(m·K)]
    (1780 * 1379, 2030 * 1450, 1510 * 1300, 2600 * 878), # rho_c_rock [J/(m³·K)]
    (500.0, 1000.0, 1500.0, 2000.0),                     # layer_depths [m]
    0.618,                   # k_water [W/(m·K)]
    4.166e6,                 # rho_c_water [J/(m³·K)]
    41.0,                    # k_steel [W/(m·K)]
    7850 * 475,              # rho_c_steel [J/(m³·K)] (estimated)
    0.4,                     # k_insulating [W/(m·K)]
    1.955e6,                 # rho_c_insulating [J/(m³·K)] (estimated)
    1.5,                     # k_backfill [W/(m·K)]
    1.76e6                   # rho_c_backfill [J/(m³·K)] (estimated)
)


# =============================================================================
# Grid setup
# =============================================================================
# Domain boundaries (adjusted to encompass entire array with buffer)
xmin, xmax = -160.0, 160.0
ymin, ymax = -160.0, 160.0
zmin, zmax = 0.0, 2200.0

# Grid parameters
dx_fine = 0.0025      # fine spacing near each borehole [m]
growth_factor = 1.3   # geometric growth rate
dx_max = 10.0         # maximum spacing far from boreholes [m]
dz = 100.0            # vertical spacing [m]

# Create adaptive grids (fine near all boreholes, coarse elsewhere)
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
# Visualize grid structure (optional)
# =============================================================================
# Create overview plot of the grid and borehole positions
# p_full = plot_grid(
#     adapt(CPU(), gridx), adapt(CPU(), gridy),
#     size=(400, 400),
#     boreholes=boreholes,
#     legend=false
# )
# savefig(p_full, joinpath(plots_dir(), "array_grid_overview.png"))

# # Create zoomed-in plot showing fine grid detail near boreholes
# lims = 0.17
# p_zoom = plot_grid(
#     adapt(CPU(), gridx), adapt(CPU(), gridy),
#     size=(400, 400),
#     boreholes=boreholes,
#     legend=false,
#     xlims=(-lims, lims),
#     ylims=(-lims, lims),
#     annotate=false
# )


# =============================================================================
# Initial condition
# =============================================================================
# Linear thermal gradient
# T(z) = T_surface + gradient * z
ϕ = initial_condition_thermal_gradient(
    backend, Float_used, gridx, gridy, gridz;
    T_surface=16.65,    # surface temperature [°C]
    gradient=0.02       # thermal gradient [K/m]
)

# =============================================================================
# Inlet model
# =============================================================================
# Heat exchanger inlet with fixed temperature difference
# All boreholes operate with same ΔT between outlet and inlet
ΔT = 4.1  # temperature difference [K]
inlet_model = HeatExchangerInlet{Float_used}(ΔT)

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
tspan = (0.0, 3600 * 24)# 1 days or * 365)  # 1 year [s]

prob = ODEProblem(rhs_diffusion_z!, ϕ, tspan, cache)

# Save solution at initial and final times only
n_saves = 2
saveat = range(tspan..., n_saves)
callback, saved_values = get_callback(
    saveat=saveat,
    print_every_n=2000,
    write_to_jld=false
)

# Time step and solver
Δt = 80  # [s]

println("Simulating $(length(XC))×$(length(YC)) array with Δt = $(Δt)s for $(tspan[2] / (3600 * 24)) days")

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

filename = "array_$(length(XC))x$(length(YC))_simulation_data.jld2"
@save joinpath(simulation_data_dir(), filename) saved_values Δt cache_cpu t_elapsed

println("Simulation data saved to $(simulation_data_dir())")

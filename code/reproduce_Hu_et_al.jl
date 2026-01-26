# Reproduction of Hu et al. (2020) validation case
# Reference: Numerical modeling of a coaxial borehole heat exchanger to exploit 
#            geothermal energy from abandoned petroleum wells in Hinton, Alberta
# https://doi.org/10.1016/j.renene.2019.09.141
# Compares simulation results with numerical data from Figure 7 at 1, 5, 10, and 25 years

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
# Material properties from Hu et al. Table 1 (at T = 30°C)
materials = HomogenousMaterialProperties{Float_used}(
    2.8811795365701656,      # k_rock - thermal conductivity [W/(m·K)]
    2.1663426199999996e6,    # rho_c_rock - volumetric heat capacity [J/(m³·K)]
    0.6,                     # k_water [W/(m·K)]
    4.17686808e6,            # rho_c_water [J/(m³·K)]
    44.5,                    # k_steel (outer pipe) [W/(m·K)]
    3.728750e6,              # rho_c_steel [J/(m³·K)]
    0.26,                    # k_insulating (inner pipe) [W/(m·K)]
    1.955e6,                 # rho_c_insulating [J/(m³·K)]
    1.0,                     # k_backfill [W/(m·K)] (not used - no backfill region)
    1.0                      # rho_c_backfill [J/(m³·K)] (not used)
)

# =============================================================================
# Borehole geometry
# =============================================================================
# Hinton borehole geometry from Hu et al.
# Deep coaxial borehole (3500m depth) with insulation on inner pipe down to 1000m
borehole = Borehole{Float_used}(
    0.0,                     # xc [m]
    0.0,                     # yc [m]
    3500.0,                  # h - borehole depth [m]
    0.0381,                  # r_inner - inner pipe radius [m]
    0.01,                    # t_inner - inner pipe wall thickness [m]
    0.0889,                  # r_outer - outer pipe inner radius [m]
    0.01,                    # t_outer - outer pipe wall thickness [m]
    0.0989,                  # r_backfill - outer radius (r_outer + t_outer) [m]
    10.0,                    # ṁ - mass flow rate [kg/s]
    1000.0                   # insulation_depth [m]
)

boreholes = (borehole,)

# =============================================================================
# Grid setup
# =============================================================================
# Domain boundaries
xmin, xmax = -100.0, 100.0
ymin, ymax = -100.0, 100.0
zmin, zmax = 0.0, 3700.0

# Grid parameters
dx_fine = 0.0025      # fine spacing near borehole [m]
growth_factor = 1.3   # geometric growth rate
dx_max = 10.0         # maximum spacing far from borehole [m]
dz = 100.0            # vertical spacing [m]

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
# Linear thermal gradient from Hu et al.
# T(z) = T_surface + gradient * z
ϕ = initial_condition_thermal_gradient(
    backend, Float_used, gridx, gridy, gridz;
    T_surface=2.29,     # surface temperature [°C]
    gradient=0.035      # thermal gradient [K/m]
)

# =============================================================================
# Inlet model
# =============================================================================
# Constant inlet temperature of 20°C
inlet_model = ConstantInlet{Float_used}(20.0)

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
tspan = (0, 3600 * 24 * 365 * 25)  # 25 years [s]

prob = ODEProblem(rhs_diffusion_z!, ϕ, tspan, cache)

# Save solution at 0, 1, 5, 10, and 25 years
saveat = [3600 * 24 * 365 * year for year in [0,1,5,10,25]]
callback, saved_values = get_callback(
    saveat=saveat,
    print_every_n=100_000,
    write_to_jld=false
)

# Time step and solver
Δt = 80  # [s]

println("Simulating with Δt = $(Δt)s for 25 years")

t_simulation = @elapsed solve(
    prob,
    ROCK2(max_stages=100, eigen_est=eigen_estimator),
    save_everystep=false,
    callback=callback,
    adaptive=false,
    dt=Δt,
    maxiters=Int(1e10)
)

println("Simulation completed in $(round(t_simulation / 3600, digits=2)) hours")

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

@save joinpath(simulation_data_dir(), "Hu_et_al_simulation_data.jld2") saved_values Δt cache_cpu t_simulation

println("Simulation data saved to $(simulation_data_dir())")

# =============================================================================
# Analysis and visualization
# =============================================================================

# If you don't want to run the simulation yourself, you can load previously saved simulation
# @load joinpath(simulation_data_dir(), "Hu_et_al_simulation_data.jld2") saved_values Δt cache_cpu t_simulation
# raw"\Users\colli\OneDrive - JGU\13. Semester\GeoThermal\geo-playground\code\GeothermalWells.jl\data\Hu_et_al\simulation_result_Hu_25y.jld2"

# Set default plot styling
default(
    grid=true,
    box=:on,
    size=(600, 400),
    dpi=300,
    linewidth=3,
    gridlinewidth=2,
    markersize=4,
    markerstrokewidth=0.1,
    xtickfontsize=14,
    ytickfontsize=14,
    xguidefontsize=16,
    yguidefontsize=16,
    legendfontsize=12
)

# Extract grids from cache
gridx_cpu = cache_cpu.gridx
gridy_cpu = cache_cpu.gridy
gridz_cpu = cache_cpu.gridz

# =============================================================================
# Visualization: Figure 7 - Temperature profiles at 1, 5, 10, and 25 years
# =============================================================================

# Create individual plots for each time point
# Indices correspond to: 1 year, 5 years, 10 years, 25 years
time_indices = [2, 6, 11, 26]
# time_indices = [2,3,4,5]
plots_array = []

for idx in time_indices
    # Extract temperatures along borehole depth
    T_inner, T_outer, gridz_adjusted = GeothermalWells.get_temperatures_along_z_single_well(
        saved_values.saveval[idx], cache_cpu
    )
    
    year = Int(saved_values.t[idx] / (3600 * 24 * 365))
    
    # Create temperature profile plot (inlet down, outlet up)
    p = scatter(
        vcat(T_inner, reverse(T_outer)),
        vcat(-gridz_adjusted, reverse(-gridz_adjusted)),
        label="GeothermalWells.jl",
        ylabel="Depth [m]",
        xlabel="Temperature [°C]",
        xlims=(19.519284893437614, 36.545016559411344),
        legend=(idx == 3),  # Only show legend on one plot
        title="after $year year" * (year == 1 ? " " : "s")
    )
    
    # Manually set y-axis ticks (since yflip affects arrow directions)
    yticks!(p, 0:-1000:-3000, string.(0:1000:3000))
    
    # Add Hu et al. numerical data
    depth, temperature = data_hu(year)
    scatter!(p, temperature, -depth,
        label="Numerical data (Hu et al.)",
        color=2,
        markershape=:diamond)
    
    # Add inlet/outlet annotations with arrows
    color = :black
    if year == 1
        annotate!(p, [(26.7, -1600, text("Inlet", color, 14)),
                      (31.3, -1500, text("Outlet", color, 14))])
        plot!(p, [25, 27], [-1550, -1950], arrow=true, color=color, linewidth=2, label="")
        plot!(p, [33, 32.2], [-1750, -1050], arrow=true, color=color, linewidth=2, label="")
    elseif year == 5
        annotate!(p, [(26.2, -1600, text("Inlet", color, 14)),
                      (29.8, -1500, text("Outlet", color, 14))])
        plot!(p, [24.5, 26.5], [-1550, -1950], arrow=true, color=color, linewidth=2, label="")
        plot!(p, [31.5, 30.7], [-1750, -1050], arrow=true, color=color, linewidth=2, label="")
    elseif year == 10
        annotate!(p, [(25.7, -1600, text("Inlet", color, 14)),
                      (29.3, -1500, text("Outlet", color, 14))])
        plot!(p, [24, 26], [-1550, -1950], arrow=true, color=color, linewidth=2, label="")
        plot!(p, [31, 30.2], [-1750, -1050], arrow=true, color=color, linewidth=2, label="")
    elseif year == 25
        annotate!(p, [(25.7, -1600, text("Inlet", color, 14)),
                      (28.6, -1500, text("Outlet", color, 14))])
        plot!(p, [23.7, 25.7], [-1550, -1950], arrow=true, color=color, linewidth=2, label="")
        plot!(p, [30.3, 29.5], [-1750, -1050], arrow=true, color=color, linewidth=2, label="")
    end
    
    push!(plots_array, p)
end

# Adjust labels for combined plot layout
# Remove redundant y-labels and x-labels from interior plots
yticks!(plots_array[2], [0, -1000, -2000, -3000], [""])
yticks!(plots_array[4], [0, -1000, -2000, -3000], [""])
ylabel!(plots_array[2], "")
ylabel!(plots_array[4], "")
xticks!(plots_array[1], [20, 25, 30, 35], [""])
xlabel!(plots_array[1], "")
xticks!(plots_array[2], [20, 25, 30, 35], [""])
xlabel!(plots_array[2], "")

# Combine into 2x2 layout
p_combined = plot(
    plots_array...,
    layout=(2, 2),
    size=(1200, 850),
    dpi=300,
    left_margin=6Plots.mm,
    top_margin=2Plots.mm,
    bottom_margin=8Plots.mm
)

# Save figure
savefig(p_combined, joinpath(plots_dir(), "Hu_et_al_in_outlet.pdf"))
savefig(p_combined, joinpath(plots_dir(), "Hu_et_al_in_outlet.png"))
println("Figure 7 saved to $(plots_dir())")
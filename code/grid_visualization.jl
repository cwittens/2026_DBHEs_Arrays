using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using GeothermalWells
using KernelAbstractions: CPU
using Plots

# =============================================================================
# Setup directories
# =============================================================================
plots_dir() = joinpath(@__DIR__, "plots")
!isdir(plots_dir()) && mkdir(plots_dir())

# =============================================================================
# Borehole array configuration
# =============================================================================
XC = [-60, 0, 60]
YC = [-60, 0, 60]

boreholes = tuple(
    (Borehole{Float64}(
        xc,            # xc
        yc,            # yc
        2000,             # h 
        0.0511,           # r_inner 
        0.0114,           # t_inner 
        0.0885,           # r_outer 
        0.00833,          # t_outer 
        0.115,            # r_backfill
        42 * 998.2 / 3600,# flow rate
        0  # no insulation of the outer pipe
    ) for xc in XC, yc in YC)...
)

# =============================================================================
# Grid setup
# =============================================================================
xmin, xmax = -160, 160
ymin, ymax = -160, 160
zmin, zmax = 0, 2200
dxdy = 0.0025
max_dxy = 10
scaling = 1.3

gridx = create_adaptive_grid_1d(xmin=xmin, xmax=xmax, dx_fine=dxdy, growth_factor=scaling, dx_max=max_dxy,
    boreholes=boreholes, backend=CPU(), Float_used=Float64, direction=:x);
gridy = create_adaptive_grid_1d(xmin=ymin, xmax=ymax, dx_fine=dxdy, growth_factor=scaling, dx_max=max_dxy,
    boreholes=boreholes, backend=CPU(), Float_used=Float64, direction=:y);

# =============================================================================
# Create plots
# =============================================================================

p1 = plot_grid(gridx, gridy, size=(400, 400), boreholes=boreholes, legend =false)
lims = 0.17
p2 = plot_grid(gridx, gridy, size=(400, 400), boreholes=boreholes, legend =:bottomleft, xlims=(-lims+XC[1], lims+XC[1]), ylims=(-lims+YC[1], lims+YC[1]), annotate=false)


p_combined = plot(p1, p2, layout=(1, 2), size=(800, 400), dpi=300, left_margin=3Plots.mm,) 
@info  savefig(p_combined, joinpath(plots_dir(), "Grid_overview.pdf"))

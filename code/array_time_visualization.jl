# Simulation time scaling analysis for borehole arrays
# 
# NOTE: The data below comes from running simulate_NxM_array.jl with different array configurations.
# Each configuration (1×1, 3×1, 5×1, 3×3, 3×4, 4×4, 4×5, 5×5) was run separately
# and the simulation time was recorded.
# 
# IMPORTANT: For the larger arrays (3×4, 4×4, 4×5, 5×5), only 36 days were simulated
# (instead of a full year) to save computation time. These times are then extrapolated
# to 365 days for comparison.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Plots

# =============================================================================
# Setup directories
# =============================================================================
plots_dir() = joinpath(@__DIR__, "plots")
!isdir(plots_dir()) && mkdir(plots_dir())

# Raw simulation times [seconds] and the duration simulated
# Small arrays were simulated for 360 days
# Large arrays were simulated for only 36 days (then extrapolated)
times = [
    1795.970100116 * 365 / 360,  # 1×1
    4044.209330266 * 365 / 360,  # 3×1
    10188.398726748 * 365 / 360,  # 5×1
    23541.658622322 * 365 / 360,  # 3×3
    4111.3172313466 * 365 / 36,  # 3×4 (extrapolated from 36 days)
    6890.191763069 * 365 / 36,  # 4×4 (extrapolated from 36 days)
    10520.052246 * 365 / 36,  # 4×5 (extrapolated from 36 days)
    15892.395824521 * 365 / 36,  # 5×5 (extrapolated from 36 days)
]

array_types = ["1×1", "3×1", "5×1", "3×3", "3×4", "4×4", "4×5", "5×5"]
n_wells = [1, 3, 5, 9, 12, 16, 20, 25]

# Plot total simulation time
p = plot(n_wells, times ./ 3600,
    xlabel="Number of Wells",
    ylabel="Time [hours]",
    label="Simulation Time vs Number of Wells",
    xlims=(0.01, 26),
    legend=:topleft,
    marker=:o, markersize=6,
    grid=true, box=:on,
    size=(600, 400), dpi=300,
    linewidth=3, gridlinewidth=2,
    xtickfontsize=14, ytickfontsize=14,
    xguidefontsize=16, yguidefontsize=16,
    ztickfontsize=14, zguidefontsize=16,
    legendfontsize=12,
)

# Add annotations for array types
for i in 1:4
    annotate!(p, n_wells[i], times[i] / 3600 + 1.5, text(array_types[i], 13, :bottom))
end
for i in 5:8
    annotate!(p, n_wells[i], times[i] / 3600 - 1.5, text(array_types[i], 13, :top))
end


# Calculate normalized time (time per well)
normalized_times = times ./ n_wells

p2 = plot(n_wells, normalized_times ./ 3600,
    xlabel="Number of Wells",
    ylabel="Time per well [hours]",
    label="Normalized Simulation Time\nvs Number of Wells",
    xlims=(0.01, 26),
    legend=:topleft,
    marker=:o, markersize=6,
    grid=true, box=:on,
    size=(600, 400), dpi=300,
    linewidth=3, gridlinewidth=2,
    xtickfontsize=14, ytickfontsize=14,
    xguidefontsize=16, yguidefontsize=16,
    ztickfontsize=14, zguidefontsize=16,
    legendfontsize=12,
)

# Add annotations for array types
for i in 1:length(n_wells)
    y_offset = i <= 3 ? 0.05 : -0.05
    annotate!(p2, n_wells[i], normalized_times[i] / 3600 + y_offset,
        text(array_types[i], 13, i <= 3 ? :bottom : :top))
end


# Combined plot
p_combined = plot(p, p2, layout=(1, 2), size=(1200, 400),
    left_margin=7Plots.mm, right_margin=0Plots.mm,
    top_margin=5Plots.mm, bottom_margin=8Plots.mm)

@info  savefig(p_combined, joinpath(plots_dir(), "simulation_time_vs_number_of_wells_combined.pdf"))

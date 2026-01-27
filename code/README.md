# Reproducing Numerical Results

This document provides instructions for reproducing the numerical experiments from the paper.

> [!IMPORTANT]
> This repository is intended solely for reproducing the results from the paper.
> If you want to use the simulation framework for your own research or applications,
> it would probably make more sense to use the [GeothermalWells.jl](https://github.com/cwittens/GeothermalWells.jl) package directly.


## Installing Julia

Download Julia from [https://julialang.org/downloads/](https://julialang.org/downloads/) and follow the instructions there.


## Running the Simulations

Each script in this directory is self-contained and can be run independently. The scripts automatically handle package installation and environment setup by using the `Project.toml` and `Manifest.toml` files in this directory. These files specify the exact package versions used in the paper, ensuring reproducibility. Make sure they are in the same folder as your Julia script.

> [!NOTE]
> The simulations in this repository were run using Julia 1.12. It is recommended to use Julia 1.12 as well.

### Quick Start

1. Open a terminal and navigate to the `code` directory
2. Start Julia:
   ```bash
   julia
   ```
3. Run any script by including it:
   ```julia
   include("reproduce_Li_et_al.jl")
   ```

Alternatively, run directly from the command line:
```bash
julia reproduce_Li_et_al.jl
```

### First Run: Package Installation

On the first run, each script will:
1. Activate the local environment
2. Install all required packages
3. Precompile the packages

This initial setup may take several minutes but only needs to happen once. Subsequent runs will be much faster.


## Compute Backend

Running these simulations requires a GPU. By default, the scripts use CUDA (NVIDIA GPUs). The `backend` variable near the top of each script controls the compute backend:

```julia
# NVIDIA GPUs (default, recommended)
backend = CUDABackend()

# AMD GPUs
using AMDGPU
backend = ROCBackend()

# CPU (for testing/debugging only)
backend = CPU()
```

To use AMD GPUs, you need to install the AMDGPU.jl package:
```julia
using Pkg
Pkg.add("AMDGPU")
```

> [!WARNING]
> Running on CPU is possible but will take significantly longer (hours instead of minutes). CPU execution is intended only for testing and debugging on small problem sizes.


## Output Directories

Each script automatically creates two directories:
- `plots/` - Contains generated figures (PDF format)
- `simulation_data/` - Contains saved simulation results (JLD2 format)


## Skipping the Simulation

If you just want to look at the results without running the simulations yourself, you can skip the simulation step and load the results from the production runs (included in the `simulation_data/` directory). Each script contains a commented line that demonstrates this:

```julia
# If you don't want to run the simulation yourself, you can load previously saved data:
# @load joinpath(simulation_data_dir(), "Li_et_al_simulation_data.jld2") saved_values Δt cache_cpu t_elapsed
```

Simply uncomment this line and comment out the simulation code to use the pre-computed results from the paper.


## Reproducing Specific Figures

### Section 4.1: Convergence Studies (Figures 2 & 3)

```julia
include("convergence_diffusion.jl")
```

Generates:
- `plots/spatial_convergence.pdf` - Spatial convergence study
- `plots/temporal_convergence.pdf` - Temporal convergence study

**Approximate execution time:** TODO


### Section 4.2: Li et al. Validation - Case Study 1 (Figure 4)

```julia
include("reproduce_Li_et_al.jl")
```

Reproduces the comparison with Li et al. (2021) - 120 days of simulation.

Generates:
- `plots/Li_et_al_temperature_radial_profile.pdf` - Radial temperature profiles at 500m, 1000m, 1500m, and 2000m depths

**Reference:** Li et al. (2021), "Heat extraction model and characteristics of coaxial deep borehole heat exchanger", https://doi.org/10.1016/j.renene.2021.01.036

**Approximate execution time:** TODO


### Section 4.3: Hu et al. Validation - Case Study 2 (Figure 5)

```julia
include("reproduce_Hu_et_al.jl")
```

Reproduces the comparison with Hu et al. (2020) - 25 years of simulation.

Generates:
- `plots/Hu_et_al_in_outlet.pdf` - Temperature profiles along borehole depth at 1, 5, 10, and 25 years

**Reference:** Hu et al. (2020), "Numerical modeling of a coaxial borehole heat exchanger to exploit geothermal energy from abandoned petroleum wells in Hinton, Alberta", https://doi.org/10.1016/j.renene.2019.09.141

**Approximate execution time:** TODO


### Section 4.4: Brown et al. Validation - Case Study 3 (Figures 6 & 7)

```julia
include("reproduce_Brown_et_al.jl")
```

Reproduces the comparison with Brown et al. (2023) - 20 years of simulation.

Generates:
- `plots/Brown_et_al_temperature_in_outlet.pdf` - Temperature profiles along borehole depth
- `plots/Brown_et_al_temperature_rock.pdf` - Radial temperature profiles at 300m, 600m, and 920m depths

**Reference:** Brown et al. (2023), "Investigating scalability of deep borehole heat exchangers: Numerical modelling of arrays with varied modes of operation", https://doi.org/10.1016/j.renene.2022.11.100

**Approximate execution time:** TODO


### Section 3.1: Grid Visualization (Figure 1)

```julia
include("grid_visualization.jl")
```

Generates:
- `plots/Grid_overview.pdf` - Visualization of the adaptive grid structure for a 3x3 well array

**Approximate execution time:** A few seconds (CPU only)


### Section 5: Well Array Simulation

```julia
include("simulate_NxM_array.jl")
```

Demonstrates simulation of N x M well arrays. The array configuration can be modified by changing the `XC` and `YC` variables at the top of the script.

**Approximate execution time:** TODO


## Troubleshooting

### CUDA not available
If you see errors about CUDA not being available:
1. Ensure you have an NVIDIA GPU with CUDA support
2. Install the latest NVIDIA drivers
3. Julia's CUDA.jl should automatically download the appropriate CUDA toolkit


### Slow performance on CPU
CPU execution is intended only for testing and debugging. Full simulations should be run on GPU for reasonable execution times. A single-well 120-day simulation that takes ~10 minutes on an H200 GPU would take many hours on CPU. If you do not have access to a GPU, consider using the pre-computed results from the `simulation_data/` directory instead.

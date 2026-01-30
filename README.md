# A Full Three-Dimensional GPU-Accelerated Model for Deep Borehole Heat Exchangers (DBHEs) Enabling Simulation of Well Arrays

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18405149.svg)](https://doi.org/10.5281/zenodo.18405149)

This repository contains information and code to reproduce the results presented
in the article
```bibtex
@inproceedings{wittenstein2026dbhe,
  author    = {Wittenstein, Collin and Lujan, Emmanuel and Inglis, Andrew and Metcalfe, Robert and Edelman, Alan and Ranocha, Hendrik},
  title     = {A Full Three-Dimensional {GPU}-Accelerated Model for Deep Borehole Heat Exchangers ({DBHEs}) Enabling Simulation of Well Arrays},
  booktitle = {Proceedings of the 51st Stanford Geothermal Workshop},
  year      = {2026},
  address   = {Stanford, California},
  publisher = {Stanford University},
  url       = {https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2026/Wittenstein.pdf}
}
```

**[Read the paper](https://pangea.stanford.edu/ERE/db/GeoConf/papers/SGW/2026/Wittenstein.pdf)**

If you find these results useful, please cite the article mentioned above. If you
use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{wittenstein2026dbheRepro,
  title={Reproducibility repository for
         "A Full Three-Dimensional {GPU}-Accelerated Model for Deep Borehole Heat Exchangers ({DBHEs}) Enabling Simulation of Well Arrays"},
  author={Wittenstein, Collin and Lujan, Emmanuel and Inglis, Andrew and Metcalfe, Robert and Edelman, Alan and Ranocha, Hendrik},
  year={2026},
  howpublished={\url{https://github.com/cwittens/2026_DBHEs_Arrays}},
  doi={10.5281/zenodo.18405149}
}
```

> [!IMPORTANT]
> This repository is intended solely for reproducing the results from the paper.
> If you want to use the simulation framework for your own research or applications,
> it would probably make more sense to use the [GeothermalWells.jl](https://github.com/cwittens/GeothermalWells.jl) package directly.


## Abstract

Deep borehole heat exchangers (DBHEs) present significant computational challenges due to their multi-scale geometry and long
operational timescales. We present a GPU-accelerated three-dimensional model that makes well array simulations
computationally tractable through an operator splitting strategy tailored to the problem's physics. The method separates vertical
diffusion (stabilized explicit Runge–Kutta–Chebyshev), horizontal diffusion (alternating direction implicit), and advection (semiLagrangian), achieving near-unconditional stability with high efficiency. We validate against three published models using
different numerical approaches, showing excellent to good agreement. The vendor-agnostic Julia implementation enables full
three-dimensional simulation of multi-well arrays on a single GPU, opening new possibilities for systematic design optimization
and long-term performance assessment of geothermal well systems. The implementation is released as the open-source Julia
package [GeothermalWells.jl](https://github.com/cwittens/GeothermalWells.jl).



## Numerical experiments

To reproduce the numerical experiments presented in this article, you need
to install [Julia](https://julialang.org/). See the detailed installation
instructions in the [`code/README.md`](code/README.md).

The numerical experiments presented in this article were performed using
Julia v1.12.3

First, you need to download this repository, e.g., by cloning it with `git`
or by downloading an archive via the GitHub interface. Then, you need to start
Julia in the `code` directory of this repository and follow the instructions
described in the [`code/README.md`](code/README.md) file therein.


## Authors

- [Collin Wittenstein](https://cwittens.github.io/) (Massachusetts Institute of Technology & Johannes Gutenberg University Mainz)
- [Emmanuel Lujan](https://www.emmanuellujan.com/) (Massachusetts Institute of Technology)
- Andrew Inglis (Massachusetts Institute of Technology)
- Robert Metcalfe (Massachusetts Institute of Technology)
- [Alan Edelman](https://math.mit.edu/~edelman/) (Massachusetts Institute of Technology)
- [Hendrik Ranocha](https://ranocha.de/) (Johannes Gutenberg University Mainz)


## License

The code in this repository is published under the MIT license, see the
`LICENSE` file.


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!

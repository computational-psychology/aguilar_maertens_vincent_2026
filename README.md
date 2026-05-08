# Simulation code of manuscript "A god’s eye view: Using simulation to disambiguate transducer and noise"

This repository contains the code and siimulated data used to reproduce
the results and visualizations presented in the manuscript.


## Requirements

Code written and tested in python. It requires the following standard
scientific computing libraries:

- numpy
- matplotlib, seaborn
- scipy

It also requires our [python wrapper for MLDS](https://github.com/computational-psychology/mlds).
This wrapps the R package MLDS, and allows us to call it from python.
[Follow the instructions there](https://github.com/computational-psychology/mlds) to install it.

Note: all code is in the format of python files (.py). Scripts are
annotated in such as way that they can also be run as jupyter notebooks.
For that you need to have jupyter notebook and the 
[jupytext extension](https://jupytext.readthedocs.io/en/latest/).  


## Repository content

### Folder `simulation`

Contains code to run the simulations. The main scripts are:

- [1_mlds_recovers_under_additive_noise.py](simulation/1_mlds_recovers_under_additive_noise.py) simulations of transducers with *additive* noise. Reproduces Fig. 5 and 9A
- [2_mlds_recovers_under_multiplicative_noise.py](simulation/2_mlds_recovers_under_multiplicative_noise.py) simulations of transducers with multiplicative noise. Reproduces Fig. 6 and 9B

Both scripts run the simulations and saves data to folder `data`.
It takes several hours in a laptop computer. If data already exist (as it is the case in this repository), then scripts loads the data and skip the simulations.

- [plot_cases_equal_sensitivity.py](simulation/plot_cases_equal_sensitivity.py) reads simulated data and reproduces Fig. 8
- [plot-RMSE.py](simulation/plot-RMSE.py) reads simulate data reproduces Fig. 7

The code also uses the following utility functions

- [tranducers.py](simulation/transducers.py): transducer definitions
- [simulate_mlds_experiment.py](simulation/simulate_mlds_experiment.py): observer model and estimation of scales
- [plotting.py](simulation/plotting.py): utilities for plotting, color palettes, etc. 
- [utils.py](simulation/utils.py): miscelaneous computations

### Folder `data`

Simulated MLDS experiments and transducer parameters used.

## Authors and Acknowledgments

Code written by [Guillermo Aguilar](guillermo.aguilar@mail.tu-berlin.de)
and [Joris Vincent](joris.vincent@tu-berlin.de). For questions or issues,
please contact them.

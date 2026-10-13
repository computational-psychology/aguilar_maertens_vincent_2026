# Using simulation to disambiguate transducer and noise

Simulation code and data to reproduce the results and figures of the manuscript:

> Aguilar, G., Maertens, M., & Vincent, J. (2026). *Using simulation to disambiguate transducer and noise.*
> Journal of Vision (accepted). DOI: to be added.

The paper is a simulation study of Maximum Likelihood Difference Scaling (MLDS):
it tests whether MLDS can recover a known transducer function from data generated
under different transducer shapes and noise types (additive and multiplicative,
and Poisson in the Supplementary Material), including cases where discrimination
performance is identical.

Repository: https://github.com/computational-psychology/aguilar_maertens_vincent_2026


### Requirements

- Python 3.12 or above.
- The libraries in [`requirements.txt`](requirements.txt): numpy, scipy,
  pandas, matplotlib, seaborn.
- Only to re-run the simulations, not needed to reproduce the figures: our
  [Python wrapper for MLDS](https://github.com/computational-psychology/mlds)
  (v1.0.1), which wraps the R package `MLDS` and requires a working R
  installation. Follow the instructions in that repository to install it.
  Results were produced with mlds 1.0.1, R 4.5.3, and the MLDS R package 0.5.1.

The scripts are plain `.py` files annotated so they can also be opened as Jupyter
notebooks via the [jupytext](https://jupytext.readthedocs.io/en/latest/) extension.


## Repository layout

```
simulation/   simulation and plotting scripts
data/         saved simulated data and ground-truth parameters
figs/         figures written by the scripts
requirements.txt
LICENSE
```


## Simulations and reproducing the figures

The simulation code lives in three main scripts inside `simulation/`:

- `1_mlds_recovers_under_additive_noise.py` for additive noise
- `2_mlds_recovers_under_multiplicative_noise.py` for multiplicative noise
- `3_mlds_recovers_under_poisson_noise.py` for Poisson noise (Supplementary Material)

The scripts run the simulation (or alternatively, read simulated data run previously),
and do the plotting, writing the figures as PDFs to `figs/`.


We provided the simulated data in `data/`, so the figures can be regenerated without the need to re-run all simulations from scratch.
These CSV files contain the results of simulated MLDS experiments
(`additive-noise.scales.10.10.100.csv`, `multiplicative-noise.scales.10.10.100.csv`,
`poisson-noise.scales.10.10.100.csv`)
as well as the parameters used in the simulations
(`additive-noise.params.csv`, `multiplicative-noise.params.csv` and
`poisson-noise.params.csv`).
The `10.10.100` suffix encodes the simulation settings
`n_stimulus_levels.n_repeats.n_simulations`, i.e. 10 stimulus levels, 10 repeats
per triad, and 100 simulated experiments per condition.



| Script | Manuscript figures |
| --- | --- |
| `1_mlds_recovers_under_additive_noise.py` | Fig. 5 (scales, additive noise), Fig. 9A (exponent recovery), Suppl. Fig. S1 (additive-noise error/bias) |
| `2_mlds_recovers_under_multiplicative_noise.py` | Fig. 6 (scales, multiplicative noise), Fig. 9B (exponent recovery), Suppl. Fig. S2 (multiplicative-noise error/bias) |
| `3_mlds_recovers_under_poisson_noise.py` | Suppl. Fig. S3 (scales, Poisson noise), Suppl. Fig. S4 (Poisson-noise error/bias), Suppl. Fig. S5 (exponent recovery) |
| `plots_cases_equal_sensitivity.py` | Fig. 4 (ground-truth transducer shapes), Fig. 8 (cases of equal sensitivity) |
| `plot-RMSE.py` | Fig. 7 (RMSE summary) |

The simulation result figures of the manuscript correspond to these files in `figs/`:

| Manuscript figure | File in `figs/` |
| --- | --- |
| Fig. 4 | `ground_truths_function_shapes.pdf` |
| Fig. 5 | `additive-noise_scales.10.10.100.pdf` |
| Fig. 6 | `multiplicative-noise_scales.10.10.100.pdf` |
| Fig. 7 | `rmse_summary.10.10.100.pdf` |
| Fig. 8 | `equal_sensitivity.10.10.100.pdf` |
| Fig. 9A | `additive-noise_paramfit-hists-boxplot.10.10.100.pdf` |
| Fig. 9B | `multiplicative-noise_paramfit-hists-boxplot.10.10.100.pdf` |
| Suppl. Fig. S1 | `additive-noise_errors.10.10.100.pdf` |
| Suppl. Fig. S2 | `multiplicative-noise_errors.10.10.100.pdf` |
| Suppl. Fig. S3 | `poisson-noise_scales.10.10.100.pdf` |
| Suppl. Fig. S4 | `poisson-noise_errors.10.10.100.pdf` |
| Suppl. Fig. S5 | `poisson-noise_paramfit-exponent-boxplot.10.10.100.pdf` |

The other files in `figs/` are additional outputs of the scripts that are not
shown in the manuscript.


There are also supporting modules, which are imported by the scripts above:

- `simulation/transducers.py` - transducer function definitions (power-law and
  logarithmic) with additive and multiplicative noise.
- `simulation/simulate_mlds_experiment.py` - the simulated observer model
  (method of triads) and MLDS scale estimation. When the decision variable is
  exactly zero (which happens only with discrete, i.e. Poisson, noise), the
  observer chooses either response with probability 0.5.
- `simulation/plotting.py` - plotting helpers and color palette.
- `simulation/utils.py` - miscellaneous numerical helpers.



## Re-running the simulations

If you delete the `.scales.*.csv` files the three scripts `1_...`, `2_...` and `3_...`
will re-run the full simulation and regenerate them.
This requires the `mlds` wrapper and **takes several hours on a laptop**.

Note: the simulations are stochastic and the random number generator is 
not seeded, so re-running will produce statistically equivalent but not identical results.
The `.csv` files committed here are the exact ones used for the manuscript.


## License

Released under the GNU General Public License v3.0 (GPLv3).
See [LICENSE](LICENSE).


## Authors and contact

Code written by [Guillermo Aguilar](mailto:guillermo.aguilar@mail.tu-berlin.de)
and [Joris Vincent](mailto:joris.vincent@tu-berlin.de). For questions or issues,
please open an issue on the repository or contact the authors.

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     notebook_metadata_filter: jupytext,-kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
# ---

# %% [markdown]
# # Plotting MLDS results for cases of equal sensitivity 

# %% [markdown]
# In this notebook we plot only the combinations of transducer and noise that 
# produce equal sensitivity (defined as a constant Weber fraction k).
# It reproduces Fig. 8 of the manuscript.

# %%
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import plotting
import seaborn as sns
from plotting import plt

from transducers import logarithmic, power_func
from utils import normalize_to_range

reds = list(sns.color_palette('Reds', 7))

palette = {'log': '#636363',
          '0.33': reds[0],
          '0.5': reds[1],
          '0.75': reds[2],
          '1.0': reds[3],
          '1.5': reds[4],
          '2.0': reds[5],
          '3.0': reds[6],
          }

sns.set_context('paper')

n_samples_mlds = 10
n_repeats = 10
n_simulations = 100

suffix = f"{n_samples_mlds}.{n_repeats}.{n_simulations}"

# %%
# Define stimulus domain
s_min = 0.00000001
s_max = 1.0

s_min_log = 1
s_max_log = 100

s = np.linspace(s_min, s_max, 1000)
s_log =  np.linspace(s_min_log, s_max_log, 1000)

# mapping from s to s_log 
_, s_slope, s_intercept = normalize_to_range(s, s_min_log, s_max_log, return_factors=True)


# calculating normalizing coefficients for log transducer, to make it fall in the range 0-1
y, slope, intercept = normalize_to_range(np.log(s_log), ymin=0, ymax=1.0, return_factors=True)


# %%
def rescale_s(s, s_slope, s_intercept):
    # for the logarithmic transducer we need to rescale the stimulus value to the range 1-100
    s_log = s_slope*s + s_intercept
    return s_log

def log_rescaled(s_log, slope, intercept):
    # take the log, and rescaling so that the range of mu is in [0, 1]
    mu = slope * np.log(s_log) + intercept
    return np.abs(mu)


# %%
### Reading simulated data - additive noise case
scales_additive = pd.read_csv(
    Path().resolve().parent / "data" / "simulations" / f"additive-noise.scales.{suffix}.csv",
    low_memory=False,
)
scales_additive = scales_additive.rename(columns={'sigma': 'amount_noise'})
scales_additive['type_noise'] = 'additive'
scales_additive

# %%
params_additive = pd.read_csv(
    Path().resolve().parent / "data" / "simulations" / "additive-noise.params.csv",
)
params_additive = params_additive.rename(columns={'sigma': 'amount_noise'})
params_additive['type_noise'] = 'additive'
params_additive

# %%
### Reading simulated data - multiplicative case
scales_multiplicative = pd.read_csv(
    Path().resolve().parent / "data" / "simulations" / f"multiplicative-noise.scales.{suffix}.csv",
    low_memory=False,
)
scales_multiplicative = scales_multiplicative.rename(columns={'g': 'amount_noise'})
scales_multiplicative['type_noise'] = 'multiplicative'
scales_multiplicative


# %%
params_multiplicative = pd.read_csv(
    Path().resolve().parent / "data" / "simulations" / "multiplicative-noise.params.csv",
)
params_multiplicative = params_multiplicative.rename(columns={'g': 'amount_noise'})
params_multiplicative['type_noise'] = 'multiplicative'
params_multiplicative

# %%
scales = pd.concat((scales_additive, scales_multiplicative))
params = pd.concat((params_additive, params_multiplicative))


# %%
params['weber_fraction'] = params['weber_fraction'].round(3)
params

# %%
params.groupby('weber_fraction').count()

# %%
# remove the values where the estimated noise is stored, we do not needed it at the moment
scales = scales[scales['stimulus']!='sigma_decision']
scales['stimulus'] = scales['stimulus'].astype(float)

scales['amount_noise'] = scales['amount_noise'].astype(float)


# %%
scales


# %%
# we merge with parameters dataframe to add the weber fraction column
scales = scales.merge(params, on=['trans_type', 'type_noise', 'amount_noise'])
scales

# %%
# calculate mean of those selected cases
mean_scales = (
    scales.groupby(["type_noise", "trans_type", "amount_noise", "weber_fraction", "stimulus"])
    .agg(
        mean_scale=("scale", "mean"),
        CI_low=("scale", lambda x: np.quantile(x, 0.025)),
        CI_high=("scale", lambda x: np.quantile(x, 0.975)),
    )
    .reset_index()
)

# %%
mean_scales


# %%
s_samples_mlds = np.sort(np.array(mean_scales['stimulus'].unique()))

# %%
# select only cases for specific Weber fraction
weber_fractions = [0.05, 0.075, 0.1, 0.2]
mask = mean_scales['weber_fraction'].isin(weber_fractions)
mean_scales = mean_scales[mask]

# %%
mean_scales.groupby(['type_noise', 'trans_type', 'weber_fraction'])['trans_type'].count()


# %%
def trans_add(s, trans_type):
    if trans_type == 'log':
        s_log = rescale_s(s, s_slope, s_intercept)
        mu = log_rescaled(s_log, slope, intercept)
    else:
        exponent = float(trans_type)
        mu = power_func(s, exponent=exponent)
        
    return mu

def trans_mult(s, trans_type):
    if trans_type == 'log':
        s_log = rescale_s(s, s_slope, s_intercept)
        mu = log_rescaled(s_log, slope, intercept)
    else:
        exponent = float(trans_type)
        mu = power_func(s, exponent=exponent)
        
    return mu



# %%
grid_scales = sns.FacetGrid(
    data=mean_scales, col="weber_fraction", 
    #col_wrap=2,
    #row='type_noise',
    hue='trans_type', palette=palette,
    height=3.5, margin_titles=True, aspect=0.9,
    sharey=True, sharex=True,
    ylim=(-0.15, 1.15),
    xlim=(-0.15, 1.15),
)

def plot_panel(stimulus, mean_scale, CI_low, CI_high, trans_type, type_noise, **kwargs):
    
    ax = plt.gca()
    
    CIs = np.vstack([CI_low, CI_high]).T
    
    assert(len(trans_type.unique())==1)
    assert(len(type_noise.unique())==1)

    type_noise = type_noise.unique()[0]
    trans_type = trans_type.unique()[0]
    
    if type_noise=='additive':
        mu = trans_add(s, trans_type)
    elif type_noise == 'multiplicative':
        mu = trans_mult(s, trans_type)
        
    ax.errorbar(stimulus, mean_scale, yerr=[mean_scale - CIs[:, 0], CIs[:, 1] - mean_scale],
                color=palette[trans_type], fmt="o")
    
    plt.plot(s, mu, '-', c=palette[trans_type], alpha=0.75)
    
    
grid_scales.map(plot_panel, "stimulus", "mean_scale", "CI_low", "CI_high", "trans_type", "type_noise")


grid_scales.set_titles(
    "", col_template=r"Weber fraction ($k$) = {col_name}"
)
grid_scales.set_axis_labels(
    y_var="", x_var="", clear_inner=True
)

grid_scales.fig.supylabel(r"Perceptual scale", x=0)
grid_scales.fig.supxlabel(r"Stimulus intensity $(s)$")

plt.savefig(
    Path().resolve().parent / "figs" / f"equal_sensitivity.{suffix}.pdf",
    bbox_inches="tight",
)



# %%
## visualize ground-truth transducers only

# %%
exponents = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
mu_log = trans_add(s, 'log')
ax.plot(s, mu_log, label="$log(s)$", c=palette['log'], linewidth=2)

for e in exponents:
    y = power_func(s, exponent=e)
    ax.plot(s, power_func(s, exponent=e), label=f"$s^{{{e}}}$", c=palette[str(e)], linewidth=2)


ax.legend(bbox_to_anchor=(0.9, 0.5, 0.5, 0.5))
ax.set_ylabel(r'$\mu(s)$')
ax.set_xlabel('Stimulus intensity (s)')
ax.set_xlim((-0.05, 1.05))
ax.set_ylim((-0.05, 1.05))
fig.savefig(Path().resolve().parent / "figs" / "ground_truths_function_shapes.pdf", bbox_inches="tight")

# %%
# determinining sets of parameters to be run for the case of multiplicative noise,
# starting from a set of Weber fractions and desired power law exponents
k_set = [0.05] # desired weber fractions
exponent_set = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] # desired exponents

gs =[]
for k in k_set:
    for alpha in exponent_set:
        g = k*alpha 
        g = round(g, 4)
        print(f"exponent: {alpha} --> g: {g}")
        print(f"predicted k: {g/alpha}")
        print('')
        gs.append(g)

np.unique(np.array(gs))

# %%

# %%
# for Appendix table
table = params_multiplicative[params_multiplicative['trans_type']!='log'].copy()


# %%
print(pd.pivot_table(table, values='weber_fraction', index='amount_noise', columns='trans_type', 
               #aggfunc='count')
               aggfunc='mean').to_latex(float_format=lambda x: '%.3f' % x ))

# %%

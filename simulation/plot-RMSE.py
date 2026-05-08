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
# # Plotting MLDS results - RMSE 

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
def trans(s, trans_type):
    if trans_type == 'log':
        s_log = rescale_s(s, s_slope, s_intercept)
        mu = log_rescaled(s_log, slope, intercept)
    else:
        exponent = float(trans_type)
        mu = power_func(s, exponent=exponent)
        
    return mu



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
# keeping only reported additive noise values
#sigmas = [0.05, 0.075, 0.1, 0.2]
#mask = scales_additive['amount_noise'].isin(sigmas)
#scales_additive = scales_additive[mask]
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
scales


# %%
# we merge with parameters dataframe to add the weber fraction column
scales = scales.merge(params, on=['trans_type', 'type_noise', 'amount_noise'])
scales

# %%
# remove the values where the estimated noise is stored, we do not needed it at the moment
scales = scales[scales['stimulus']!='sigma_decision'].copy()
scales['stimulus'] = scales['stimulus'].astype(float)

scales['amount_noise'] = scales['amount_noise'].astype(float)

# %%
scales = scales[scales['amount_noise']!=0.0248] # omit here a rounding error
#scales = scales[scales['amount_noise']!=0.0375]

# %%
scales['amount_noise'].unique()


# %%
# calculate RMSE

# %%
def calculate_error(scales):
    # Ground truth values at sampled stimulus levels
    scales["ground_truth"] = scales.apply(
        func=lambda row: trans(s=row["stimulus"], 
                               trans_type=row["trans_type"]),
        axis="columns",
    )

    # Error
    scales["error"] = scales["scale"] - scales["ground_truth"]
    return scales



# %%
scales = calculate_error(scales)

# %%
rmses = (
    scales.groupby(["trans_type", "amount_noise", "type_noise", "simulation_id"])
    .agg(
        rmse=("error", lambda x: np.sqrt(np.mean(x**2))),
    )
    .reset_index(drop=False)
)
rmses

# %%
rmses_agg = (
    rmses.groupby(["trans_type", "type_noise", "amount_noise"])
    .agg(
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
    )
    .reset_index(drop=False)
    .sort_values(
        by=["trans_type", "type_noise", "amount_noise"],
        ascending=[True, True, True],
    )
)

rmses_agg

# %%
# double checking that numbers agree with Fig. 4 and 5
rmses_agg[(rmses_agg['type_noise']=='multiplicative') & 
          (rmses_agg['trans_type']=='0.5') &
          (rmses_agg['amount_noise']==0.15)][['mean_rmse', 
                                             'std_rmse']]*100  # in percent

# I doubled check a sample of panels from fig. 4 and 5, all numbers matched.

# %%
rmses_agg[(rmses_agg['type_noise']=='additive')].groupby('amount_noise')['mean_rmse'].min()*100

# %%
rmses_agg[(rmses_agg['type_noise']=='additive')].groupby('amount_noise')['mean_rmse'].max()*100

# %%
#some_transducers = ['log', '0.33', '0.5', '1.0']
some_transducers = ['2.0', '3.0']
mask = rmses_agg['trans_type'].isin(some_transducers)

rmses_agg[(rmses_agg['type_noise']=='multiplicative') & mask].groupby(['amount_noise'])['mean_rmse'].min()*100

# %%
rmses_agg[(rmses_agg['type_noise']=='multiplicative') & mask].groupby(['amount_noise'])['mean_rmse'].max()*100

# %%
# visualize error

# %%
grid_scales = sns.FacetGrid(
    data=rmses, col="type_noise", 
    hue='trans_type', palette=palette,
    height=3.5, margin_titles=True, aspect=1,
    #xlim=(0, 0.41),
    #sharey=False, sharex=False,
    sharex=True, sharey=True,
    col_order = ['additive', 'multiplicative'],
)

grid_scales.map(sns.lineplot, 'amount_noise', 'rmse', err_style="bars", errorbar=("sd", 1),
               linewidth=2)
grid_scales.add_legend(title='Tranducer')
#grid_scales.set(xscale="log")
#grid_scales.set(yscale="log")

grid_scales.set_titles(
    "", col_template=r"{col_name} noise"
)
grid_scales.set_axis_labels(
    y_var="", x_var="", clear_inner=True
)
grid_scales.axes[0][0].set_xlabel(r'$\sigma$')
grid_scales.axes[0][1].set_xlabel(r'$g$')
grid_scales.axes[0][0].set_title(r'Additive noise')
grid_scales.axes[0][1].set_title(r'Multiplicative noise')

grid_scales.fig.supylabel(r"RMSE", y=0.55)
grid_scales.fig.supxlabel(r"Noise magnitude", y=-0.05)

plt.savefig(
    Path().resolve().parent / "figs" / f"rmse_summary.{suffix}.pdf",
    bbox_inches="tight",
)


# %%
from scipy.stats import linregress

# %%
## slopes for additive noise
trans_types = rmses[rmses['type_noise'] == 'additive']['trans_type'].unique()

for t in trans_types:
    c = rmses[(rmses['type_noise'] == 'additive') & (rmses['trans_type'] == t)]
    means = c.groupby('amount_noise')['rmse'].mean().reset_index()

    slope, intercept, r, p, se = linregress(x=means['amount_noise'], y=means['rmse'])
    print(f"fit for trans_type: {t} => slope : {slope:.2f}, intercept: {intercept:.2f}, r={r:.3f}")



# %%
## slopes for multiplicative noise
trans_types = rmses[rmses['type_noise'] == 'multiplicative']['trans_type'].unique()

for t in trans_types:
    c = rmses[(rmses['type_noise'] == 'multiplicative') & (rmses['trans_type'] == t)]
    means = c.groupby('amount_noise')['rmse'].mean().reset_index()

    slope, intercept, r, p, se = linregress(x=means['amount_noise'], y=means['rmse'])
    print(f"fit for trans_type: {t} => slope : {slope:.2f}, intercept: {intercept:.2f}, r={r:.3f}")

# %%

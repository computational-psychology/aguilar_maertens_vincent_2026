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
# # 1.  MLDS recovers (power-law) transducers under multiplicative noise


# %% [markdown]
# Here we show that Maximum Likelihood Dimensional Scaling (MLDS) can recover
# transducers with multiplicative noise.
# We define various transducers with different shapes as ground truths,
# and add multiplicative noise that scales with the transducer.
# For each transducer, we then simulate MLDS experiments that produce perceptual scales,
# and we compare the recovered perceptual scales to the ground truth.
# If MLDS is able to recover a ground truth transducer(s),
# then the resulting perceptual scales should be similar to that transducer.

# %%
import itertools
from functools import partial
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import plotting
import seaborn as sns
from plotting import plt
from scipy.optimize import curve_fit
import math

# requires MLDS python package(https://github.com/computational-psychology/mlds)
# A requirement of the package is also to have R and the R packages 'MLDS' and 'psyphy'
from simulate_mlds_experiment import simulate_MLDS_experiment
from transducers import power_func, power_noisy_multiplicative
from utils import normalize_to_range

# %% [markdown]
# ## Ground-truth Transducers
#
# We chose two types of transducers:
#
# - logarithmic
# - power law
#

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
plt.plot(s, y, label='log')

l = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

for e in l:
    y = power_func(s, exponent=e)
    plt.plot(s, power_func(s, exponent=e), label=f"s^{e}")

plt.legend()

# %% [markdown]
# ### Definitions of transducers and noise 
#
# Here we explore the case where the ground-truth transducer has multiplicative (Gaussian) noise.
# The noisy response of the transducer $\Psi(s)$ is a realization of a Gaussian random variable with mean
# $\mu(s)$ and variance $\sigma^2(s)$, which is a function of stimulus magnitude.
#
# $$
# \Psi(s) \sim \mathcal{N}(\mu(s), \sigma^2)
# $$
#
# The multiplicative noise is proportional to the mean by a noise scaling factor $g$.
#
# $$
# \sigma(s) = g \cdot \mu(s)
# $$
#

# %% [markdown]
# The transducer that gives rise to a Weber's law sensitivity is a power law transducer,
# defined as 
#
# $$
# \mu(s) = s^{\alpha}
# $$
# where $\alpha$ is the transducer's exponent.


# %% [markdown]
# ### Derivation
#
#
# Assuming the definition of Fisher sensitivity (as introduced recently by ...), we have that sensivitity $D(s)$
#
# $$ D(s) = \frac{(\mu(s))^{'}}{\sigma(s)} = \frac{(s^{\alpha})^{'}}{g \mu(s)} = \frac{\alpha s^{\alpha-1}}{g s^{\alpha}} = \frac{\alpha}{g s} = \frac{\alpha}{g} \cdot \frac{1}{s}
# $$
#
# This derivation shows that for a power law transducer, sensitivity is inversely related to stimulus magnitude $s$ by a constant factor ($\frac{\alpha}{g}$). This constant factor is related to the noise magnitude $g$ and also the non-linearity in the transducer expressed in the exponent $\alpha$. Thus, $\alpha$ and $g$ can trade-off to deliver the exact same Weber's law sensitivity.
#
#
# Sensitivity is defined experimentally as the inverse of a JND ($\Delta s$),
#
# $$
# D(s) = \frac{1}{\Delta s}
# $$
#
# Putting the two equations together, we have that
#
# $$
# D(s) = \frac{1}{\Delta s} =  \frac{\alpha}{g} \cdot \frac{1}{s}
# $$
#
# Rearranging we have that $\Delta s =k \cdot s$, that is, Weber's law, with a Weber fraction of magnitude 
# $$k = \frac{g}{\alpha}$$

# %% [markdown]
# ## Determining parameter values to run
#
# We want to have a coverage of different shapes of transducer functions, which translates in different exponents. We also want to have cases where sensitivity is identical, that is, an identical Weber fraction $k$. As derived above, the $k$ is proportional to the amount of multiplicative noise $g$ and inversely proportional to the exponent value $\alpha$.
#
# In the following code we set fixed a set of different values for $k$ and $\alpha$, and derive for each combination of $k$ and  $\alpha$ which $g$ will be necessary so that the equation above holds.

# %%
# determinining sets of parameters to be run for the case of multiplicative noise,
# starting from a set of Weber fractions and desired power law exponents
k_set = [0.05, 0.075, 0.1, 0.2] # desired weber fractions
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

# %% [markdown]
# ## Defining parameters
#
# Having now a set of noise values $g$ and exponents $\alpha$, we will run simulations for all possible combinations of these. Some cases will give identical Weber fractions, other not. 
#
# Later, we will see our simulations results from two angles: cases with identical Weber fraction (and different exponent and noise magnitudes), and cases with identical noise magnitude (but different exponent and weber fraction).
#
# For completion, we also include the logarithmic transducer.

# %%
# exponents
exponent_set = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] # desired exponents

# multiplicative noise magnitude
g_set = [0.0165, 0.0248, 0.025, 0.033, 0.0375, 
         0.05, 0.0562, 0.066, 0.075, 0.1, 0.1125,
         0.15, 0.2, 0.225, 0.3, 0.4, 0.45, 0.6
        ]

# parameters for power law transducer
ground_truth_params2 = pd.DataFrame(
    list(itertools.product(exponent_set, g_set)), columns=["exponent", "g"]
)

# %%
ground_truth_params2['weber_fraction'] = ground_truth_params2['g'] / ground_truth_params2['exponent']
ground_truth_params2

# %%
ground_truth_params2 = ground_truth_params2.rename(columns={'exponent': "trans_type"})

# %%
# parameters for log transducer
ground_truth_params1 = pd.DataFrame(
    list(itertools.product(["log"], g_set)), columns=["trans_type", "g"]
)

ground_truth_params = pd.concat((ground_truth_params1, ground_truth_params2))
ground_truth_params

# %%
# save parameters
ground_truth_params.to_csv(
    Path().resolve().parent / "data" / "simulations" / "multiplicative-noise.params.csv",
    index=False,
)


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
def trans_mult(s, trans_type):
    if trans_type == 'log':
        s_log = rescale_s(s, s_slope, s_intercept)
        mu = log_rescaled(s_log, slope, intercept)
    else:
        exponent = float(trans_type)
        mu = power_func(s, exponent=exponent)
        
    return mu


def plot_trans_mult(trans_type, g, **kwargs):
    trans_type = trans_type.unique()[0]
    g = g.unique()[0]
    
    mu = trans_mult(s, trans_type)
    
    ax = plotting.transducer(s=s, mu=mu, sigma=g*mu)
    ax.set_title("")


# %%
def get_weber_fraction(trans_type, g):
    if trans_type == 'log':
        return np.nan
    else:
        trans_type = float(trans_type)
        g = float(g)
        
        w = ground_truth_params[(ground_truth_params['trans_type'] == trans_type) & 
                                (ground_truth_params['g'] == g)]['weber_fraction']
        assert(len(w)==1)
        
        return   w.iloc[0]      

# %%
def add_label(trans_type, g, **kwargs):
    ax = plt.gca()

    trans_type = trans_type.unique()[0]
    g = g.unique()[0]
    
    weber_fraction = get_weber_fraction(trans_type, g)

    xyposition = (0, 0.95) if (trans_type=='2.0') or (trans_type=='3.0') else (1, 0.2)
    ha = "left" if (trans_type=='2.0') or (trans_type=='3.0') else "right"
    
    if not math.isnan(weber_fraction):
        ax.annotate(
            rf"k = {weber_fraction:.2}",
            xy=xyposition,
            fontsize=sns.plotting_context()["axes.labelsize"],
            ha=ha,
        )


# %%
grid_scales = sns.FacetGrid(
    data=ground_truth_params, col="trans_type", row="g", height=2, margin_titles=True, aspect=1.0
)

grid_scales.map(plot_trans_mult, "trans_type", "g")
grid_scales.map(add_label, "trans_type", "g")


grid_scales.set(ylim=[-0.1, 1.1])
grid_scales.set_titles("", row_template=r"$g = {row_name:.3f}$", col_template=r"{col_name}",
)
grid_scales.set_axis_labels(x_var=r"Stimulus value $s$", y_var=r"$\Psi(s)$", clear_inner=True)

#plt.show()

# %% [markdown]
# The Weber fraction is annotated in each panel. Note that there are several cases with identical Weber fraction, indicating the cases that produce identical sensitivity.


# %%
def logarithmic_noisy_multiplicative(s, g=0.0):
    """Noisy logarithmic transducer with multiplicative noise
    psi(s) = N(mu(s), sigma)
    where:
    - N(mu(s), sigma) is the normal distribution with mean mu(s) and standard deviation sigma
    - g is a scaling factor for the multiplicative noise
    - mu(s) = log(s)
    - s is the stimulus value

    Parameters
    ----------
    s : array-like
        stimulus value(s)
    sigma : float, optional
        standard deviation of the noise, by default 0.0

    Returns
    -------
    array-like
        noisy transduced/encoded stimulus value(s) into psi(s)
    """
    
    s_log = rescale_s(s, s_slope, s_intercept)
    
    mu = log_rescaled(s_log, slope, intercept)
    
    # we pass the size to make it a vector, different samples
    noise = np.random.normal(loc=0, scale=g*mu, size=mu.shape)

    psi = mu + noise

    return psi


# %%
def trans_mult_noisy(s, trans_type, g):
    """General function for noisy tranducer, either logarithmic or power-law"""
    if trans_type == 'log':
        psi = logarithmic_noisy_multiplicative(s, g=g)
    else:
        exponent = float(trans_type)
        psi = power_noisy_multiplicative(s, exponent=exponent, g=g)
    
    return psi


# %% [markdown]
# ## Simulating an MLDS experiment

# %% [markdown]
# We simulate MLDS experiments for all these transducers with multiplicative noise.
#
# We sample 11 values of $s$
# and construct all possible triads from these values.
# For each triad, we put the stimulus values through the noisy transducer
# ($\mu(s)$ and constant $\sigma$)
# to get three perceptual values ${\psi_A, \psi_B, \psi_C}$.
# Then, we simulate the response to a triad by
# calculating the intervals between the two pairs in the triad
# $\delta_{AB} = \psi_A-\psi_B, \delta_{BC} = \psi_B-\psi_C$
# and deciding which interval is larger
# $$
# R =
# \begin{cases}
# 0 & \text{if } \delta_{AB} > \delta_{BC} \\
# 1 & \text{if } \delta_{BC} > \delta_{AB}
# \end{cases}
# $$
# We repeat this process $n_{repeats} = 5$ or $10$ times per triad.
#
# To estimate the perceptual scale from these binary responses,
# we use the MLDS method as described by Knoblauch and Maloney (2012).

# %%
# stimulus sample values
n_samples_mlds = 10
s_samples_mlds = np.linspace(s_min, s_max, n_samples_mlds)

n_repeats = 10 # per triad
n_simulations = 100

suffix = f"{n_samples_mlds}.{n_repeats}.{n_simulations}"


# %%
# If simulations are already done, then read them
try:
    ### Reading simulated data
    scales = pd.read_csv(
        Path().resolve().parent / "data" / "simulations" / f"multiplicative-noise.scales.{suffix}.csv",
        low_memory=False,
    )

# otherwise run simulations
except:
    # Loop over parameters, simulate, and aggregate results
    scales = []
    for _, case in ground_truth_params.iterrows():
        # Run simulations for this transducer case
        scale = simulate_MLDS_experiment(
            stim_levels=s_samples_mlds,
            transducer=partial(trans_mult_noisy, trans_type=case["trans_type"], g=case["g"]),
            n_repeats=n_repeats,
            n_simulations=n_simulations,
            standardscale=True,
        )

        # Accumulate the estimated scales
        scale = (
            scale.reset_index(names="simulation_id", drop=False)
            .melt(var_name="stimulus", value_name="scale", id_vars=["simulation_id"])
            .assign(trans_type=case["trans_type"], g=case["g"])
        )
        if case["trans_type"]!='log':
            scale['weber_fraction'] = scale['g'] / scale['trans_type']
        scales.append(scale)

    # Concatenate and save
    scales = pd.concat(scales, ignore_index=True)
    scales['trans_type'] = scales['trans_type'].astype(str)
    
    scales = scales.reindex(
        ["trans_type", "g", "simulation_id", "stimulus", "scale"],
        axis="columns",
    )
    scales.to_csv(
        Path().resolve().parent / "data" / "simulations" / f"multiplicative-noise.scales.{suffix}.csv",
        index=False,
    )

# %%
scales

# %%
scales = scales[(scales['g']!=0.0248) & (scales['g']!=0.45)].copy() # omit here a rounding error

# %%
scales['g'].unique()

# %%
# remove the values where the estimated noise is stored, we do not needed it at the moment
scales = scales[scales['stimulus']!='sigma_decision'].copy()
scales['stimulus'] = scales['stimulus'].astype(float)

# %%
mean_scales = (
    scales.groupby(["trans_type", "g", "stimulus"])
    .agg(
        mean_scale=("scale", "mean"),
        CI_low=("scale", lambda x: np.quantile(x, 0.025)),
        CI_high=("scale", lambda x: np.quantile(x, 0.975)),
    )
    .reset_index()
)
mean_scales

# %%
show_trans_type = ['log', '0.33', '0.5', '1.0', '2.0', '3.0']
show_g = [0.0165, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6]

def filter_cases(scales, column, values):
    # showing only some values for the given column
    mask = scales[column].isin(values)
    scales = scales[mask]
    return scales


mean_scales_all = mean_scales.copy()
mean_scales = filter_cases(mean_scales, 'trans_type', show_trans_type)
mean_scales = filter_cases(mean_scales, 'g', show_g)

# %%
## plot all simulated cases
grid_scales = sns.FacetGrid(
    data=mean_scales, col="trans_type", row="g", 
    col_order = ['log', '0.33', '0.5', '1.0', '2.0', '3.0'], # exponents
    height=2, margin_titles=True, aspect=1.0,
    sharey=True,
)

grid_scales.map(plot_trans_mult, "trans_type", "g")
grid_scales.map(plotting.scale_MLDS_df, "mean_scale", "CI_low", "CI_high", ss=s_samples_mlds)

grid_scales.set(ylim=[-0.1, 1.1])
grid_scales.set_titles(
    "", 
    row_template=r"$g = {row_name:.4f}$",
    col_template=r"$\alpha = {col_name}$"
)
grid_scales.map(add_label, "trans_type", "g")
grid_scales.set_axis_labels(
    y_var="", x_var="", clear_inner=True
)

grid_scales.fig.supylabel(r"Perceptual scale $\Psi(s)$")
grid_scales.fig.supxlabel(r"Stimulus value $s$")


plt.savefig(
    Path().resolve().parent / "figs" / f"multiplicative_scales.{suffix}.pdf",
    bbox_inches="tight",
)

#plt.show()


# %% [markdown]
# Clearly the perceptual scales produced by MLDS for the three transducers are different.
# Generally, these scales match the shape of the transducer mean $\mu(s)$ in each case.
# IF MLDS would only reflect the integrated sensitivity,
# the scales would have been identical for the cases where sensitivity was identical (same Weber fraction).
# We can conclude that MLDS is able to recover the underlying transducer
# even in the presence of multiplicative noise.

# %% [markdown]
# ## Evaluating goodness of MLDS recovery
# We evaluate the goodness of the perceptual scales by MLDS in recovering the ground truth transducer.
# We do this by comparing the perceptual scale $\hat{\mu}(s)$ to the ground truth $\mu(s)$,
# and calculating the errors between the two.

# %%
# Ground truth values at sampled stimulus levels
scales["ground_truth"] = scales.apply(
    func=lambda row: trans_mult(s=row["stimulus"], 
                                trans_type=row["trans_type"]),
    axis="columns",
)

# Error
scales["error"] = scales["scale"] - scales["ground_truth"]

# %%
scales

# %%
# Rescale error to proportion of ground truth range (per group)
scales["error"] = scales.groupby(["trans_type", "g", "simulation_id"])["error"].transform(
    lambda x: x / scales.loc[x.index, "ground_truth"].max()
)

scales

# %% [markdown]
# ### Overall goodness: RMSE

# %%
rmses = (
    scales.groupby(["trans_type", "g", "simulation_id"])
    .agg(
        rmse=("error", lambda x: np.sqrt(np.mean(x**2))),
    )
    .reset_index(drop=False)
)
rmses

# %%
rmses_agg = (
    rmses.groupby(["trans_type", "g"])
    .agg(
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
    )
    .reset_index(drop=False)
    .sort_values(
        by=["trans_type", "g"],
        ascending=[True, True],
    )
)

rmses_agg.round(3)


# %% [markdown]
# We see a difference in the goodness of recovering the various transducers.
# Overall, the greater the exponent $\alpha$, the worse the recovery (higher average RMSE).
# However, to align the sensitivities, the transducers with greater $\alpha$ also have greater noise ($g$).
# Thus, it may also be that the greater noise $g$ is the reason for the greater RMSE.


# %%
def add_RMSE_label(trans_type, g, rmses, **kwargs):
    ax = plt.gca()

    trans_type = trans_type.unique()[0]
    g = g.unique()[0]

    rmses_subset = rmses[(rmses["trans_type"] ==trans_type) & (rmses["g"] ==g)]

    xyposition = (0, 0.80) if (trans_type=='2.0') or (trans_type=='3.0') else (1, 0.05)
    ha = "left" if (trans_type=='2.0') or (trans_type=='3.0') else "right"
    
    ax.annotate(
        rf"{rmses_subset['mean_rmse'].values[0]:.2%} $\pm$ {rmses_subset['std_rmse'].values[0]:.2%}",
        xy=xyposition,
        fontsize=sns.plotting_context()["axes.labelsize"],
        ha=ha,
    )


grid_scales.map(add_RMSE_label, "trans_type", "g", rmses=rmses_agg)
grid_scales.set_titles(
    "", 
    row_template=r"$g = {row_name:.2f}$",
    col_template=r"$\alpha = {col_name}$"
)
grid_scales.set_axis_labels(
    y_var="", x_var="", clear_inner=True
)

grid_scales.fig.supylabel(r"Perceptual scale $\Psi(s)$")
grid_scales.fig.supxlabel(r"Stimulus value $s$")

plt.savefig(
    Path().resolve().parent / "figs" / f"multiplicative-noise_scales.{suffix}.pdf",
    bbox_inches="tight",
)

#plt.show()

# %% [markdown]
# (EDIT THIS, just copied from additive) Under additive noise, we see that MLDS recovery performance is great (mean RMSE 1-3%),
# and consistent across the different transducer shapes (Weber fractions).

# %% [markdown]
# ### Distribution of errors
# Here we also look at the distribution of the errors compared to ground truth.

# %%
scales_to_plot = filter_cases(scales, 'trans_type', show_trans_type)
scales_to_plot = filter_cases(scales_to_plot, 'g', show_g)

# %%
hists = sns.relplot(
    data=scales_to_plot,
    # Faceting
    row="g",
    col="trans_type",
    # Panel definition
    x="stimulus",
    y="error",
    kind="scatter",
    height=2,
    aspect=1.0,
    facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
    alpha=0.15,
    edgecolors=None,
    s=20, # markersize
)

hists.set(ylim=(-0.25, 0.25))

for ax in hists.axes.flat:
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0, clip_on=False)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))

hists.set_titles(
    "", 
    row_template=r"$g = {row_name:.2f}$",
    col_template=r"$\alpha = {col_name}$"
)

hists.set_axis_labels(
    y_var="", x_var="", clear_inner=True
)
hists.fig.supylabel(r"Error")
hists.fig.supxlabel(r"Stimulus value $s$")

plt.savefig(
    Path().resolve().parent / "figs" / f"multiplicative-noise_errors.{suffix}.pdf",
    bbox_inches="tight",
)

# %%
# hists = sns.displot(
#     data=scales,
#     # Faceting
#     row="g",
#     col="trans_type",
#     # Panel definition
#     x="stimulus",
#     y="error",
#     kind="hist",
#     # Histogram settings
#     #discrete=(False, False),
#     #common_bins=True,
#     #binwidth=(None, 0.01),
#     bins=11,
#     # Styling
#     color="black",
#     height=2,
#     aspect=1.0,
#     facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
# )

# for ax in hists.axes.flat:
#     ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0, clip_on=False)
#     ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))

# hists.set_titles(
#     "", 
#     row_template=r"$\sigma = {row_name:.2f}$",
#     col_template=r"{col_name}"
# )

# hists.set_axis_labels(
#     y_var="", x_var="", clear_inner=True
# )
# hists.fig.supylabel(r"Error")
# hists.fig.supxlabel(r"Stimulus value $s$")


# plt.savefig(
#     Path().resolve().parent / "figs" / "multiplicative-noise_errors.pdf",
#     bbox_inches="tight",
# )

# %% [markdown]
# ## Fit resulting scales
# Another way to evaluate the goodness of MLDS recovery,
# is to ask how closely we would be able to estimate the true transducer shape
# from a perceptual scale.
# Here we do this by fitting a power-law to the perceptual scale for each simulation separately,
# and comparing the fitted exponents to the ground truth exponents.
#
# We find that the fitted exponent is very close to the ground truth exponent in all cases.

# %%
# we remove the log one
scales_to_fit = scales[scales['trans_type']!='log']

# %%
fit_params = []
for (trans_type, g, sim_id), subset in scales_to_fit.groupby(["trans_type", "g", "simulation_id"]):
    popt, _ = curve_fit(
        f=power_func,
        xdata=subset["stimulus"],
        ydata=subset["scale"],
        p0=[0.5],  # initial guess
    )

    result = pd.DataFrame(
        {
            "simulation_id": sim_id,
            "exponent": float(trans_type),
            "g": g,
            "fit_exponent": popt[0],
        },
        index=[0],
    )
    fit_params.append(result)

fit_params = pd.concat(fit_params).reset_index(drop=True)
fit_params

# %%
# error, in percent of exponent value
fit_params['error'] = (fit_params['fit_exponent'] - fit_params["exponent"])*100/fit_params["exponent"]


# %%
# Try a violinplot
def add_zeroline(data, **kwargs):
    ax = plt.gca()
    ax.axhline(
        y=0,
        color="red",
        linestyle="--",
        zorder=-1,
    )

exp_hist = sns.FacetGrid(data=fit_params, 
                         col="g", col_wrap=3,
                         margin_titles=True, 
                         palette='Blues', height=3,
                         sharex=True, sharey=True,
                         )

#exp_hist.map(sns.violinplot, 'exponent', 'error', 
exp_hist.map(sns.boxplot, 'exponent', 'error', 
             order=[0.33, 0.5, 1.0, 2.0, 3.0],
            ) 
            
exp_hist.map_dataframe(add_zeroline)  
exp_hist.set(ylim=(-40, 70))
exp_hist.set_ylabels('Error in exponent estimation [%]')
exp_hist.set_xlabels('Ground-truth exponent')
plt.savefig(
    #Path().resolve().parent / "figs" / f"multiplicative-noise_paramfit-hists-violins.{suffix}.pdf",
    Path().resolve().parent / "figs" / f"multiplicative-noise_paramfit-hists-boxplot.{suffix}.pdf",
    bbox_inches="tight",
)

#plt.show()

# %%
def add_diagline(data, **kwargs):
    ax = plt.gca()
    ax.plot(
        [0, 5],
        [0, 5],
        color="gray",
        linestyle="--",
        zorder=-1,
    )



# %%
show_g = [0.0165, 0.05, 0.066, 0.1, 0.15, 0.2, 0.4, 0.6]
mask = fit_params['g'].isin(show_g)

# %%
fit_params_toplot = fit_params[mask]

# %%
exp_hist = sns.FacetGrid(data=fit_params_toplot, 
                         col="g", col_wrap=4,
                         margin_titles=True, 
                         palette='Blues', height=3,
                         sharex=True, sharey=True,
                         )

#exp_hist.map(sns.violinplot, 'exponent', 'error', 
exp_hist.map(sns.boxplot, 'exponent', 'fit_exponent', 
             order=[0.33, 0.5, 1.0, 2.0, 3.0],
             native_scale=True,
             width=2,
            ) 
            
exp_hist.map_dataframe(add_diagline)  
exp_hist.set(ylim=(0, 5))
exp_hist.set(xlim=(0, 5))
exp_hist.set_ylabels('Estimated exponent')
exp_hist.set_xlabels('Ground-truth exponent')
plt.savefig(
    #Path().resolve().parent / "figs" / f"multiplicative-noise_paramfit-hists-violins.{suffix}.pdf",
    Path().resolve().parent / "figs" / f"multiplicative-noise_paramfit-hists-boxplot.{suffix}.pdf",
    bbox_inches="tight",
)

#plt.show()

# %%
# Apply different bins per facet
exp_hist = sns.FacetGrid(data=fit_params, 
                         row='g', col="exponent", 
                         sharex=False,
                         margin_titles=True, 
                         sharey=True, 
                         )


def plot_histogram(data, **kwargs):
    ax = plt.gca()
    ground_truth = data["exponent"].unique()[0]

    # Center bins around the ground truth exponent
    n_bins = 25
    width = 0.5 / n_bins
    data_range = data["fit_exponent"].max() - data["fit_exponent"].min()
    bin_width = max(width, data_range / n_bins)
    bins = np.arange(
        ground_truth - (bin_width * n_bins / 2),
        ground_truth + (bin_width * n_bins / 2) + bin_width,
        bin_width,
    )

    # Plot
    ax.hist(data["fit_exponent"], bins=bins, alpha=1, linewidth=1.5, fill=True)


exp_hist.map_dataframe(plot_histogram)


def add_refline(data, **kwargs):
    ax = plt.gca()
    ax.axvline(
        x=data["exponent"].unique()[0],
        color="black",
        linestyle="--",
        label="ground truth",
    )


def add_meanline(data, **kwargs):
    ax = plt.gca()
    ax.axvline(
        x=data[f"fit_exponent"].mean(),
        color="red",
        linestyle="-",
        label="mean fit",
    )


exp_hist.map_dataframe(add_refline)
exp_hist.map_dataframe(add_meanline)

exp_hist.set_ylabels('Count')
exp_hist.set_xlabels('Fitted exponent')
plt.savefig(
    Path().resolve().parent / "figs" / f"multiplicative-noise_paramfit-hists.{suffix}.pdf",
    bbox_inches="tight",
)

#plt.show()

# %%

# %%

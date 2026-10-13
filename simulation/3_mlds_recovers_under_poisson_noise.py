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
# # MLDS recovers transducers under Poisson noise

# %% [markdown]
# Here we show that Maximum Likelihood Difference Scaling (MLDS) can recover
# transducers with Poisson noise.
# We define the same transducers as in previous simulation (logarithmic and power-law),
# and add Poisson noise that scales with the transducer output.
#
# For each transducer, we then simulate MLDS experiments that produce perceptual scales,
# and we compare the recovered perceptual scales to the ground truth.
# If MLDS is able to recover a ground truth transducer(s),
# then the resulting perceptual scales should be similar to that transducer.
#

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

# requires MLDS python package(https://github.com/computational-psychology/mlds)
# A requirement of the package is also to have R and the R packages 'MLDS' and 'psyphy'
from simulate_mlds_experiment import simulate_MLDS_experiment
from transducers import power_func
from utils import normalize_to_range

# %% [markdown]
# ## Ground-truth Transducers
#
# We use the same two types of transducers as in the additive and multiplicative noise simulations:
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
s_log = np.linspace(s_min_log, s_max_log, 1000)

# mapping from s to s_log
_, s_slope, s_intercept = normalize_to_range(s, s_min_log, s_max_log, return_factors=True)

# calculating normalizing coefficients for log transducer, to make it fall in the range 0-1
y, slope, intercept = normalize_to_range(np.log(s_log), ymin=0, ymax=1.0, return_factors=True)

# %%
plt.plot(s, y, label="log")

l = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

for e in l:
    y = power_func(s, exponent=e)
    plt.plot(s, power_func(s, exponent=e), label=f"s^{e}")

plt.legend()

# %% [markdown]
# ### Definitions of transducers and noise
#
# Here we explore the case where the ground-truth transducer has Poisson noise.
# Poisson noise has a different noise-generating process than Gaussian noise,
# ending up asymmetric (skewed), as well as discrete rather than continuous.
# Poisson noise always has a variance equal to the mean transducer value:
#
# $$\begin{align}
# \sigma^2(s) &= \mu(s) \\
# \sigma(s) &= \sqrt{\mu(s)}
# \end{align}$$
#
# We apply this noise to the same transducers as in the additive and multiplicative noise simulations:
#
# - logarithmic: $\mu(s) = \log(s)$ (rescaled to $[0, 1]$)
# - power law: $\mu(s) = s^{\alpha}$
#


# %%
def rescale_s(s, s_slope, s_intercept):
    # for the logarithmic transducer we need to rescale the stimulus value to the range 1-100
    s_log = s_slope * s + s_intercept
    return s_log


def log_rescaled(s_log, slope, intercept):
    # take the log, and rescaling so that the range of mu is in [0, 1]
    mu = slope * np.log(s_log) + intercept
    return np.abs(mu)


# %%
def trans_poisson(s, trans_type):
    """Deterministic (mean) transducer, same as for additive and multiplicative noise."""
    if trans_type == "log":
        s_log = rescale_s(s, s_slope, s_intercept)
        mu = log_rescaled(s_log, slope, intercept)
    else:
        exponent = float(trans_type)
        mu = power_func(s, exponent=exponent)
    return mu


# %%
def trans_poisson_noisy(s, noise_scalar, trans_type):
    """Noisy transducer with Poisson noise applied to the transducer mean."""
    mu = trans_poisson(s, trans_type)
    # rescale to the range of 0-noise_scalar
    mu_rescaled = mu * noise_scalar  
    psi = np.random.poisson(mu_rescaled, size=mu_rescaled.shape).astype(float)
    # rescale back to the range [0-1]
    psi_rescaled = psi / noise_scalar  
    return psi_rescaled


# %%
def plot_trans_Poisson(trans_type, **kwargs):
    trans_type = trans_type.unique()[0]

    mu = trans_poisson(s, trans_type)
    sigma = np.sqrt(mu)  # Poisson noise

    ax = plotting.transducer(s=s, mu=mu, sigma=sigma)
    ax.set_title("")


# %% [markdown]
# ## Determining parameter values to run
#
# We use the same set of transducer types as in the additive and multiplicative noise simulations:
# logarithmic and power law with exponents $\alpha \in \{0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0\}$.
#

# %%
exponent_set = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # power-law exponents
transducers = ["log"] + [str(e) for e in exponent_set]  # transducer types

# Noise magnitude, i.e. the noise standard deviation at mu(s) = 1. This is the
# same quantity that sigma denotes for additive noise and g for multiplicative
# noise, so these values are taken from the noise levels used in those
# simulations. A magnitude of 1 is the largest this noise model can produce for
# a transducer defined on [0, 1].
noise_magnitude_set = [1.0, 0.6, 0.2, 0.1]

# The mean of the Poisson distribution is scaled by c = 1 / g**2 to obtain a
# noise magnitude of g, since sigma(s) = sqrt(mu(s) / c).
noise_scalar_set = [round(1.0 / g**2, 2) for g in noise_magnitude_set]

ground_truth_params = pd.DataFrame(
    list(itertools.product(transducers, noise_scalar_set)),
    columns=["trans_type", "noise_scalar"],
)
ground_truth_params


# %%
# save parameters
ground_truth_params.to_csv(
    Path().resolve().parent / "data" / "simulations" / "poisson-noise.params.csv",
    index=False,
)


# %%
col_order = ["log"] + [str(e) for e in exponent_set]

grid_scales = sns.FacetGrid(
    data=ground_truth_params,
    col="trans_type",
    #row="noise_scalar",
    col_order=col_order,
    height=3,
    margin_titles=True,
    aspect=1.0,
)

grid_scales.map(plot_trans_Poisson, "trans_type")
grid_scales.set(ylim=[-0.1, 1.1])
grid_scales.set_titles("", col_template=r"$\alpha = {col_name}$")

# the logarithmic transducer has no exponent, so show just "log"
for _ax, _col in zip(grid_scales.axes[0], grid_scales.col_names):
    if _col == "log":
        _ax.set_title(r"$log$")

grid_scales.set_axis_labels(x_var=r"Stimulus value $s$", y_var=r"$\Psi(s)$", clear_inner=True)

plt.show()


# %% [markdown]
# ## Simulating an MLDS experiment

# %% [markdown]
# We simulate MLDS experiments for all these transducers with Poisson noise.
#
# We sample $n$ values of $s$
# and construct all possible triads from these values.
# For each triad, we put the stimulus values through the noisy transducer
# ($\mu(s)$ and Poisson noise)
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
# We repeat this process $n_{repeats} = 10$ times per triad.
#
# To estimate the perceptual scale from these binary responses,
# we use the MLDS method as described by Knoblauch and Maloney (2012).
#

# %%
# stimulus sample values
n_samples_mlds = 10
s_samples_mlds = np.linspace(s_min, s_max, n_samples_mlds)

n_repeats = 10  # per triad
n_simulations = 100

suffix = f"{n_samples_mlds}.{n_repeats}.{n_simulations}"

# %%
# If simulations are already done, then read them
try:
    scales = pd.read_csv(
        Path().resolve().parent / "data" / "simulations" / f"poisson-noise.scales.{suffix}.csv",
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
            transducer=partial(
                trans_poisson_noisy,
                noise_scalar=case["noise_scalar"],
                trans_type=case["trans_type"],
            ),
            n_repeats=n_repeats,
            n_simulations=n_simulations,
            standardscale=True,
        )

        # Accumulate the estimated scales
        scale = (
            scale.reset_index(names="simulation_id", drop=False)
            .melt(var_name="stimulus", value_name="scale", id_vars=["simulation_id"])
            .assign(trans_type=case["trans_type"], noise_scalar=case["noise_scalar"])
        )
        scales.append(scale)

    # Concatenate and save
    scales = pd.concat(scales, ignore_index=True)
    scales["trans_type"] = scales["trans_type"].astype(str)

    scales = scales.reindex(
        ["trans_type", "noise_scalar", "simulation_id", "stimulus", "scale"],
        axis="columns",
    )
    scales.to_csv(
        Path().resolve().parent / "data" / "simulations" / f"poisson-noise.scales.{suffix}.csv",
        index=False,
    )


# %%
scales

# %%
# At the largest noise magnitude the responses approach chance, and the
# generalized linear model behind MLDS can become weakly identified: the
# estimated decision noise then runs away and the scale is not interpretable.
# We treat such fits as non-converged and exclude them.
SIGMA_DECISION_MAX = 10.0

_keys = ["trans_type", "noise_scalar", "simulation_id"]

sigma_decision = (
    scales[scales["stimulus"] == "sigma_decision"]
    .rename(columns={"scale": "sigma_decision"})
    .astype({"sigma_decision": float})
)
converged = sigma_decision[sigma_decision["sigma_decision"] <= SIGMA_DECISION_MAX][_keys]
n_excluded = len(sigma_decision) - len(converged)
print(f"excluded {n_excluded} of {len(sigma_decision)} fits as non-converged")

# remove the values where the estimated noise is stored, we do not needed it at the moment
scales = scales[scales["stimulus"] != "sigma_decision"].copy()
scales["stimulus"] = scales["stimulus"].astype(float)
scales["trans_type"] = scales["trans_type"].astype(str)
scales["noise_scalar"] = scales["noise_scalar"].astype(float)

scales = scales.merge(converged.astype({"noise_scalar": float}), on=_keys, how="inner")


# %%
mean_scales = (
    scales.groupby(["trans_type", "noise_scalar", "stimulus"])
    .agg(
        mean_scale=("scale", "mean"),
        CI_low=("scale", lambda x: np.quantile(x, 0.025)),
        CI_high=("scale", lambda x: np.quantile(x, 0.975)),
    )
    .reset_index()
)
mean_scales


# %%
def filter_cases(scales, column, values):
    # showing only some values for the given column
    mask = scales[column].isin(values)
    return scales[mask]


show_trans_type = ["log", "0.33", "0.5", "1.0", "2.0", "3.0"]

mean_scales_all = mean_scales.copy()
mean_scales = filter_cases(mean_scales, "trans_type", show_trans_type)


# %%
## plot all simulated cases
sns.set_context("paper", font_scale=1.3)
grid_scales = sns.FacetGrid(
    data=mean_scales,
    col="trans_type",
    row="noise_scalar",
    col_order=show_trans_type,
    height=2,
    margin_titles=True,
    aspect=1.0,
    sharey=True,
)

grid_scales.map(plot_trans_Poisson, "trans_type")
grid_scales.map(plotting.scale_MLDS_df, "mean_scale", "CI_low", "CI_high", ss=s_samples_mlds)

grid_scales.set(ylim=[-0.1, 1.1])
grid_scales.set_titles(
    "",
    col_template=r"$\alpha = {col_name}$",
    row_template=r"$c = {row_name:g}$"
)

# the logarithmic transducer has no exponent, so show just "log"
for _ax, _col in zip(grid_scales.axes[0], grid_scales.col_names):
    if _col == "log":
        _ax.set_title(r"$log$")

grid_scales.set_axis_labels(y_var="", x_var="", clear_inner=True)

grid_scales.fig.supylabel(r"Perceptual scale $\Psi(s)$")
grid_scales.fig.supxlabel(r"Stimulus value $s$")

# keep three x-axis ticks despite the larger font
for _ax in grid_scales.axes.flat:
    _ax.set_xticks([0, 0.5, 1.0])

# prevent labels from being clipped in the saved PDF
for _t in grid_scales.fig.findobj(mpl.text.Text):
    _t.set_clip_on(False)

plt.savefig(
    Path().resolve().parent / "figs" / f"poisson-noise_scales.{suffix}.pdf",
    bbox_inches="tight",
)

# plt.show()


# %% [markdown]
# Clearly the perceptual scales produced by MLDS for the different transducers are distinct.
# Generally, these scales match the shape of the transducer mean $\mu(s)$ in each case.
# We can conclude that MLDS is able to recover the underlying transducer
# even in the presence of Poisson noise.
#

# %% [markdown]
# ## Evaluating goodness of MLDS recovery
# We evaluate the goodness of the perceptual scales by MLDS in recovering the ground truth transducer.
# We do this by comparing the perceptual scale $\hat{\mu}(s)$ to the ground truth $\mu(s)$,
# and calculating the errors between the two.

# %%
# Ground truth values at sampled stimulus levels
scales["ground_truth"] = scales.apply(
    func=lambda row: trans_poisson(row["stimulus"], trans_type=row["trans_type"]),
    axis="columns",
)

# Normalize ground truth to [0, 1] range (per group)
scales["ground_truth"] = scales.groupby(["trans_type", "simulation_id"])["ground_truth"].transform(
    lambda x: normalize_to_range(x)
)

# Error
scales["error"] = scales["scale"] - scales["ground_truth"]

# Rescale error to proportion of ground truth range (per group)
scales["error"] = scales.groupby(["trans_type", "simulation_id"])["error"].transform(
    lambda x: x / scales.loc[x.index, "ground_truth"].max()
)

scales


# %% [markdown]
# ### Overall goodness: RMSE

# %%
rmses = (
    scales.groupby(["trans_type", "noise_scalar", "simulation_id"])
    .agg(
        rmse=("error", lambda x: np.sqrt(np.mean(x**2))),
    )
    .reset_index(drop=False)
)


# %%
rmses_agg = (
    rmses.groupby(["trans_type", "noise_scalar"])
    .agg(
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
    )
    .reset_index(drop=False)
    .sort_values(by=["trans_type", "noise_scalar"], ascending=[True, True])
)

rmses_agg.round(3)


# %% [markdown]
# We see a difference in the goodness of recovering the various transducers.
# Overall, transducers with stronger nonlinearity (higher exponent $\alpha$) tend to show
# greater deviation from the ground truth.
#


# %%
def add_RMSE_label(trans_type, noise_scalar, rmses, **kwargs):
    ax = plt.gca()

    trans_type = trans_type.unique()[0]
    noise_scalar = noise_scalar.unique()[0]
    rmses_subset = rmses[(rmses["trans_type"] == trans_type) & (rmses["noise_scalar"] == noise_scalar)]

    xyposition = (0, 0.95) if (trans_type=='2.0') or (trans_type=='3.0') else (1, 0.05)
    ha = "left" if (trans_type=='2.0') or (trans_type=='3.0') else "right"
    
    ax.annotate(
        rf"{rmses_subset['mean_rmse'].values[0]:.2%} $\pm$ {rmses_subset['std_rmse'].values[0]:.2%}",
        xy=xyposition,
        fontsize=0.85 * sns.plotting_context()["axes.labelsize"],
        ha=ha,
    )
    

grid_scales.map(add_RMSE_label, "trans_type", "noise_scalar", rmses=rmses_agg)
grid_scales.set_titles(
    "",
    col_template=r"$\alpha = {col_name}$",
    row_template=r"$c = {row_name:g}$",
)
# the logarithmic transducer has no exponent, so show just "log"
for _ax, _col in zip(grid_scales.axes[0], grid_scales.col_names):
    if _col == "log":
        _ax.set_title(r"$log$")
grid_scales.set_axis_labels(y_var="", x_var="", clear_inner=True)
grid_scales.fig.supylabel(r"Perceptual scale $\Psi(s)$", x=0.035, y=0.5)
grid_scales.fig.supxlabel(r"Stimulus value $s$", y=0.02)

# keep three x-axis ticks despite the larger font
for _ax in grid_scales.axes.flat:
    _ax.set_xticks([0, 0.5, 1.0])

# prevent labels from being clipped in the saved PDF
for _t in grid_scales.fig.findobj(mpl.text.Text):
    _t.set_clip_on(False)

plt.savefig(
    Path().resolve().parent / "figs" / f"poisson-noise_scales.{suffix}.pdf",
    bbox_inches="tight",
)

# plt.show()


# %% [markdown]
# ### Distribution of errors
# Here we also look at the distribution of the errors compared to ground truth.

# %%
scales_to_plot = filter_cases(scales, "trans_type", show_trans_type)


# %%
hists = sns.relplot(
    data=scales_to_plot,
    # Faceting
    col="trans_type",
    col_order=show_trans_type,
    row="noise_scalar",
    # Panel definition
    x="stimulus",
    y="error",
    kind="scatter",
    height=2,
    aspect=1.0,
    facet_kws={"margin_titles": True, "sharey": True, "sharex": True},
    alpha=0.15,
    edgecolors=None,
    s=20,  # markersize
)

hists.set(ylim=(-0.25, 0.25))

for ax in hists.axes.flat:
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0, clip_on=False)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))

hists.set_titles("", col_template=r"$\alpha = {col_name}$", row_template=r"$c = {row_name:g}$")

# the logarithmic transducer has no exponent, so show just "log"
for _ax, _col in zip(hists.axes[0], hists.col_names):
    if _col == "log":
        _ax.set_title(r"$log$")

hists.set_axis_labels(y_var="", x_var="", clear_inner=True)
hists.fig.supylabel(r"Error", x=-0.02, y=0.5)
hists.fig.supxlabel(r"Stimulus value $s$")

plt.savefig(
    Path().resolve().parent / "figs" / f"poisson-noise_errors.{suffix}.pdf",
    bbox_inches="tight",
)


# %% [markdown]
# ## Fit resulting scales
# Another way to evaluate the goodness of MLDS recovery,
# is to ask how closely we would be able to estimate the true transducer shape
# from a perceptual scale.
# Here we do this by fitting a power-law to the perceptual scale for each simulation separately
# (for the power-law transducers),
# and comparing the fitted exponents to the ground truth exponents.
#
# The fitted exponent is close to the ground truth exponent at the higher scaling
# parameters, with a slight overestimation for the expansive transducers that
# grows as the scaling parameter decreases (i.e., as the noise magnitude grows).
#

# %%
# we remove the log one, since it has no exponent to recover
scales_to_fit = scales[scales["trans_type"] != "log"]

fit_params = []
for (trans_type, noise_scalar, sim_id), subset in scales_to_fit.groupby(["trans_type", "noise_scalar", "simulation_id"]):
    try:
        popt, _ = curve_fit(
            f=power_func,
            xdata=subset["stimulus"],
            ydata=subset["scale"],
            p0=[0.5],  # initial guess
        )
    except RuntimeError:
        popt = [np.nan]

    result = pd.DataFrame(
        {
            "simulation_id": sim_id,
            "exponent": float(trans_type),
            "noise_scalar": float(noise_scalar),
            "fit_exponent": popt[0],
        },
        index=[0],
    )
    fit_params.append(result)

fit_params = pd.concat(fit_params).reset_index(drop=True)
fit_params


# %%
# error, in percent of exponent value
fit_params["error_exponent"] = (
    (fit_params["fit_exponent"] - fit_params["exponent"]) * 100 / fit_params["exponent"]
)


# %%
def add_zeroline(data, **kwargs):
    ax = plt.gca()
    ax.axhline(
        y=0,
        color="red",
        linestyle="--",
        zorder=-1,
    )


# Both axes show the same quantity (a power-law exponent), so they are drawn on
# the same log-spaced range, which makes the identity line meaningful and gives
# the boxes a usable width at every exponent.
EXPONENT_LIMS = (0.1, 8.0)
EXPONENT_TICKS = [0.33, 0.5, 1.0, 2.0, 3.0]


def add_diagline(data, **kwargs):
    """Identity line, y = x, on equal log-scaled axes."""
    ax = plt.gca()
    ax.plot(
        EXPONENT_LIMS,
        EXPONENT_LIMS,
        color="gray",
        linestyle="--",
        zorder=-1,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(EXPONENT_LIMS)
    ax.set_ylim(EXPONENT_LIMS)

    # Square panel: with equal limits on both axes, this puts the identity line
    # at 45 degrees regardless of how the figure itself is laid out.
    ax.set_box_aspect(1)

    for _axis, _setter in ((ax.xaxis, ax.set_xticks), (ax.yaxis, ax.set_yticks)):
        _setter(EXPONENT_TICKS)
        _axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        _axis.set_minor_formatter(mpl.ticker.NullFormatter())

    for _label in ax.get_xticklabels():
        _label.set_rotation(45)
        _label.set_horizontalalignment("right")


# %%
show_exponents = [0.33, 0.5, 1.0, 2.0, 3.0]

exp_hist = sns.FacetGrid(
    data=fit_params,
    col="noise_scalar",
    margin_titles=True,
    palette="Blues",
    height=3,
    aspect=1.0,
    sharex=True,
    sharey=True,
)

exp_hist.map(
    sns.boxplot,
    "exponent",
    "fit_exponent",
    order=show_exponents,
    native_scale=True,
    log_scale=True,
    width=0.7,
)

exp_hist.map_dataframe(add_diagline)
exp_hist.set_titles(col_template=r"$c = {col_name:g}$")
exp_hist.set_ylabels("Estimated $\\alpha$")
exp_hist.set_xlabels("Ground-truth $\\alpha$")

plt.savefig(
    Path().resolve().parent / "figs" / f"poisson-noise_paramfit-exponent-boxplot.{suffix}.pdf",
    bbox_inches="tight",
)

# plt.show()


# %%
# Apply different bins per facet
exp_hist = sns.FacetGrid(
    data=fit_params,
    col="exponent",
    row="noise_scalar",
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
        x=data["fit_exponent"].mean(),
        color="red",
        linestyle="-",
        label="mean fit",
    )


exp_hist.map_dataframe(plot_histogram)
exp_hist.map_dataframe(add_refline)
exp_hist.map_dataframe(add_meanline)

exp_hist.set_ylabels("Count")
exp_hist.set_xlabels("Fitted $\\alpha$")
exp_hist.set_titles(col_template=r"$\alpha = {col_name:.2f}$",
                    row_template=r"$c = {row_name:g}$",
                   )

plt.savefig(
    Path().resolve().parent / "figs" / f"poisson-noise_paramfit-exponent-hists.{suffix}.pdf",
    bbox_inches="tight",
)

# plt.show()

# %%

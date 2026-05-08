import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Better scaling, ticks, etc.
# import seaborn as sns
# sns.set_context("paper")
plt.style.use("seaborn-v0_8-paper")
# plt.style.use('seaborn-v0_8-talk')
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["figure.figsize"] = (4.0, 4.0)  # figure size in inches

# Set the font
matplotlib.rcParams["font.sans-serif"] = "Arial"
matplotlib.rcParams["font.family"] = "sans-serif"

# Color palette
palette = {
    "sensitivity": "#377eb8",  # Renamed from "blue"
    "transducer": "#e41a1c",  # Renamed from "red"
    "modulated-poisson": "#bd0026",
    "multiplicative": "#fd8d3c",
    "additive": "#feb24c",
    "suprathreshold": "#4daf4a",  # New green color
    "mlds": "#252525",  # dark gray almost black
}


def transducer(s, mu, sigma=None, logscale=False, ax=None, color=palette["transducer"], **kwargs):
    if ax is None:
        ax = plt.gca()

    # Plot the transducer mean \mu(s)
    ax.plot(s, mu, color=color, **kwargs)

    # Plot the transducer standard deviation \sigma(s)
    if sigma is not None:
        ax.fill_between(s, mu - sigma, mu + sigma, alpha=0.3, color=color, **kwargs)

    # Scaling
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Labeling
    ax.set_xlabel(r"Stimulus value $s$")
    ax.set_ylabel(r"$\Psi(s)$")
    ax.set_title("Transducer")

    return ax


def sampled_transducer(
    ss, means, stds=None, logscale=False, ax=None, color=palette["transducer"], **kwargs
):
    if ax is None:
        ax = plt.gca()

    ax.errorbar(ss, means, stds, color=color, **kwargs, fmt="o")

    # Scaling
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Labeling
    ax.set_xlabel(r"Stimulus value $s$")
    ax.set_ylabel(r"$\Psi(s)$")
    ax.set_title("Transducer")

    return ax


def tvi(s, thresholds, logscale=False, ax=None, color=palette["sensitivity"], **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.plot(s, thresholds, "-", color=color, **kwargs)

    # Scaling
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Labeling
    ax.set_xlabel(r"Stimulus value $s$")
    ax.set_ylabel("Threshold")
    ax.set_title("Threshold vs (stimulus) Intensity (TvI)")

    return ax


def tvi_sampled(
    ss, thresholds, errors=None, logscale=False, ax=None, color=palette["sensitivity"], **kwargs
):
    if ax is None:
        ax = plt.gca()

    ax.errorbar(ss, thresholds, yerr=errors, fmt="o", color=color, **kwargs)

    # Scaling
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Labeling
    ax.set_xlabel(r"Stimulus value $s$")
    ax.set_ylabel("Threshold")
    ax.set_title("Threshold vs (stimulus) Intensity (TvI)")

    return ax


def sensitivity(s, values, logscale=False, ax=None, color=palette["sensitivity"], **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.plot(s, values, color=color, **kwargs)

    # Scaling
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Labeling
    ax.set_xlabel(r"Stimulus value $s$")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Sensitivity")

    return ax


def sensitivity_sampled(
    ss, sensitivities, errors=None, logscale=False, ax=None, color=palette["sensitivity"], **kwargs
):
    if ax is None:
        ax = plt.gca()

    ax.errorbar(ss, sensitivities, yerr=errors, color=color, **kwargs, fmt="o")

    # Scaling
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Labeling
    ax.set_xlabel(r"Stimulus value $s$")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Sampled Sensitivity")

    return ax


def integrated_sensitivity(
    s, I, logscale=False, ax=None, color=palette["suprathreshold"], **kwargs
):
    if ax is None:
        ax = plt.gca()

    ax.plot(s, I, color=color, **kwargs)

    # Scaling
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Labeling
    ax.set_xlabel(r"Stimulus value $s$")
    ax.set_ylabel(r"Integrated sensitivity $I(s)$")
    ax.set_title("Integrated sensitivity")

    return ax


def scale_ME(ss, means, stds=None, logscale=False, ax=None, color=palette["transducer"], **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.errorbar(ss, means, stds, color=color, **kwargs, fmt="o")

    # Scaling
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Labeling
    ax.set_xlabel(r"Stimulus value $s$")
    ax.set_ylabel("Magnitude rating")
    ax.set_title("Magnitude Estimation Scale")

    return ax


def scale_MLDS(ss, means, CIs=None, logscale=False, ax=None, color=palette["mlds"], **kwargs):
    if ax is None:
        ax = plt.gca()

    if CIs is None:
        ax.plot(
            ss,
            means,
            "o",
            color=color,
            **kwargs,
        )
    else:
        ax.errorbar(
            ss,
            means,
            yerr=[means - CIs[:, 0], CIs[:, 1] - means],
            color=color,
            **kwargs,
            fmt="o",
        )

    # Scaling
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Labeling
    ax.set_xlabel(r"Stimulus value $s$")
    ax.set_ylabel("Difference scale")
    ax.set_title("MLDS Scale")

    return ax


def suprathreshold_sampled(
    ss, means, errors=None, logscale=False, ax=None, color=palette["suprathreshold"], **kwargs
):
    if ax is None:
        ax = plt.gca()

    ax.errorbar(ss, means, yerr=errors, fmt="o", color=color, **kwargs)

    # Scaling
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    # Labeling
    ax.set_xlabel(r"Stimulus value $s$")
    ax.set_ylabel("Perceptual scale")
    # ax.set_title(" Scale")

    return ax


def scale_MLDS_df(means, CI_low=None, CI_high=None, **kwargs):
    if CI_low is not None and CI_high is not None:
        CIs = np.vstack([CI_low, CI_high]).T

    ax = scale_MLDS(
        ss=kwargs.get("ss"),
        means=means,
        CIs=CIs,
    )
    ax.set_title(None)
    ax.set_ylabel("Perceived magnitude $\Psi(s)$")

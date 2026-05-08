#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to run a simulation of an MLDS experiment

It requires the mlds package
(https://github.com/computational-psychology/mlds)

@author: G. Aguilar - Joris Vincent, 2025
"""

import itertools
import os
from functools import partial

import mlds
import numpy as np
import pandas as pd


def generate_triads(stim_levels, n_repeats=15):
    """All possible combinations of stimuli in triads to be presented to the (simulated) observer

    It computed all possible combinations of stimuli in triads, and repeats
    them n_repeats times

    Parameters
    ----------
    stim_levels : Sequence
        stimulus values to be used in the experiment
    n_repeats : int, optional
        number of repeats for all possible triads, by default 15

    Returns
    -------
    pandas.DataFrame
        triads to be presented to a (simulated) observer, in a dataframe
        with three columns "s1", "s2", "s3" corresponding to the stimulus
        values to be presented.

    """
    # GENERATE TRIADS

    # Generate all unique triads
    triads = list(itertools.combinations(stim_levels, 3))

    # As dataframe
    triads = pd.DataFrame.from_records(triads, columns=["s1", "s2", "s3"])

    # Repeat
    triads = pd.concat([triads] * n_repeats).reset_index(drop=True)

    return triads


def simulate_observer(triads, transducer):
    """Given a (noisy) transducer function, simulate an observer doing the method of triads

    Parameters
    ----------
    triads : triads to be presented to the simulated observer
        pandas.Dataframe with columns "s1", "s2" and "s3" containing the stimulus values
    transducer : callable
        Transducer function to be called for all stimulus values

    Returns
    -------
    pandas.Dataframe
        Input dataframe triads but with added columns for the transduced
        values ("psi_1..3"), decision variable ("delta_..") and binary response variable
        ("Response")
    """
    # APPLY TRANSDUCER
    # Samples from transducer
    triads["psi_1"] = transducer(triads["s1"])
    triads["psi_2a"] = transducer(triads["s2"])
    triads["psi_2b"] = transducer(triads["s2"])
    triads["psi_3"] = transducer(triads["s3"])

    # RESPONSE
    # Delta
    triads["delta_21"] = triads["psi_2a"] - triads["psi_1"]
    triads["delta_32"] = triads["psi_3"] - triads["psi_2b"]
    triads["delta"] = triads["delta_32"] - triads["delta_21"]

    # Binarize
    triads["Response"] = triads["delta"] > 0
    triads["Response"] = triads["Response"].astype(int)

    return triads


def estimate_scales(data, standardscale=True, boot=False):
    """Estimates perceptual scales with MLDS

    Parameters
    ----------
    data : pandas.DataFrame
        input data to MLDS, with columns "Response", "s1", "s2" and "s3"
    standardscale : bool, optional
        whether to standardize scales (to range [0, 1]), or natural scale
        (range [0, 1/sigma], where sigma is the estimated decision noise).
        The default is True.
    boot : bool, optional
        whether calculate confidence intervals using bootstrap, by default False

    Returns
    -------
    pandas.DataFrame
        Estimated perceptual scale on a dataframe with columns "stimulus" and "scale"
    """
    # ESTIMATE
    df_out = data[["Response", "s1", "s2", "s3"]]

    filename = "triads.csv"
    df_out.to_csv(filename, index=True, index_label="Trial")

    # Estimate
    obs = mlds.MLDSObject(filename, boot=boot, standardscale=standardscale, verbose=False)
    obs.run()

    # Extract scale and return dataframe
    scale_df = pd.DataFrame({"stimulus": obs.stim, "scale": obs.scale})

    # Extract sigma
    if standardscale:
        sigma = obs.sigma
    else:
        sigma = 1 / scale_df["scale"].max()

    # TODO: return CIs in dataframe and numpy array
    os.remove(filename)
    os.remove(obs.Rdatafile)

    return scale_df, sigma


def simulate_MLDS_experiment(
    stim_levels, transducer, n_repeats=15, standardscale=True, n_simulations=1
):
    """Simulates an MLDS experiment with a given transducer and triads.

    Parameters
    ----------
    stim_levels : Sequence
        stimulus values to be used in the experiment
    transducer : callable
        Transducer function to be called for all stimulus values
    n_repeats : int, optional
        number of repeats for all possible triads, by default 15
    standardscale : bool, optional
        whether to standardize scales (to range [0, 1]), or natural scale
        (range [0, 1/sigma], where sigma is the estimated decision noise).
        The default is True.
    n_simulations : int, optional
        number of simulations to run, by default 1

    Returns
    -------
    pandas.DataFrame
        estimated perceptual scale values (columns) for each simulation (rows)
    """
    # Generate triads
    triads = generate_triads(stim_levels, n_repeats=n_repeats)

    # Initialize empty arrays
    scales_arr = np.zeros((n_simulations, len(stim_levels)))
    sigmas_arr = np.zeros((n_simulations, 1))

    # Loop over rest, accumulate
    for i in range(n_simulations):
        # simulate the observer
        df = simulate_observer(triads, transducer)

        # estimate the scales
        scale_df, sigma = estimate_scales(data=df, standardscale=standardscale)

        # extract & transpose
        scale = scale_df["scale"].values.T

        # accumulate the scales
        scales_arr[i, :] = scale
        sigmas_arr[i] = sigma

    # Convert to DataFrame
    scales_df = pd.DataFrame(scales_arr, columns=stim_levels)
    scales_df.index.name = "simulation_id"
    sigmas_df = pd.DataFrame(sigmas_arr, columns=["sigma_decision"])
    sigmas_df.index.name = "simulation_id"

    scales_df = pd.merge(scales_df, sigmas_df, left_index=True, right_index=True)

    return scales_df


if __name__ == "__main__":
    import plotting
    import transducers
    from utils import normalize_to_range

    print("Running a simulation of an MLDS experiment")
    n_simulations = 50
    n_samples = 10
    n_repeats = 15

    # Sample stimulus space
    stim_levels = np.linspace(0, 1, n_samples)

    # Transducer: power-law with additive noise
    transducer_params = {"exponent": 0.5, "sigma": 0.02}
    transducer = partial(transducers.power_noisy_additive, **transducer_params)

    # Simulate the MLDS experiment
    scales_simulated = simulate_MLDS_experiment(
        stim_levels,
        transducer,
        n_repeats=n_repeats,
        standardscale=True,
        n_simulations=n_simulations,
    )

    # Compute mean and standard deviation of the simulated scales
    mean_scale = scales_simulated.mean(axis=0)
    sem_scale = scales_simulated.std(axis=0) / np.sqrt(n_simulations)

    # Plot together with ground truth scale
    s = np.linspace(0, 1, 100)
    mu = transducers.power_func(s=s, exponent=transducer_params["exponent"])
    sigma_s = transducer_params["sigma"]
    mu_normed = normalize_to_range(mu, 0, 1)

    ax = plotting.transducer(s=s, mu=mu_normed, sigma=sigma_s)
    plotting.scale_MLDS(ss=stim_levels, means=mean_scale[:-1], ax=ax)
    plotting.plt.show()

import numpy as np


def normalize_to_range(x, ymin=0, ymax=1.0, return_factors=False):
    """Returns the input vector x normalized to a particular range.

    Default is [0, 1]
    Optionally it can return the scaling and intercept factors used
    during normalization

    """
    s = (ymax - ymin) / (np.max(x) - np.min(x))
    i = ymax - s * np.max(x)

    output = s * x + i
    if return_factors:
        return output, s, i
    else:
        return output


def remove_nans(x, y):
    """Removes nans present in the vector y,
    and also their correspondin x values in vector x"""
    idx = np.where(~np.isnan(y))[0]
    if len(idx) != len(y):
        print("removing nans")
    return (x[idx], y[idx])


def rmse(x, y):
    """Root-mean squared error"""
    assert len(x) == len(y)
    n = len(x)

    return np.sqrt(np.sum((x - y) ** 2) / n)


def log_likelihood_small_sample(ss_res, n_data, k_params):
    """Calculate log-likelihood for small samples with bias correction

    Parameters:
    ss_res: residual sum of squares
    n_data: number of observations
    k_params: number of parameters in the model

    Returns:
    log_likelihood: log-likelihood value
    """

    # Use unbiased estimator for sigma (divide by n-p instead of n)
    # This is crucial for small samples
    degrees_of_freedom = n_data - k_params

    if degrees_of_freedom <= 0:
        print(f"Warning: Model has {k_params} parameters but only {n_data} data points!")
        return -np.inf

    sigma_squared = ss_res / degrees_of_freedom

    # Calculate log-likelihood (assuming Gaussian errors)
    log_likelihood = -n_data / 2 * np.log(2 * np.pi * sigma_squared) - ss_res / (2 * sigma_squared)

    return log_likelihood


def calculate_aic(log_likelihood, n_params):
    """Calculate AIC from log-likelihood"""
    return 2 * n_params - 2 * log_likelihood


def calculate_aicc(log_likelihood, n_params, n_data):
    """Calculate AICc (corrected AIC) for small samples"""
    aic = calculate_aic(log_likelihood, n_params)
    # Small sample correction
    correction = (2 * n_params * (n_params + 1)) / (n_data - n_params - 1)
    return aic + correction


def calculate_bic(log_likelihood, n_params, n_data):
    """Calculate BIC from log-likelihood"""
    return np.log(n_data) * n_params - 2 * log_likelihood


def calculate_r2(ss_res, ss_tot):
    """Calculate R-squared"""
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def adjust_r2(r_squared, n_data, n_params):
    """Calculate adjusted R-squared"""
    return (
        1 - (1 - r_squared) * (n_data - 1) / (n_data - n_params - 1)
        if n_data > n_params + 1
        else 0
    )

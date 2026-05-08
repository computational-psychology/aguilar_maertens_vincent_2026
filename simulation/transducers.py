import numpy as np


def power_func(s, exponent, a=1.0, b=0.0):
    """Power law transducer function

    mu(s) = a * s^exponent + b
    where:
    - mu(s) is the transduced/encoded stimulus value
    - s is the stimulus value
    - exponent is the exponent of the power function
    - a is the scaling factor
    - b is the offset

    Parameters
    ----------
    s : array-like
        stimulus value(s)
    exponent : float
        exponent of the power function
    a : float, optional
        scaling factor, by default 1.0
    b : float, optional
        offset, by default 0.0

    Returns
    -------
    array-like
        transduced/encoded stimulus value(s) into mu(s)
    """
    return a * s**exponent + b


def logarithmic(s, a=1.0, b=0.0):
    """Logarithmic transducer function

    mu(s) = a * log(s + 1) + b
    where:
    - mu(s) is the transduced/encoded stimulus value
    - s is the stimulus value
    - a is the scaling factor
    - b is the offset

    Parameters
    ----------
    s : array-like
        stimulus value(s)
    a : float, optional
        scaling factor, by default 1.0
    b : float, optional
        offset, by default 0.0

    Returns
    -------
    array-like
        transduced/encoded stimulus value(s) into mu(s)
    """
    return a * np.log(s + 1) + b


def power_noisy_additive(s, exponent, sigma=0.0, a=1.0, b=0.0):
    """Noisy power law transducer with additive noise

    psi(s) = N(mu(s), sigma)
    where:
    - N(mu(s), sigma) is the normal distribution with mean mu(s) and standard deviation sigma
    - sigma is a constant, the standard deviation of the noise
    - mu(s) = a * s^exponent + b
    - s is the stimulus value
    - exponent is the exponent of the power function
    - a is the scaling factor
    - b is the offset

    Parameters
    ----------
    s : array-like
        stimulus value(s)
    exponent : float
        exponent of the power function
    sigma : float, optional
        standard deviation of the noise, by default 0.0
    a : float, optional
        scaling factor, by default 1.0
    b : float, optional
        offset, by default 0.0

    Returns
    -------
    array-like
        noisy transduced/encoded stimulus value(s) into psi(s)
    """
    mu = power_func(s, exponent=exponent, a=a, b=b)
    # we pass the size to make it a vector, different samples
    noise = np.random.normal(loc=0, scale=sigma, size=mu.shape)

    psi = mu + noise

    return psi


def power_noisy_multiplicative(s, exponent, g=1.0, a=1.0, b=0.0):
    """Noisy power law transducer with multiplicative noise

    This case implements the power law transducer

    mu(s) = a * s^exponent + b

    with a noise function for multiplicative noise

    sigma(s) = g*mu(s)


    where:
    - s is the stimulus value
    - exponent is the exponent of the power function
    - a is the scaling factor
    - b is the offset
    - g is the scaling factor for the multiplicative noise

    Parameters
    ----------
    s : array-like
        stimulus value(s)
    exponent : float
        exponent of the power function
    g : float, optional
        scaling factor for the multiplicative noise, by default 1.0
    a : float, optional
        scaling factor, by default 1.0
    b : float, optional
        offset, by default 0.0

    Returns
    -------
    array-like
        noisy transduced/encoded stimulus value(s) into psi(s)
    """
    mu = power_func(s, exponent=exponent, a=a, b=b)
    sigma = g * mu  # a vector
    noise = np.random.normal(loc=0, scale=sigma)  # also a vector

    psi = mu + noise

    return psi


def log_noisy_additive(s, sigma=0.0, a=1.0, b=0.0):
    """Noisy logarithmic transducer with additive noise

    psi(s) = N(mu(s), sigma)
    where:
    - N(mu(s), sigma) is the normal distribution with mean mu(s) and
      standard deviation sigma
    - sigma is a constant, the standard deviation of the noise
    - mu(s) = a * log(s + 1) + b
    - s is the stimulus value
    - a is a scaling factor
    - b is an offset

    Parameters
    ----------
    s : array-like
        stimulus value(s)
    a : float, optional
        scaling factor, by default 1.0
    b : float, optional
        offset, by default 0.0
    sigma : float, optional
        standard deviation of the noise, by default 0.0

    Returns
    -------
    array-like
        noisy transduced/encoded stimulus value(s) into psi(s)
    """
    mu = logarithmic(s=s, a=a, b=b)

    noise = np.random.normal(loc=0, scale=sigma, size=mu.shape)
    psi = mu + noise

    psi = np.clip(psi, 0, None)

    return psi


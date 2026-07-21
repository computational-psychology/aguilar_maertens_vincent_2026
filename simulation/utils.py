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


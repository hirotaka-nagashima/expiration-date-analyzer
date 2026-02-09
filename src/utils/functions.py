"""Additional functions for numpy."""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def power_law(x, a, c):
    return c * (x ** a)


def power_law_with_exponential_cutoff(x, a, b, c):
    return c * (x ** a) * (np.e ** (b * x))


def centered_moving_average(x, width: int, excludes_edges=False):
    mask = np.ones(width) / width
    ma = np.convolve(x, mask, mode="same")
    if excludes_edges:
        excluded_width = int(width / 2)
        ma[:excluded_width] = x[:excluded_width]
        ma[-excluded_width:] = x[-excluded_width:]
    return ma

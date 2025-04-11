import numpy as np


def trapezoidal_membership(x: np.ndarray, abcd: np.ndarray) -> np.ndarray:
    """
    Compute the trapezoidal membership function.

    Parameters
    ----------
    x : np.ndarray
        Independent variable array.
    abcd : np.ndarray
        Four-element array defining the trapezoidal shape 

    Returns
    -------
    np.ndarray
        Trapezoidal membership values.
    """
    abcd = np.sort(abcd) 
    a, b, c, d = abcd

    y = np.ones_like(x)

    # Left slope
    y[x < a] = 0
    left_idx = (a <= x) & (x < b)
    y[left_idx] = triangular_membership(x[left_idx], np.array([a, b, b]))

    # Right slope
    y[x > d] = 0
    right_idx = (c < x) & (x <= d)
    y[right_idx] = triangular_membership(x[right_idx], np.array([c, c, d]))

    return y


def triangular_membership(x: np.ndarray, abc: np.ndarray) -> np.ndarray:
    """
    Compute the triangular membership function.

    Parameters
    ----------
    x : np.ndarray
        Independent variable array.
    abc : np.ndarray
        Three-element array defining the triangular shape (a <= b <= c).

    Returns
    -------
    np.ndarray
        Triangular membership values.
    """
    abc = np.sort(abc) 
    a, b, c = abc

    y = np.zeros_like(x)

    # Left slope
    left_idx = (a < x) & (x < b)
    y[left_idx] = (x[left_idx] - a) / (b - a)

    # Right slope
    right_idx = (b < x) & (x < c)
    y[right_idx] = (c - x[right_idx]) / (c - b)

    # Peak
    y[x == b] = 1

    return y

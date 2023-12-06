from typing import Callable
import numpy as np
from numpy.random._generator import Generator
from SquareDivision.config import config

# rng:Generator = np.random.default_rng(config['seed'])


def grid_pts(
    n: int = 3,
    xmin: float = 0,
    xmax: float = 1,
    ymin: float = 0,
    ymax: float = 1,
    randQ: bool = False,
    rng: Generator = np.random.default_rng(config["seed"]),
):
    """Return (n*n, 2) np.ndarray of gridpoints in (0,1)"""
    x_cord = np.linspace(xmin, xmax, n)[1:-1]
    y_cord = np.linspace(ymin, ymax, n)[1:-1]
    xy_cord = np.array(np.meshgrid(x_cord, y_cord)).T.reshape(-1, 2)
    if randQ is True:
        xy_cord = rng.permutation(xy_cord)
    return xy_cord


def uniform_pts(
    n: int = 1,
    xmin: float = 0,
    xmax: float = 1,
    ymin: float = 0,
    ymax: float = 1,
    rng: Generator = np.random.default_rng(config["seed"]),
):
    """Return (n, 2) np.ndarray:
            sample of size n*n from uniform distribution
            in a rectangle (xmin, xmax) by (ymin, ymax)
    """
    return rng.uniform([xmin, ymin], [xmax, ymax], size=(n, 2))

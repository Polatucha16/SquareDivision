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
    """ Return (n*n, 2) np.ndarray:
            sample of size n*n from uniform distribution
            in a rectangle (xmin, xmax) by (ymin, ymax)
        Example:

            import numpy as np
            from numpy.random._generator import Generator
            from SquareDivision.src.generators import uniform_pts
            from SquareDivision.config import config, figure_settings, axis_settings
            import matplotlib.pyplot as plt

            rng:Generator = np.random.default_rng(config['seed'])
            pts_centers = uniform_pts(5, rng=rng)

            fig, ax = plt.subplots(**figure_settings)
            ax.set(**axis_settings)
            ax.scatter(*pts_centers.T, marker='.')
            plt.show()

    """
    pts = []
    for i in range(n):
        x = rng.uniform(low=xmin, high=xmax, size=n)
        y = rng.uniform(low=ymin, high=ymax, size=n)
        curr_pts = np.vstack(tup=(x, y))
        pts.append(curr_pts)
    return np.hstack(pts).T

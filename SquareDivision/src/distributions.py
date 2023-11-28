from functools import partial
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
from numpy.random._generator import Generator
from typing import Callable

from SquareDivision.config import config

from abc import ABC, abstractmethod
from typing import Callable


# rectangles centers strategies
class CentersStrategy(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs) -> np.ndarray:
        pass


# for using numpy distibutions just:
# from SquareDivision.src.rectangle_class import Rectangulation
# rects = Rectangulation(config={'seed' : 1234})
# rects.centers = rects.rng.uniform([0, 0], [1, 1], (11, 2))


# FIX: this is pointless XD, change this to appling a mapping to set of points
class RngDistribution(CentersStrategy):
    """Returns rng.distribution(**kwargs)
    Example:
        from SquareDivision.src.rectangle_class import Rectangulation
        from SquareDivision.src.distributions import RngDistribution
        rects = Rectangulation(config={'seed' : 1234})
        rects.sample_centers(
            RngDistribution(),
            distribution='uniform',
            low=[0, 0],
            high=[1, 1],
            size=(3, 2))
        print(f'uniform\n{rects.centers}')
        rects.sample_centers(
            RngDistribution(),
            distribution='normal',
            loc=0.0,
            scale=1.0,
            size=(3, 2))
        print(f'normal\n{rects.centers}')"""

    def generate(self, rng: Generator, distribution: str, **kwargs):
        return rng.__getattribute__(distribution)(**kwargs)


class FixedCenters(CentersStrategy):
    """Returns array given"""

    def generate(self, arr: np.ndarray):
        return arr


# rectangles width & height strategies
class SizeStrategy(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate(self, *args, **kwargs) -> np.ndarray:
        pass


class FromFunction(SizeStrategy):
    """At the point (x,y), draw from dirac delta distribution
    supported in the point func((x,y))"""

    def __init__(self, func: Callable):
        """func : ((2,) np.ndarray, kwargs) -> float"""
        self.func = func

    def generate(self, centers: np.ndarray, **kwargs):
        values = np.apply_along_axis(self.func, 1, centers)
        return values


class BetweenFunctions(SizeStrategy):
    """At the point (x,y), draw from uniform distribution supported
    between func_0((x,y)) and func_1((x,y))."""

    def __init__(self, func_0: Callable, func_1: Callable, rng):
        self.func_0 = func_0
        self.func_1 = func_1
        self.rng:Generator = rng

    def generate(self, centers: np.ndarray, **kwargs):
        pts_0 = np.apply_along_axis(self.func_0, 1, centers)
        pts_1 = np.apply_along_axis(self.func_1, 1, centers)
        pts: np.ndarray = np.abs(np.c_[pts_0, pts_1])
        pts.sort(axis=-1)
        return self.rng.uniform(low=pts[:, 0], high=pts[:, 1])


class SizeFixed(SizeStrategy):
    def generate(self, widths_or_heights: np.ndarray):
        return widths_or_heights


def linear_on_position(
    centers: np.ndarray, a: np.ndarray = np.array([0.3, 0.3]), b: float = 0.1
):
    """<centers> (N,2) dot broatcasting <a> (2,)"""
    return centers.dot(a) + b

def tepui(
    pt,
    top: float = 0.3,
    bottom: float = 0.05,
    slope: float = 4,
    vertex: float = 1,
    pts: np.ndarray = np.array([[0.25, 0.25], [0.75, 0.75]]),
):
    """
    Plot function:
        from SquareDivision.src.distributions import tepui
        from SquareDivision.draw.draw import draw_func
        tepui_kwargs = {'bottom': 0.1, 'top': 0.45, 'vertex': 0.6, 'slope': 2}
        draw_func(tepui, func_kwargs = tepui_kwargs )
    """
    return np.minimum(
        top,
        np.maximum(bottom, vertex - slope * np.min(np.linalg.norm(pts - pt, axis=1))),
    )


def surface_perp_to(pt, vect: np.ndarray, val_at_0: float):
    """
    Return value of function : (x,y) -> z whichs graph is a surface
    perpendicular to argument vect = (a, b, c) and passing thorough the point (0,0, val_at_0).
    Argument vect cannot have c = 0
    Plot function:
        import numpy as np
        from SquareDivision.src.distributions import surface_perp_to
        from SquareDivision.draw.draw import draw_func
        surface_perp_to_kwargs = {'vect' : np.array([-1, -1, 3]), 'val_at_0' : 0.2}
        draw_func(surface_perp_to, func_kwargs = surface_perp_to_kwargs )
    """
    return -vect[:2].dot(pt) / vect[2] + val_at_0

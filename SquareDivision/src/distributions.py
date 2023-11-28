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
        pts_1 = np.apply_along_axis(self.func_0, 1, centers)
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


# def x_plus_y_func(
#     x: float,
#     y: float,
#     min_00: float,
#     max_00: float,
#     min_11: float,
#     max_11: float,
#     rng: Generator,  # = np.random.default_rng(config['seed'])
# ):
#     """Random variable (width,height) with the following distibution.
#     At the point:
#         (0,0) width is uniform in [min_00, max_00]
#         (1,1) width is uniform in [min_11, max_11]
#         for other (x,y) width is uniform on interval which has
#         endpoints linearly interpolated, depending on x,
#         from the boudaries above.

#         Distibution of height is analogus.

#         Example of use:

#             import functools
#             import numpy as np
#             from numpy.random._generator import Generator
#             import matplotlib.pyplot as plt

#             from SquareDivision.src.distributions import x_plus_y_func
#             from SquareDivision.config import figure_settings, axis_settings

#             rng:Generator = np.random.default_rng(1234)

#             func = functools.partial(x_plus_y_func,
#                                     min_00=0.1, max_00=0.1,
#                                     min_11=0.3, max_11=0.5)

#             # sample of possible widths, heigths at (x, y)=
#             x, y, pts = 0.1, 0.9, []
#             for i in range(10):
#                 pt = np.array(func(x=x,y=y, rng=rng))
#                 pts.append(pt)
#             pts = np.array(pts)

#             fig, ax = plt.subplots(**figure_settings)
#             ax.set(**axis_settings)
#             ax.scatter(*pts.T, marker='.')
#             plt.show()

#     """

#     b_x = (max_11 - max_00) * x + max_00
#     a_x = (min_11 - min_00) * x + min_00
#     width = (b_x - a_x) * rng.random() + a_x

#     b_y = (max_11 - max_00) * y + max_00
#     a_y = (min_11 - min_00) * y + min_00
#     height = (b_y - a_y) * rng.random() + a_y
#     return (width, height)


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
        from functools import partial
        import numpy as np

        import matplotlib.pyplot as plt
        from matplotlib import cm

        from SquareDivision.src.distributions import tepui

        x = np.arange(0, 1, 0.01)
        y = np.arange(0, 1, 0.01)

        X, Y = np.meshgrid(x, y)
        points = np.array([X,Y])

        func = partial(tepui,
            top=0.3,
            bottom=0.05,
            slope=4.0,
            vertex=1.0,
            pts=np.array(
                [[0.25, 0.25],
                [0.75, 0.75]]
                )
        )
        Z = np.apply_along_axis(func, 0, points)

        fig, ax =  plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_xlim3d(left=0, right=1)
        ax.set_ylim3d(bottom=0, top=1)
        ax.set_zlim3d(bottom=0, top=1)
        ax.plot_surface(X, Y, Z,
                        vmin=Z.min(),
                        vmax=Z.max() + 0.1,
                        rstride=1, cstride=1,
                        cmap=cm.terrain
                        )
        plt.show()
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
        from functools import partial
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from SquareDivision.src.distributions import surface_perp_to

        x = np.arange(0, 1, 0.01)
        y = np.arange(0, 1, 0.01)

        X, Y = np.meshgrid(x, y)
        points = np.array([X,Y])

        func = partial(surface_perp_to,
            vect = np.array([-1, -1, 3]),
            val_at_0 = 0.2
        )
        Z = np.apply_along_axis(func, 0, points)

        fig, ax =  plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_xlim3d(left=0, right=1)
        ax.set_ylim3d(bottom=0, top=1)
        ax.set_zlim3d(bottom=0, top=1)
        ax.plot_surface(X, Y, Z,
                        vmin=Z.min(),
                        vmax=Z.max() + 0.1,
                        rstride=1, cstride=1,
                        cmap=cm.terrain
                        )
        plt.show()
    """
    return -vect[:2].dot(pt) / vect[2] + val_at_0

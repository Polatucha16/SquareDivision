import numpy as np
from numpy.random._generator import Generator
from typing import Callable

from SquareDivision.config import config

def x_plus_y_func(
        x : float,
        y : float,
        min_00 : float,
        max_00 : float,
        min_11 : float,
        max_11 : float,
        rng: Generator #= np.random.default_rng(config['seed'])
):
    """ Random variable (width,height) with the following distibution.
    At the point:
        (0,0) width is uniform in [min_00, max_00]
        (1,1) width is uniform in [min_11, max_11]
        for other (x,y) width is uniform on interval which has
        endpoints linearly interpolated, depending on x,
        from the boudaries above.

        Distibution of height is analogus.

        Example of use:

            import functools
            import numpy as np
            from numpy.random._generator import Generator
            import matplotlib.pyplot as plt

            from SquareDivision.src.distributions import x_plus_y_func
            from SquareDivision.config import figure_settings, axis_settings

            rng:Generator = np.random.default_rng(1234)

            func = functools.partial(x_plus_y_func, 
                                    min_00=0.1, max_00=0.1,
                                    min_11=0.3, max_11=0.5) 

            # sample of possible widths, heigths at (x, y)=
            x, y, pts = 0.1, 0.9, []
            for i in range(10):
                pt = np.array(func(x=x,y=y, rng=rng))
                pts.append(pt)
            pts = np.array(pts)

            fig, ax = plt.subplots(**figure_settings)
            ax.set(**axis_settings)
            ax.scatter(*pts.T, marker='.')
            plt.show()
        
        """
    
    b_x = (max_11 - max_00) * x + max_00
    a_x = (min_11 - min_00) * x + min_00
    width = (b_x - a_x) * rng.random() + a_x
    
    b_y = (max_11 - max_00) * y + max_00
    a_y = (min_11 - min_00) * y + min_00
    height = (b_y - a_y) * rng.random() + a_y
    return (width, height)

pts_diag = np.array(
    [[0.25, 0.25],
     [0.75, 0.75]]
    )
def tepui(
        base:float=0.05,
        top:float=0.3,
        slope:float=4,
        vertex:float=1,
        pts:np.ndarray=pts_diag
):
    """ Example:
            import matplotlib.pyplot as plt
            from matplotlib.axes import Axes

            tepui_at = tepui(top=0.4)

            x = np.arange(0, 1, 0.02)
            y = np.arange(0, 1, 0.02)

            X, Y = np.meshgrid(x, y)
            points = np.array([X,Y])

            nu = np.apply_along_axis(tepui_at, 0, points)

            fig = plt.figure()
            ax:Axes = fig.add_subplot(projection='3d')
            ax.axes.set_xlim3d(left=0, right=1) 
            ax.axes.set_ylim3d(bottom=0, top=1) 
            ax.axes.set_zlim3d(bottom=0, top=1) 
            ax.plot_surface(X, Y, nu)
            plt.show()
    """
    return lambda pt : np.minimum(top, np.maximum(base, vertex - slope * np.min(np.linalg.norm(pts - pt, axis=1))))
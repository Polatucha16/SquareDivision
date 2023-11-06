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

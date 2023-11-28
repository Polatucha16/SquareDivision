import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
from SquareDivision.config import figure_settings, axis_settings
# from SquareDivision.src.dataflow import struc_arr

# def set_settings(fig:Figure, ax: Axes ):
#     fig.set_size_inches(*figure_settings['figsize'])
#     ax.set(**axis_settings)
#     return fig, ax

def ax_settings(ax: Axes):
    ax.set(**axis_settings)
    return ax

def struc_arr(arr : np.ndarray, 
              dtype = 
               [('x', float), 
                ('y', float), 
                ('width', float), 
                ('height', float)]
):  
    """ column naming scheme """
    values = list(map(tuple, arr[:,:4]))
    return np.array(values, dtype=dtype)

def draw_rectangles(ax:Axes, arr: np.ndarray, marked:int=-1 , stop:int=-1):
    ax = ax_settings(ax)
    data = struc_arr(arr)
    for i, arg in enumerate(data):
        rect = Rectangle(xy=(arg['x'], arg['y']), 
                        width=arg['width'],
                        height=arg['height'],
                        fill=True,
                        fc=to_rgba('tab:blue', 0.5), 
                        ec='r' if i == marked else 'k')
        ax.add_patch(rect)
        if i == stop:
            break
    ax.vlines([0,1], ymin=0, ymax=1)
    ax.hlines([0,1], xmin=0, xmax=1)
    # ax.plot(*walls_to_plot.T, lw=1, c='tab:orange')
    
    return ax

def rectangle_numbers(ax:Axes, arr: np.ndarray):
    for i, rect in enumerate(arr):
        x, y = rect[:2]
        ax.text(x, y, str(i), horizontalalignment='left', verticalalignment='bottom',)
    return ax

def draw_suspended_walls(ax:Axes, dir, sus_walls):
    if dir == 'r' or dir == 'l':
        ax.scatter(*sus_walls[:,:2].T, marker='.', c='r')
    if dir == 'd' or dir == 'u':
        ax.scatter(*sus_walls[:,[1,0]].T, marker='.', c='r')
    return ax
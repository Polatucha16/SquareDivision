import numpy as np
from typing import Callable

from SquareDivision.src.morph import homogeneus_push_all, wall_push
from SquareDivision.src.regions import homogeneous_scale_in_dir_search
from SquareDivision.contact_graph.incidence_matrix import (
    contact_graph_incidence_matrix
)
from SquareDivision.holes.detect import find_holes

def arg_rect_list(n, pts_func, sizes_func, rng):
    """ Return an np.ndarray (n,5) with rows of the form:
        (x, y, width, height, area)
        representing:
            x, y - center of a rectangle
            width, height - parameters for matplotlib.patches.Rectangle
            area - width * height
        Arguments
            pts_func : function that generates points
                Returns  
                    np.ndarray of shape (*, 2)
                Arguments
                    n   - int parameter controling number of points to generate
                    rng - numpy.random._generator.Generator
            sizes_func : function that control size of rectangle at (x, y)
                Returns 
                    (width, height) : float or array
                Arguments
                    (x, y) : float or array

        Example:
            import numpy as np
            from numpy.random._generator import Generator
            import functools
            from SquareDivision.src.generators import uniform_pts
            from SquareDivision.src.distributions import x_plus_y_func
            from SquareDivision.src.dataflow import arg_rect_list
            from SquareDivision.config import config

            rng:Generator = np.random.default_rng(config['seed'])
            func = functools.partial(x_plus_y_func, 
                                    min_00=0.01, max_00=0.01,
                                    min_11=0.2, max_11=0.3,
                                    rng=rng)
            arr = arg_rect_list(2, uniform_pts, func, rng=rng)
        """

    pts:np.ndarray = pts_func(n, rng=rng)
    x, y = pts[:, 0], pts[:, 1]
    width, height = np.array(sizes_func(x=x, y=y))
    return np.c_[x, y, width, height, width*height]

def find_anchors_and_crop(arr):
    def cut_to_01(a: np.ndarray):
        """ Return : (x, y, width, height, area) such that
                (x, y) is the xy argument for matplotlib.patches.Rectangle
                width, height are are now cropped thus they do not reach
                    outside (0,1) x (0,1) square
                area is updated accordingly to new sides
            Arguments : a = (x, y, width, height, area)
                x, y - center of a rectangle
                width, height - parameters for matplotlib.patches.Rectangle
                area - width * height
        """
        x, y, width, height, area = a
        x = max(x - width / 2, 0)
        y = max(y - height / 2, 0)
        width = min(width , abs(x-1))
        height= min(height , abs(y-1))
        return (x, y, width, height, width * height)
    return np.apply_along_axis(cut_to_01, 1, arr)

def sort_by_area(a:np.ndarray):
    return a[a[:, -1].argsort()]

def intersect_Q(rects0 : np.ndarray, rects1 : np.ndarray):
    """ Returns array res : np.ndarray (N, M)(check?) such that:
            res[i, j] = 1 if i-th rectangle do intersect j-th rectangle
            res[i, j] = 0 if i-th rectangle do not intersect j-th rectangle
        Arguments:
            rects0 = (x, y, width, height,...) of shape (N,k) 
            rects0 = (x, y, width, height,...) of shape (M,k) 
        """
    A_xmin, A_xmax = rects0[:,0], rects0[:,0] + rects0[:,2]
    A_ymin, A_ymax = rects0[:,1], rects0[:,1] + rects0[:,3]
    B_xmin, B_xmax = rects1[:,0], rects1[:,0] + rects1[:,2]
    B_ymin, B_ymax = rects1[:,1], rects1[:,1] + rects1[:,3]

    def func_on_pairs(func, arr0, arr1):
        pairs = np.transpose(np.array(np.meshgrid(arr0, arr1)), axes = (2,1,0))
        return np.apply_along_axis(func, -1, pairs)
    min_xmax = func_on_pairs(min, A_xmax, B_xmax)
    max_xmin = func_on_pairs(max, A_xmin, B_xmin)
    dx = min_xmax - max_xmin
    min_ymax = func_on_pairs(min, A_ymax, B_ymax)
    max_ymin = func_on_pairs(max, A_ymin, B_ymin)
    dy = min_ymax - max_ymin
    return (dx>0).astype(int) * (dy>0).astype(int)

def remove_smaller(arr:np.ndarray, intersect_Q:Callable=intersect_Q, flipQ:bool=True):
    """ Return array of disjoint rectangles keeping the order:
            i-th row of arr represent rectangle, call it R[i]
            if i < j and R[i] do intersect R[j] then remove R[j]. 
        Arguments
            arr : of shape (N,m) with rows (x, y, width, height,...) of shape (N,m)
            intersect_Q : function that produces every to every intersection array
            flipQ : switch to change the order of the rectangles
        """
    if flipQ is True:
        arr = np.flip(arr, axis=0)

    # intersect_arr is 1 @ (i,j) if R[i} and R[j] are disjoint, 0 otherwise 
    intersect_arr = 1 - intersect_Q(arr, arr) 

    # column with ones at the end to  keep track if we keep or remove that row
    res = np.pad(arr, pad_width=((0,0),(0,1)), mode='constant', constant_values=1)
    for i, rect in enumerate(arr[:-1]):
        # check if i-th have not been removed already 
        if np.isclose(res[i, -1], 1):
            # put 0 at rows which intersect with the current
            res[i+1:,-1] = res[i+1:,-1] * intersect_arr[i, i+1:]
    # rows with 0 at the end are to be removed:
    res = res[res[:,-1] > 0]
    return res[:,:-1]

def inflate_rectangles(arg_arr:np.ndarray):
    """ Return array of rectangles after applying the following procedure 
        to every rectangle in order they appear in arg_arr:
        1) first maximal homogeneous scaling from the midpoint
            such that they do not overlap
                another rectangle or outside boundary square;
        2) maximally push every wall parallelly again not to
            overlap aother rectangle or boundary.
        """
    arr = np.copy(arg_arr)
    for i in range(len(arr)):
        hom_scales_in_dirs = []
        for dir in ['l', 'r', 'u', 'd']:
            scale = homogeneous_scale_in_dir_search(i, arr, dir, 'scale')
            hom_scales_in_dirs.append(scale)
        hom_scale = min(hom_scales_in_dirs)
        arr[i,:4] = np.array(homogeneus_push_all(arr[i,:4], hom_scale))

        for dir in ['l', 'r', 'u', 'd']:
            push_scale = homogeneous_scale_in_dir_search(i, arr, dir, 'push')
            arr[i,:4] = np.array(wall_push(arr[i, :4], push_scale, dir))
    return arr

# def process(arr: np.ndarray):
#     arr = find_anchors_and_crop(arr)
#     arr = sort_by_area(arr)
#     arr = remove_smaller(arr)
#     clinched_rectangles = inflate_rectangles(arr)
#     output = {
#         'arr' : arr,
#         'clinched_rectangles' : clinched_rectangles,
#     }
#     return output
import numpy as np
from SquareDivision.src.regions import region_condition
from typing import Literal
mode = Literal['r', 'u', 'l', 'd']

# def add_01_rect(arr):
#     """Return array with row representing (0, 1) x (0, 1)"""
#     pass

# def right_and_top_rect_points(arr: np.ndarray):
#     """ Calculate <x + width> and <y + height> columns and add them to <arr> at the end
#         Return
#             ext_arr : extended array, rows of the form
#                 (x, y, width, height, ..., x + width, y + height )
#         Arguments
#             arr : array of rectangles rows of the form 
#                 (x, y, width, height, ...)
#     """
#     return np.c_[arr, arr[:,0] + arr[:,2], arr[:,1] + arr[:,3] ]

# FIX what if no points in the region

# max homogeneous expanding in the direction
# def max_homog_push_in(rect_num:int, ext_arr:np.ndarray, dir:mode):
#     """ Find maximal homogeneous scaling possible in the direction dir.
#     Return (scale : float, outer_edge_Q : bool )
#         scale    - the maximal homogeneous scaling in the direction <dir>
#             of a rectangle represented as the <rect_num> row of the array <arr>
#         edge_Q   - indicator if scaling is on another rectangle or 
#             boundary of (0, 1) x (0, 1) square.
#     Arguments
#         rect_num - number or rectangle in ext_arr;
#         ext_arr      - extended array of rectangles having form
#             (x, y, width, height, ..., x + width, y + height )
#         dir      - direction, one of: left, right, up or down
#     """
#     rect = ext_arr[rect_num]        # current rectangle
#     xmid, ymid = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
#     mid = np.array([xmid, ymid])    # current midpoint
#     # remove row with current rect_num from arr:
#     ext_arr = np.delete(arr=ext_arr, obj=rect_num, axis=0)
#     dir_stop_codes = {
#         # for every direction expanding wall may hit only some points, 
#         # for example moving left ('l') wall may hit only:
#         #   right down  (['r', 'd']) or 
#         #   right up    (['r', 'u']) 
#         # corners of another rectangle.
#             'l' : [[-2, 1], [-2, -1]], #[['r', 'd'], ['r', 'u']],
#             'r' : [[ 0, 1], [ 0, -1]], #[['l', 'd'], ['l', 'u']],
#             'u' : [[ 0, 1], [-2,  1]], #[['l', 'd'], ['r', 'd']],
#             'd' : [[ 0,-1], [-2, -1]]  #[['l', 'u'], ['r', 'u']]
#             }
#     inds = dir_stop_codes[dir]
#     pts_for_dir = np.vstack((ext_arr[:, inds[0]], ext_arr[:, inds[1]]))
#     in_dir_Q = region_condition(*rect[:4], dir=dir)(*pts_for_dir.T)
#     # if pts_for_dir would be [] i.e. no points to stop => scale to (0, 1)
#     if np.sum(in_dir_Q) == 0:
#         pt_00 = np.array([0, 0])
#         pt_11 = np.array([1, 1])
#         if dir == 'l' or dir == 'd':
#             # 0 is width 1 is height
#             ind = 0 if dir == 'l' else 1 
#             abs_min = abs((pt_00 - mid)[ind])
#             rect_size = rect[2 + ind] 
#         elif dir == 'r' or dir == 'u':
#             # 0 is width 1 is height
#             ind = 0 if dir == 'r' else 1 
#             abs_min = abs((pt_11 - mid)[ind])
#             rect_size = rect[2 + ind]
#         rel_scale = (2 * abs_min)/rect_size
#         edge_Q = True
#         return rel_scale, edge_Q
    
#     pts_for_dir = pts_for_dir[in_dir_Q==True]
#     if dir == 'l' or dir == 'r':
#         # min of first coord
#         abs_min = np.min(np.abs((pts_for_dir - mid)[:, 0]))
#         rect_size = rect[2] # width
#     elif dir == 'u' or dir == 'd':
#         # min of second coordinate
#         abs_min = np.min(np.abs((pts_for_dir - mid)[:, 1]))
#         rect_size = rect[3] # height
#     rel_scale = (2 * abs_min)/rect_size
#     edge_Q = False
#     return rel_scale, edge_Q

# def find_room(rect_num:int, ext_arr:np.ndarray): #, directions:list = ['r', 'u', 'l', 'd']):
#     """ Calculate homogeneous scaling coefficients in all directions 
#         of the <rect_num>-th row of the array <ext_arr>.
#     Return  -  (dir_scales : List[float], dir_edge_Q : List[bool])
#         dir_scales : scalings in the direction
#         dir_edge_Q : a list of bool where
#             True - scaling stopped on (0, 1) boundary
#             False - scaling stopped on another rectangle
#     Arguments
#         rect_num - number or rectangle in ext_arr;
#         ext_arr      - array of rectangles
#     """
#     directions:list = ['r', 'u', 'l', 'd']
#     # for dir in directions:
#         # if dir not in ['r', 'u', 'l', 'd']:
#         #     raise NameError(f" Direction {dir} not in ['r', 'u', 'l', 'd'].")
#     dir_scales = []
#     dir_edge_Q = []
#     for dir in directions:
#         scale_edge_Q_tup = max_homog_push_in(rect_num, ext_arr, dir)
#         dir_scales.append(scale_edge_Q_tup[0])
#         dir_edge_Q.append(scale_edge_Q_tup[1])
#     return dir_scales, dir_edge_Q

def wall_push(rect, scale, dir):
    """ Return modification of rect = (x, y, width, height).
        Parameter <scale> is homogeneus scaling coefficient from 
        the center of the rectangle, but only wall in <dir> is moved.

    Argument example when dir == 'r' (right wall):
        :                 +---------------+--------+
        :                 |               |        |
        :               height            | -----> |
        :                 |               |        |
        :          (xy) = +---- width ----+--------+
        :       +---------- scale * width ---------+
    Return 
        :                 +------------------------+
        :                 |                        |
        :               height                     |
        :                 |                        |
        :          (xy) = +--------- width --------+

    Argument Example when dir == 'l' (left wall):
        :       +---------+---------------+
        :       |         |               |
        :       | <------ |             height
        :       |         |               |
        :       +- (xy) = +---- width ----+
        :       +---------  scale * width ---------+
    Return 
        :       +-------------------------+
        :       |                         |
        :     height                      |
        :       |                         |
        : (xy)= +--------- width ---------+
    """
    x, y, width, height = rect[:4]
    if dir == 'l' or dir == 'r':
        width_change = (scale * width - width) / 2
        width = width + width_change
        x = x - width_change  if  dir == 'l' else x
        return x, y, width, height
    if dir == 'u' or dir == 'd':
        height_change = (scale * height - height) / 2
        height = height + height_change
        y = y - height_change  if  dir == 'd' else y
        return x, y, width, height

def homogeneus_push_all(rect, scale):
    """ Returns scaled rect with same midpoint. 
    Arguments: 
        rect    :   np.ndarray of the form  (x, y, width, height)
        acale   :   float
    Return
        :       +----------------------------------+ 
        :       |                 ^                |
        :       |                 |                | 
        :       |         +---------------+        |
        : scale * height  |               |        |
        :       |  <--  height            |   -->  |
        :       |         |               |        |
        :       |  (xy) = +---- width ----+        |
        :       |                 |                |
        :       |                 V                |
        : (xy)= +---------- scale * width ---------+
    """
    x, y, width, height = rect[:4]
    width_change = (scale * width - width) / 2
    height_change = (scale * height - height) / 2
    x = x - width_change
    y = y - height_change
    width, height = scale * width, scale * height
    return x, y, width, height


# def morph_rect(rect_num, ext_arr):
#     """ Return row rectangle after scalings coded in 
#         the lists: <dir_scales>, <dir_edge_Q>.
#         The morphing strategy:
#             first homogeneous in all directions
#             secondly if somewhere hiting the 0 or 1 scale that wall there """
#     dir_scales, dir_edge_Q = direction_scalings(rect_num, ext_arr)
#     rect = ext_arr[rect_num]
#     x, y, width, height = rect[:4]
#     xmid, ymid = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
#     hom_scale = min(dir_scales)
#     width, height = hom_scale * width, hom_scale * height
#     x, y = xmid - width / 2, ymid - height / 2

#     return np.array()
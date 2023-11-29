from functools import partial
import numpy as np
from typing import Literal


# (u,v) are variables, (x,y,w,h) <- fixed parameters defining rectangle
# cuting region functions
def left(u: np.ndarray, x: np.ndarray, w: np.ndarray):
    return u <= x + w / 2


def right(u: np.ndarray, x: np.ndarray, w: np.ndarray):
    return u >= x + w / 2


def up(v: np.ndarray, y: np.ndarray, h: np.ndarray):
    return v >= y + h / 2


def down(v: np.ndarray, y: np.ndarray, h: np.ndarray):
    return v >= y + h / 2


def above_positive_slope(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
):
    return v >= (h / w) * (u - x - w / 2) + y + h / 2


def below_positive_slope(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
):
    return v <= (h / w) * (u - x - w / 2) + y + h / 2


def above_negative_slope(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
):
    return v >= -(h / w) * (u - x - w / 2) + y + h / 2


def below_negative_slope(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
):
    return v <= -(h / w) * (u - x - w / 2) + y + h / 2


# min(abs(push_rectangles[left(push_rectangles[:,5], push_rectangles[0,0], push_rectangles[0, 2]),:][:,0]-push_rectangles[0,0]))
direction = Literal[
    "l", "r", "u", "d"
]  # left, right, up, down ( FIX to [left, right, up, down]? )


def region_condition(x, y, w, h, dir: direction):
    """For a rectangle R described by (x, y, w, h) return a function regionQ that
    yields True if points fed to it are are in the
    triangular region - the way of one of the sides (dir parameter)
        during homogeneous expanding of the rectangle R from its center

    Use Example:

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
        from SquareDivision.config import figure_settings, axis_settings
        from SquareDivision.src.regions import region_condition

        arr = np.array([
            [0.58766166, 0.75081476, 0.1744393 , 0.23920043, 0.04172596],
            [0.75821156, 0.41412042, 0.2217599 , 0.13709804, 0.03040285],
            [0.58446222, 0.45747272, 0.17355159, 0.15024461, 0.02607519],
            [0.38227909, 0.56799425, 0.11745406, 0.1837602 , 0.02158338],
            [0.8057687 , 0.27159525, 0.1942313 , 0.09387736, 0.01823392],
            [0.75340777, 0.18842556, 0.22042705, 0.0686562 , 0.01513368],
            [0.22481088, 0.83214705, 0.07376309, 0.16785295, 0.01238135],
            [0.24181214, 0.3167267 , 0.07848025, 0.10756345, 0.00844161],
            [0.32887713, 0.20493496, 0.10263721, 0.07366267, 0.00756053],
            [0.80471225, 0.07044389, 0.19528775, 0.03287832, 0.00642073],
            [0.12977667, 0.40156535, 0.047395  , 0.13329073, 0.00631731],
            [0.04781207, 0.84403462, 0.02465318, 0.15596538, 0.00384504],
            [0.04781089, 0.21318488, 0.02465285, 0.07616445, 0.00187767],
            [0.53157869, 0.        , 0.15887858, 0.0108248 , 0.00171983]
            ])
        i = 2
        reg = region_condition(*arr[i,:4], dir='l')(*arr[:,:2].T)
        rect_arg = arr[i]
        x, y, w, h = *rect_arg[:4],

        #lines
        x_mid, ymid = x + w/2, y + h/2
        x00 = 0
        y00 =  (h/w) * (x00 - x - w/2) + y+h/2
        x01 = x_mid
        y01 = (h/w) * (x01 - x - w/2) + y+h/2

        x10 = 0
        y10 =  -(h/w) * (x10 - x - w/2) + y+h/2
        x11 = x_mid
        y11 = -(h/w) * (x11 - x - w/2) + y+h/2

        fig:Figure
        ax: Axes
        fig, ax = plt.subplots(**figure_settings)
        ax.set(**axis_settings)
        ax.plot( [x00,x01], [y00,y01], c='k')
        ax.plot( [x10,x11], [y10,y11], c='k')
        ax.scatter(arr[:,0][reg==True], arr[:,1][reg==True], c='r')
        ax.scatter(arr[:,0][reg==False], arr[:,1][reg==False], c='tab:blue')
        rect = Rectangle(xy=(x, y),
                            width=w,
                            height=h,
                            fill=False,
                            fc='w',ec='k')
        ax.add_patch(rect)
        plt.show()

    """
    switcher = {
        "l": ["above_positive_slope", "below_negative_slope"],
        "r": ["below_positive_slope", "above_negative_slope"],
        "u": ["above_positive_slope", "above_negative_slope"],
        "d": ["below_positive_slope", "below_negative_slope"],
    }

    def regionQ(u: np.ndarray, v: np.ndarray):
        """How to call:
        rect = (x,y, width, height) #the rectangle from which we construct regions
        dir = 'l' #the direction from 'l', 'r', 'u', 'd'
        # arr of shape (*,2+) such that first two columns are:
        #   X, Y coordinates we want to test
        pts = arr[:,:2].T
        reg_Q = region_condition(*rect, dir=dir)(*pts) # the call
        """
        cond0 = partial(eval(switcher[dir][0]), x=x, y=y, w=w, h=h)
        cond1 = partial(eval(switcher[dir][1]), x=x, y=y, w=w, h=h)
        result0: np.ndarray = cond0(u, v)
        result1: np.ndarray = cond1(u, v)
        return result0.astype(int) * result1.astype(int)

    return regionQ


def wall_axis(dir: direction):
    idx = 0 if dir == "l" or dir == "r" else 1
    return idx


def perp_axis(dir: direction):
    return (wall_axis(dir) + 1) % 2


def opposing_walls_in_half_plane_in_dir(rect_num, rect_arr, dir: direction):
    """
    Return indices (list of True/False of length = len(rect_arr))
    of and intervals representing walls opposed to <dir>:
            right walls for dir = 'l' (left),
            upper walls for dir = d (down), and so on
        the half plane of rect in direction <dir>.
            Example for where function searchs other rectangles for dir='l'
            :  /  /  /  /  /  |
            : /  /  /  /  /  /|
            :/ Search here  / +---------------+
            :  /  /  /  /  /  |               |
            :--+ /  /  /  / height            |
            :  |/  /  /  /  / |               |
            :  |  /  / (xy) = +---- width ----+
            :--+ /  /  /  /  /|
            :/  /  /  /  /  / |
            Return array of rows: (anchor,start,stop) where rows represent:
                In case dir is 'l' or 'r'
            :          ^
            :          |
            :     +    - stop
            :     |    |
            :     |    |
            :     +    - start
            :          |
            :-----|----+---->
            :  anchor  |

                In case dir is 'u' or 'd'
                        :          ^
            :               ^
            :               |
            :               |
            :   +-------+   - anchor
            :               |
            :               |
            :---|-------|---+---->
            : start    stop

    Example:
        import numpy as np
        from numpy.random._generator import Generator
        import functools
        from SquareDivision.src.generators import uniform_pts
        from SquareDivision.src.distributions import x_plus_y_func
        from SquareDivision.src.dataflow import arg_rect_list, process, struc_arr
        from SquareDivision.config import config

        rng:Generator = np.random.default_rng(config['seed'])
        func = functools.partial(x_plus_y_func,
                                min_00=0.05, max_00=0.05,
                                min_11=0.35, max_11=0.4,
                                rng=rng)

        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
        from SquareDivision.config import figure_settings, axis_settings
        from SquareDivision.src.regions import opposing_walls_in_half_plane_in_dir

        arr = arg_rect_list(5, uniform_pts, func, rng=rng)
        arr = process(arr)

        suspect, dir= 5, 'u'
        which_list, sus_walls = opposing_walls_in_half_plane_in_dir(suspect, arr, dir)

        data = struc_arr(arr)
        fig:Figure
        ax: Axes
        fig, ax = plt.subplots(**figure_settings)
        ax.set(**axis_settings)
        for i, arg in enumerate(data):
            rect = Rectangle(xy=(arg['x'], arg['y']),
                            width=arg['width'],
                            height=arg['height'],
                            fill=True,
                            fc='w',
                            ec='k' if i!=suspect else 'r')
            ax.add_patch(rect)
            # if i == 1:
            #     break
        ax.vlines([0,1],ymin=0,ymax=1)
        ax.hlines([0,1],xmin=0,xmax=1)

        if dir == 'r' or dir == 'l':
            ax.scatter(*sus_walls[:,:2].T, marker='.', c='r')
        if dir == 'd' or dir == 'u':
            ax.scatter(*sus_walls[:,[1,0]].T, marker='.', c='r')

        plt.show()

    """
    rect = rect_arr[rect_num]
    idx = wall_axis(dir)

    if dir == "r" or dir == "u":
        # if right or up we search for x's or y's
        # greater or eq. than rectangles x + width or y+ height
        inds = np.logical_or(
            rect_arr[:, idx] >= rect[idx] + rect[idx + 2],
            np.isclose(rect_arr[:, idx], rect[idx] + rect[idx + 2]),
        )
        anchors = rect_arr[inds, idx]
    else:
        # if left or down we search for (x + width)'s or (y + height)'s
        # less or eq. than rectangles x or y
        inds = np.logical_or(
            rect_arr[:, idx] + rect_arr[:, idx + 2] <= rect[idx],
            np.isclose(rect_arr[:, idx] + rect_arr[:, idx + 2], rect[idx]),
        )
        anchors = rect_arr[inds, idx] + rect_arr[inds, idx + 2]
    perp_idx = perp_axis(dir)
    starts = rect_arr[inds, perp_idx]
    ends = starts + rect_arr[inds, perp_idx + 2]
    return inds, np.c_[anchors, starts, ends]


def intersect_intervals(a0, a1, b0, b1):
    """Return True if intervals (a0, a1) and (b0, b1) intersect."""
    a_min, a_max = min(a0, a1), max(a0, a1)
    b_min, b_max = min(b0, b1), max(b0, b1)
    min_max = min(a_max, b_max)
    max_min = max(a_min, b_min)
    dx = min_max - max_min
    return dx >= 0, (max_min, min_max)


# def positive_diag_a_b(x: float, y: float, w: float, h: float):
#     """Return (a, b)  such that a.x + b == 0 is diagonal A,C"""
#     return np.array((-h, w)), h * x - w * y


# def positive_diag(u: float, v: float, x: float, y: float, w: float, h: float):
#     """positive_diag(u,v) > 0 are points (u,v) above diagonal passing through A, C"""
#     a, b = positive_diag_a_b(x, y, w, h)
#     return np.dot(a, [u, v]) + b


# def negative_diag_a_b(x: float, y: float, w: float, h: float):
#     """Return (a, b)  such that a.x + b == 0 is diagonal B,D"""
#     return np.array((h, w)), -h * w - h * x - w * y


# def negative_diag(u: float, v: float, x: float, y: float, w: float, h: float):
#     """negative_diag(u,v) > 0 are points (u,v) above diagonal passing through B, D"""
#     a, b = negative_diag_a_b(x, y, w, h)
#     return np.dot(a, [u, v]) + b


def walls_in_view(rect: np.ndarray, dir: direction, suspended_walls: np.ndarray):
    """Return those suspended walls which are in the push view from the rectangle rect
    Arguments
        suspended_walls :   np.ndarray (N, 3) representing
            (x, y_min, y_max) vertical   interval from (x, y_min) to (x, y_max)
                or
            (y, x_min, x_max) horizontal interval from (y, x_min) to (y, x_max)
    """
    idx = wall_axis(dir)
    perp_idx = perp_axis(dir)
    s, t = rect[perp_idx], rect[perp_idx] + rect[perp_idx + 2]
    wall = [s, t]
    if len(suspended_walls) == 0:
        return np.array([]), np.array([]).reshape(0, 2)
    have_intersection = np.apply_along_axis(
        func1d=(lambda vec: intersect_intervals(*wall, *vec)[0]),
        axis=1,
        arr=suspended_walls[:, 1:],
    )
    intersections = np.apply_along_axis(
        func1d=(lambda vec: intersect_intervals(*wall, *vec)[1]),
        axis=1,
        arr=suspended_walls[:, 1:],
    )
    return suspended_walls[have_intersection, 0], intersections[have_intersection]


def walls_in_scaled_view(rect: np.ndarray, dir: direction, suspended_walls: np.ndarray):
    """Return those suspended walls which have non-empty intersection with
    triangular region spanned by <dir> wall with vertex at center of rectangle
    Arguments
        suspended_walls :   np.ndarray (N, 3) representing
            (x, y_min, y_max) vertical   interval from (x, y_min) to (x, y_max)
                or
            (y, x_min, x_max) horizontal interval from (y, x_min) to (y, x_max)
    """
    idx = wall_axis(dir)
    perp_idx = perp_axis(dir)
    mid_pt = rect[:2] + rect[2:4] / 2
    # s, t = rect[perp_idx], rect[perp_idx] + rect[perp_idx + 2]
    # wall, mid = [s, t], (t+s)/2
    anchors = suspended_walls[:, 0]
    slope = rect[perp_idx + 2] / rect[idx + 2]
    scaled_wall_in_anchors_as = (
        mid_pt[perp_idx] + np.abs(mid_pt[idx] - anchors) * (-1) * slope
    )
    scaled_wall_in_anchors_bs = mid_pt[perp_idx] + np.abs(mid_pt[idx] - anchors) * slope
    # scaled_suspended_walls = np.c_[scaled_wall_in_anchors_as, scaled_wall_in_anchors_bs]
    # return np.c_[anchors, scaled_wall_in_anchors_as, scaled_wall_in_anchors_bs]
    intervals_at_anchors = np.c_[
        suspended_walls[:, 1:], scaled_wall_in_anchors_as, scaled_wall_in_anchors_bs
    ]
    if len(suspended_walls) == 0:
        return np.array([]), np.array([]).reshape(0, 2)
    have_intersection = np.apply_along_axis(
        func1d=(lambda a0_b0_a1_b1: intersect_intervals(*a0_b0_a1_b1)[0]),
        axis=1,
        arr=intervals_at_anchors,
    )
    intersections = np.apply_along_axis(
        func1d=(lambda a0_b0_a1_b1: intersect_intervals(*a0_b0_a1_b1)[1]),
        axis=1,
        arr=intervals_at_anchors,
    )
    return suspended_walls[have_intersection, 0], intersections[have_intersection]


mode = Literal["scale", "push"]


def homogeneous_scale_in_dir_search(
    rect_num: int, rect_arr: np.ndarray, dir: direction, scale_or_push: mode
):
    """Return maximal scale to the the first obstacle from rect_arr
        in <scale_or_push> expanding of rectangle <rect>
        in the direction <dir>.
    Arguments
        rect_num        :   number of rectangle in rectangle list
        arr             :   array of rectangles (x, y, width, height,...)
        dir             :   one of ['l', 'r', 'u', 'd']
        scaled_or_fixed :   one of ['scale', 'push'] switch to
            search obstacles on strait wall push or homogeneus expansion.
    """
    idx = wall_axis(dir)
    rect = rect_arr[rect_num, :4]
    x, y, width, heigth = rect[:4]
    mid_pt = np.array([x + width / 2, y + heigth / 2])
    idx_s, all_walls = opposing_walls_in_half_plane_in_dir(rect_num, rect_arr, dir)
    if scale_or_push == "scale":
        (barrier_walls_anchors, barrier_walls_intersections) = walls_in_scaled_view(
            rect, dir, all_walls
        )
    if scale_or_push == "push":
        (barrier_walls_anchors, barrier_walls_intersections) = walls_in_view(
            rect, dir, all_walls
        )
    # barrier_walls_anchors are places on <idx> axis, when we found
    if len(barrier_walls_anchors) > 0:
        # we found something
        dist_to_mid = np.min(np.abs(barrier_walls_anchors - mid_pt[idx]))
        scale = 2 * dist_to_mid / [width, heigth][idx]
        return scale
    else:  # len(barrier_walls_anchors) is 0
        barrier_anchor = 0 if (dir == "l" or dir == "d") else 1
        dist_to_mid = np.abs(mid_pt[idx] - barrier_anchor)
        scale = 2 * dist_to_mid / [width, heigth][idx]
        return scale

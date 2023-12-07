import numpy as np
from scipy.optimize import LinearConstraint


def low_boundary_constraint_args(
    clinched_rectangles: np.ndarray, upper_neighbours: np.ndarray, axis: int
):
    """Return table and rhs for constraints: (examples for X axis)
        x_i == 0
    for those rectangles which are on:
        left side (axis = 0)
        bottom (axis = 1)
    of [0, 1] x [0, 1] square.
    Arguments :
        clinched_rectangles : (n, 4) array [x, y, width, height]
        upper_neighbours    : (n, n) array
            i-th row represet the east or north neighbours of i-th rectangle
            placing 1 in j-th place if i-th rectangle is in contact with j-th rectangle
            (incidence matrix of horizontal or vertical contact grah between rectangles)
        axis                : indicator if we mean horizontal or vertical lower boundary.
    """
    shape = clinched_rectangles.shape
    rects_on_low_boundary = np.where(np.sum(upper_neighbours, axis=0) == 0)[0]
    num_of_EQ = len(rects_on_low_boundary)
    # shape of output
    arg_A = np.zeros(shape=(num_of_EQ, np.prod(shape)))
    rhs = np.zeros(shape=(num_of_EQ,))
    # where are low_boundary rects positions after flattening:
    idxs = np.ravel_multi_index((rects_on_low_boundary, axis), shape)
    for num, idx in enumerate(idxs):
        arg_A[num, idx] = 1
    return arg_A, rhs


def high_boundary_constraint_args(
    clinched_rectangles: np.ndarray, upper_neighbours: np.ndarray, axis: int
):
    """Return table and rhs for constraints: (examples for X axis)
        x_i + w_i == 1
    for those rectangles which are on:
        right side (axis = 0)
        top (axis = 1)
    of [0, 1] x [0, 1] square.
    """
    shape = clinched_rectangles.shape
    rects_on_high_boundary = np.where(np.sum(upper_neighbours, axis=1) == 0)[0]
    num_of_EQ = len(rects_on_high_boundary)
    # shape of output
    arg_A = np.zeros(shape=(num_of_EQ, np.prod(shape)))
    rhs = np.ones(shape=(num_of_EQ,))
    # where are high_boundary rects positions after flattening:
    idxs = np.ravel_multi_index((rects_on_high_boundary, axis), shape)
    for num, idx in enumerate(idxs):
        arg_A[num, [idx, idx + 2]] = 1
    return arg_A, rhs


def contact_constraint_args(
    clinched_rectangles: np.ndarray, upper_neighbours: np.ndarray, axis: int
):
    """Return table and rhs for constraints: (examples for X axis)
    if i-th row of upper_neighbours have 1 in k-th column it means
    that k-th rectangle is one of upper neighbours of i-th rectangle threfore
        x_i + w_i - x_k == 0
    Return:
        contact_arr of shape (sum of upper_neighbours, product of clinched_rectangles.shape)
        contact_rhs of shape (sum of upper_neighbours,)
    """
    # n, cols = clinched_rectangles.shape
    shape = clinched_rectangles.shape
    arg_len = np.prod(shape)
    m = int(np.sum(upper_neighbours))
    contact_arr = np.zeros(shape=(m, arg_len))
    for contact_num, (low_neigh, high_neigh) in enumerate(
        zip(*np.where(upper_neighbours > 0))
    ):
        # low_position + low_size - high_position == ...
        lpos, lsize, hpos = np.ravel_multi_index(
            [[low_neigh, low_neigh, high_neigh], [axis, axis + 2, axis]], shape
        )
        contact_arr[contact_num, [lpos, lsize, hpos]] = np.array([1, 1, -1])
    # ... == 0
    contact_rhs = np.zeros(shape=(m,))
    return contact_arr, contact_rhs


def hole_width_height(hole_closing_idxs, clinched_rectangles: np.ndarray):
    """Return sizes (width, height) of a hole between
    rectangles numbered hole_closing_idxs"""
    [[left, right], [down, up]] = hole_closing_idxs
    l, r = clinched_rectangles[left], clinched_rectangles[right]
    d, u = clinched_rectangles[down], clinched_rectangles[up]
    width = r[0] - (l[0] + l[2])
    height = u[1] - (d[1] + d[3])
    return width, height


def linear_args_closing_holes_brutal(
    hole_closing_idxs: list, clinched_rectangles: np.ndarray
):
    """Return A , lb, ub for LinearConstraint class closing hole stored in <hole_closing_idxs>"""
    shape = clinched_rectangles.shape
    arg_len = np.prod(shape)
    arg_A = np.zeros(shape=(arg_len))
    lb, ub = 0, 0
    [[left, right], [down, up]] = hole_closing_idxs
    # hole prospecting
    hole_size = hole_width_height(hole_closing_idxs, clinched_rectangles)
    axis_to_close = np.argmin(hole_size)
    if axis_to_close == 0:
        # width is smaller => contact left to right : rx - (lx + lw) == 0
        rx, lx, lw = np.ravel_multi_index([[right, left, left], [0, 0, 2]], shape)
        arg_A[[rx, lx, lw]] = np.array([-1, 1, 1])
        return arg_A, lb, ub
    else:
        # height is smaller => contact down to up : uy - (dy + dh) == 0
        uy, dy, dh = np.ravel_multi_index([[up, down, down], [1, 1, 3]], shape)
        arg_A[[uy, dy, dh]] = np.array([-1, 1, 1])
        return arg_A, lb, ub


def linear_constraint(
    clinched_rectangles,
    east_neighbours,
    north_neighbours,
    idxs_to_close,
    keep_feasible=True,
):
    """Joining linear constraints into scipy.optimize.LinearConstraint object"""
    lhs = []
    rhs = []
    low__X_A, low__X_rhs = low_boundary_constraint_args(
        clinched_rectangles, east_neighbours, axis=0
    )
    lhs.append(low__X_A)
    rhs.append(low__X_rhs)
    low__Y_A, low__Y_rhs = low_boundary_constraint_args(
        clinched_rectangles, north_neighbours, axis=1
    )
    lhs.append(low__Y_A)
    rhs.append(low__Y_rhs)
    high_X_A, high_X_rhs = high_boundary_constraint_args(
        clinched_rectangles, east_neighbours, axis=0
    )
    lhs.append(high_X_A)
    rhs.append(high_X_rhs)
    high_Y_A, high_Y_rhs = high_boundary_constraint_args(
        clinched_rectangles, north_neighbours, axis=1
    )
    lhs.append(high_Y_A)
    rhs.append(high_Y_rhs)
    cont_X_A, cont_X_rhs = contact_constraint_args(
        clinched_rectangles, east_neighbours, axis=0
    )
    lhs.append(cont_X_A)
    rhs.append(cont_X_rhs)
    cont_Y_A, cont_Y_rhs = contact_constraint_args(
        clinched_rectangles, north_neighbours, axis=1
    )
    lhs.append(cont_Y_A)
    rhs.append(cont_Y_rhs)

    holes__A = []
    holes_lb = []
    holes_ub = []
    for idx_pair in idxs_to_close:
        arg_A, lb, ub = linear_args_closing_holes_brutal(idx_pair, clinched_rectangles)
        holes__A.append(arg_A)
        holes_lb.append(lb)
        holes_ub.append(ub)

    lhs.extend(holes__A)
    rhs.extend(holes_lb)
    A = np.vstack(lhs)
    lb = np.hstack(rhs)
    ub = np.hstack(rhs)
    constr = LinearConstraint(A=A, lb=lb, ub=ub, keep_feasible=keep_feasible)
    return constr

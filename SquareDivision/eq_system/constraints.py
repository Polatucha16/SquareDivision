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


# region LEGACY for minimalization distance
#
# import networkx as nx
# from SquareDivision.contact_graph.incidence_matrix import contact_graph_incidence_matrix
# from SquareDivision.holes.detect import hole_closing_idxs
#
# def hole_orientation(hole_closing_idxs, east_graph:nx.Graph, north_graph:nx.Graph):
#     """ Return if hole is left or right.
#         [[left, right],[down, up] ] = [[i,j], [k, l]] = hole_closing_idxs
#          _______________________________________________________
#         : LEFT HOLE               : RIGHT HOLE                  :
#         :         || up           :           up      ||        :
#         :   left  |+------------- : ------------------+|  right :
#         : --------+       +------ : --------+          +--------:
#         : ---------------+| right :  left   |+------------------:
#         :         down   ||       :         ||    down          :
#         :_________________________:_____________________________:
#         Two possibilities:
#             RIGHT HOLE when:
#             rect. <right> is among east neighbours of rectangle <top>
#         OR
#             LEFT HOLE when:
#             rect <top> is among north neighbours of rectangle <right>
#         """
#     [[left, right], [down, up]] = hole_closing_idxs
#     if (up, right) in east_graph.edges:
#         return 'right'
#     elif (right, up) in north_graph.edges:
#         return 'left'
# def factors_of_4_way_hole(x:np.ndarray, hole_closing_idxs:list, east_graph:nx.Graph, north_graph:nx.Graph):
#     """ Nonlinear constraint from closing hole between:
#             [[left, right], [down, up]].
#         - h -> is horizontal connection
#         - v -> is vertical connection
#         Arguments:
#             hole_closing_idxs : [[ int, int], [int, int]]
#         For four indices of a hole, return multiplication representing,
#         depending on orientation of a hole, concontacts after hole clousure
#         RIGTH HOLE:
#                 top -h-> right
#                   ^         ^
#                   |         |
#                   v         v
#                   |         |
#                 left -h-> down
#             to close we can *add to the connections graph* the following edges
#                 lrh : left  - h -> right
#                 lrv : left  - v -> right
#                 udh : down <- h -  up
#                 duv : down  - v -> up
#         LEFT HOLE:
#                 top <-v- right
#                   ^         ^
#                   |         |
#                   h         h
#                   |         |
#                 left <-v- down
#             to close we can *add to the connections graph* the following edges
#                 lrh : left  - h -> right
#                 rlv : left <- v -> right
#                 duh : down  - h ->  up
#                 duv : down  - v -> up
#         """
#     arr:np.ndarray = x.reshape(-1, 4)
#     [[left, right], [down, up]] = hole_closing_idxs
#     l, r = arr[left], arr[right]
#     d, u, = arr[down], arr[up]
#     orientation = hole_orientation(hole_closing_idxs, east_graph, north_graph)
#     if orientation == 'right':
#         # _ _ _ : 3 letters code: L-eft, R-ight, U-p, D-own, H-orizontal, V-ertival
#         lrh = l[0] + l[2] - r[0]
#         lrv = l[1] + l[3] - r[1] # l[3] here only
#         udh = u[0] + u[2] - d[0] # u[2] here only
#         duv = d[1] + d[3] - u[1]
#         return {'lrh' : lrh,
#                 'lrv' : lrv,
#                 'udh' : udh,
#                 'duv' : duv}
#     elif orientation == 'left':
#         lrh = l[0] + l[2] - r[0]
#         rlv = r[1] + r[3] - l[1] # r[3] here only
#         duh = d[0] + d[2] - u[0] # d[2] here only
#         duv = d[1] + d[3] - u[1]
#         return {'lrh' : lrh,
#                 'rlv' : rlv,
#                 'duh' : duh,
#                 'duv' : duv}
#     # caution (LRH <=> DUH) and (LRV <=> DUV) because of contacts forming hole
# def closing_holes(x:np.ndarray, hole_closing_idxs:list, east_graph:nx.Graph, north_graph:nx.Graph):
#     """ CAUTIUON, for variables in conditions, beacuse of contacts forming the hole
#         for right hole we have:
#             (LRH <=> *UDH) and (LRV <=> DUV)
#         for left  hole we have:
#             (LRH <=> DUH) and (*RLV <=> DUV)"""
#     val_dict = factors_of_4_way_hole(x, hole_closing_idxs, east_graph, north_graph)
#     # val = np.prod(np.array(list(val_dict.values())))
#     val = val_dict['lrh'] * val_dict['duv']
#     return val
# def closing_holes_jac(x:np.ndarray, hole_closing_idxs:list, east_graph:nx.Graph, north_graph:nx.Graph):
#     arr = x.reshape(-1,4)
#     jac = np.zeros(shape=arr.shape)
#     [[left, right], [down, up]] = hole_closing_idxs
#     vd = factors_of_4_way_hole(x, hole_closing_idxs, east_graph, north_graph)
#     lrh, duv = vd['lrh'], vd['duv']
#     # lrh = l[0] + l[2] - r[0]
#     # duv = d[1] + d[3] - u[1]
#     jac[[left, right, down, up]] = np.array(
#                [[ duv,   0, duv,   0],
#                 [-duv,   0,   0,   0],
#                 [   0, lrh,   0, lrh],
#                 [   0,-lrh,   0,   0]]
#             )
#     return jac.flatten()
# def closing_holes_4_way_jac(x:np.ndarray, hole_closing_idxs:list, east_graph:nx.Graph, north_graph:nx.Graph):
#     arr:np.ndarray = x.reshape(-1, 4)
#     jac_arr = np.zeros(shape=arr.shape)
#     [[left, right], [down, up]] = hole_closing_idxs
#     orientation = hole_orientation(hole_closing_idxs, east_graph, north_graph)
#     if orientation == 'right':
#         vd = factors_of_4_way_hole(x, hole_closing_idxs, east_graph, north_graph)
#         lrh, lrv, udh, duv = vd['lrh'], vd['lrv'], vd['udh'], vd['duv']
#         m0 = 2*lrh  * lrv**2 * udh**2 * duv**2
#         m1 = lrh**2 * 2*lrv  * udh**2 * duv**2
#         m2 = lrh**2 * lrv**2 * 2*udh  * duv**2
#         m3 = lrh**2 * lrv**2 * udh**2 * 2*duv
#         jac_arr[[left, right, down, up]] = np.array(
#                [[ m0, m1, m0, m1],
#                 [-m0,-m1,  0,  0],
#                 [-m2, m3,  0, m3],
#                 [ m2,-m3, m2,  0]]
#             )
#         return jac_arr.flatten()
#     elif orientation == 'left':
#         vd = factors_of_4_way_hole(x, hole_closing_idxs, east_graph, north_graph)
#         lrh, rlv, duh, duv = vd['lrh'], vd['rlv'], vd['duh'], vd['duv']
#         m0 = 2*lrh  * rlv**2 * duh**2 * duv**2
#         m1 = lrh**2 * 2*rlv  * duh**2 * duv**2
#         m2 = lrh**2 * rlv**2 * 2*duh  * duv**2
#         m3 = lrh**2 * rlv**2 * duh**2 * 2*duv
#         jac_arr[[left, right, down, up]] = np.array(
#                [[ m0,-m1, m0,  0],
#                 [-m0, m1,  0, m1],
#                 [ m2, m3, m2, m3],
#                 [-m2,-m3,  0,  0]]
#             )
#         return jac_arr.flatten()
# def closing_holes_4_way_jac(x:np.ndarray, hole_closing_idxs:list, east_graph:nx.Graph, north_graph:nx.Graph):
#     """ Jacobian of closing_holes_4_way function, linear part in every and every :
#     D_var_i ( f1 * f2 * f3(...,var_i,...) * f4 ) = f1 * f2 * (+/- 1) * f4
#     indices named left, right, down, up have the following variables :
#     arr[left]  have l[0], l[1], l[2], l[3]
#     arr[right] have r[0], r[1], r[2], r[3]
#     and so on ...
#     Now function closing_holes_4_way in the case of left hole is
#         lrh * rlv * duh * duv
#         = (l[0] + l[2] - r[0]) * (r[1] + r[3] - l[1])
#             * (d[0] + d[2] - u[0]) * (d[1] + d[3] - u[1])
#     we see that every variable occurs in the above formula in a single term only
#     therfore for example the partial derivative d/(d l[0]) is equal
#         1* rlv * duh * duv
#     we denothe this as m0 as it is multiplication of terms lacking first.
#     """
#     arr:np.ndarray = x.reshape(-1, 4)
#     jac_arr = np.zeros(shape=arr.shape)
#     [[left, right], [down, up]] = hole_closing_idxs
#     orientation = hole_orientation(hole_closing_idxs, east_graph, north_graph)
#     if orientation == 'right':
#         vd = factors_of_4_way_hole(x, hole_closing_idxs, east_graph, north_graph)
#         lrh, lrv, udh, duv = vd['lrh'], vd['lrv'], vd['udh'], vd['duv']
#         m0 =       lrv * udh * duv
#         m1 = lrh       * udh * duv
#         m2 = lrh * lrv       * duv
#         m3 = lrh * lrv * udh
#         jac_arr[[left, right, down, up]] = np.array(
#                [[ m0, m1, m0, m1],
#                 [-m0,-m1,  0,  0],
#                 [-m2, m3,  0, m3],
#                 [ m2,-m3, m2,  0]]
#             )
#         return jac_arr.flatten()
#     elif orientation == 'left':
#         vd = factors_of_4_way_hole(x, hole_closing_idxs, east_graph, north_graph)
#         lrh, rlv, duh, duv = vd['lrh'], vd['rlv'], vd['duh'], vd['duv']
#         m0 =       rlv * duh * duv
#         m1 = lrh       * duh * duv
#         m2 = lrh * rlv       * duv
#         m3 = lrh * rlv * duh
#         jac_arr[[left, right, down, up]] = np.array(
#                [[ m0,-m1, m0,  0],
#                 [-m0, m1,  0, m1],
#                 [ m2, m3, m2, m3],
#                 [-m2,-m3,  0,  0]]
#             )
#         return jac_arr.flatten()
# def nonlinear_constraints(keep_feasible=True): #idxs_to_close, clinched_rectangles,
#     constr_list = []
#     # area constraint
#     constr_list.append(
#         NonlinearConstraint(
#             fun=area_constraint_fun,
#             jac=area_jac,
#             lb=0, ub=0,
#             keep_feasible=keep_feasible)
#     )
#     return constr_list
# def area_constraint_fun(x:np.ndarray):
#     """ Nonlinear constraints
#         argument x is flattened array of shape <clinched_rectangles>"""
#     arr:np.ndarray = x.reshape(-1, 4)
#     width, height =  arr[:,2], arr[:,3]
#     return 1 - width.dot(height)

# def area_jac(x:np.ndarray):
#     """ Calculates the jacobian of area_constraint_fun at x"""
#     arr:np.ndarray = x.reshape(-1, 4)
#     width, height =  arr[:,2], arr[:,3]
#     jac:np.ndarray = np.zeros(shape=arr.shape)
#     jac[:,2:] = (-1) * arr[:,[3,2]]
#     return jac.flatten()

# def linear_constraints(clinched_rectangles, east_neighbours, north_neighbours, keep_feasible=True):

#     # boundary rectangles constraints
#     low__X_A, low__X_rhs = low_boundary_constraint_args(clinched_rectangles, east_neighbours, axis=0)
#     low__Y_A, low__Y_rhs = low_boundary_constraint_args(clinched_rectangles, north_neighbours, axis=1)
#     high_X_A, high_X_rhs = high_boundary_constraint_args(clinched_rectangles, east_neighbours, axis=0)
#     high_Y_A, high_Y_rhs = high_boundary_constraint_args(clinched_rectangles, north_neighbours, axis=1)
#     low__X_constr = LinearConstraint( A=low__X_A, lb=low__X_rhs, ub=low__X_rhs, keep_feasible=keep_feasible)
#     low__Y_constr = LinearConstraint( A=low__Y_A, lb=low__Y_rhs, ub=low__Y_rhs, keep_feasible=keep_feasible)
#     high_X_constr = LinearConstraint( A=high_X_A, lb=high_X_rhs, ub=high_X_rhs, keep_feasible=keep_feasible)
#     high_Y_constr = LinearConstraint( A=high_Y_A, lb=high_Y_rhs, ub=high_Y_rhs, keep_feasible=keep_feasible)

#     # constacts from constact graphs
#     cont_X_A, cont_X_rhs = contact_constraint_args(clinched_rectangles, east_neighbours, axis=0)
#     cont_Y_A, cont_Y_rhs = contact_constraint_args(clinched_rectangles, north_neighbours, axis=1)
#     horizontal_contacts = LinearConstraint( A=cont_X_A, lb=cont_X_rhs, ub=cont_X_rhs, keep_feasible=keep_feasible)
#     vertical___contacts = LinearConstraint( A=cont_Y_A, lb=cont_Y_rhs, ub=cont_Y_rhs, keep_feasible=keep_feasible)


#     constr_list = [
#         low__X_constr, low__Y_constr,
#         high_X_constr, high_Y_constr,
#         horizontal_contacts,
#         vertical___contacts,
#         ]
#     return constr_list

# def hole_closing_constraints(idxs_to_close, clinched_rectangles, keep_feasible=True):
#     holes_constraints = []
#     for idx_pair in idxs_to_close:
#         arg_A, lb, ub = linear_args_closing_holes_brutal(idx_pair, clinched_rectangles)
#         holes_constraints.append(
#             LinearConstraint(A=arg_A, lb=lb, ub=ub, keep_feasible=keep_feasible)
#         )
#     return holes_constraints
# endregion.

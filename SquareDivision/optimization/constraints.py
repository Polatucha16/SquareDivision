import numpy as np

def low_boundary_constraint_args(
    clinched_rectangles:np.ndarray, 
    upper_neighbours:np.ndarray, 
    axis:int):
    """ Return table and rhs for constraints: (examples for X axis)
            x_i = 0
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
    n, cols = clinched_rectangles.shape
    id_lower = np.zeros(shape=(cols * n, cols * n))
    for rect_num in range(n):
        if np.sum(upper_neighbours.T[rect_num]) == 0:
            # rectangle do not apper as an upper neighbour 
            # of any other rectangle => it is on lower boundary
            idx = cols * rect_num + axis
            id_lower[idx, idx] = 1
    rhs_lower = np.zeros(shape=(cols * n,))
    return id_lower, rhs_lower

def high_boundary_constraint_args(
    clinched_rectangles:np.ndarray, 
    upper_neighbours:np.ndarray, 
    axis:int):
    """ Return table and rhs for constraints: (examples for X axis)
            x_i + w_i = 1
        for those rectangles which are on: 
            right side (axis = 0)
            top (axis = 1)
        of [0, 1] x [0, 1] square.
        """
    n, cols   = clinched_rectangles.shape
    id_upper  = np.zeros(shape=(cols * n, cols * n))
    rhs_upper = np.zeros(shape=(cols * n,))
    for rect_num in range(n):
        if np.sum(upper_neighbours[rect_num]) == 0:
            # rectangle has no upper neighbours => it is on upper boundary
            idx = cols * rect_num + axis
            id_upper[idx, idx], id_upper[idx, idx + 2] = 1, 1
            rhs_upper[ idx ] = 1
    return id_upper, rhs_upper

def contact_constraint_args(
    clinched_rectangles:np.ndarray, 
    upper_neighbours:np.ndarray, 
    axis:int):
    """ Return table and rhs for constraints: (examples for X axis)
        if i-th row of upper_neighbours have 1 in k-th place it means
        that k-th rectangle is one of upper neighbours of i-th rectangle threfore
            x_k - x_i - w_i
        """
    n, cols = clinched_rectangles.shape
    m = np.sum(upper_neighbours)
    contact_arr = np.zeros(shape=(m, cols * n))
    for contact_num, (low_neighbour, high_neighbour) in enumerate(zip(*np.where(upper_neighbours > 0))):
        # x_k - x_i - w_i
        contact_arr[contact_num, cols*high_neighbour+axis   ] =  1
        contact_arr[contact_num, cols*low_neighbour +axis   ] = -1
        contact_arr[contact_num, cols*low_neighbour +axis +2] = -1
    contact_rhs = np.zeros(shape=(m,))
    return contact_arr, contact_rhs

def contacts_after_hole_closing(x:np.ndarray, hole_closing_idxs:list):
    """ Nonlinear constraint from closing hole between:
            [i_X, j_X] or [n_Y, m_Y]
        Arguments:
            hole_closing_idxs = [[i_X, j_X], [n_Y, m_Y]]
        Returns value of 
        (X_right - X_left - width_left ) * (Y_top - X_down - height_down )
        one of which should be zero in final rectangulation.
        """
    arr:np.ndarray = x.reshape(-1, 4)
    idx_left, idx_right = hole_closing_idxs[0]
    idx_down, idx___top = hole_closing_idxs[1]
    diff_X = arr[idx_right, 0] - (arr[idx_left, 0] + arr[idx_left, 2])
    diff_Y = arr[idx___top, 0] - (arr[idx_down, 1] + arr[idx_down, 3])
    return diff_X * diff_Y 

def hole_closing_jac(x:np.ndarray, hole_closing_idxs:list):
    arr:np.ndarray = x.reshape(-1, 4)
    jac_arr = np.zeros(shape=arr.shape)
    idx_left, idx_right = hole_closing_idxs[0]
    idx_down, idx___top = hole_closing_idxs[1]
    diff_X = arr[idx_right, 0] - (arr[idx_left, 0] + arr[idx_left, 2])
    diff_Y = arr[idx___top, 0] - (arr[idx_down, 1] + arr[idx_down, 3])
    jac_arr[idx_right, 0] = diff_Y
    jac_arr[idx_left, 0] = -diff_Y
    jac_arr[idx_left, 2] = -diff_Y
    jac_arr[idx___top, 1] = diff_X
    jac_arr[idx_down, 1] = -diff_X
    jac_arr[idx_down, 3] = -diff_X
    return jac_arr.flatten()

def area_constraint_fun(x:np.ndarray, columns = 4):
    """ Nonlinear constraints
        argument x is flattened array of shape <clinched_rectangles>"""
    arr:np.ndarray = x.reshape(-1, columns)
    width, height =  arr[:,2], arr[:,3]
    return 1 - width.dot(height)

def area_jac(x:np.ndarray, columns = 4):
    """ Calculates the jacobian of area_constraint_fun at x"""
    arr:np.ndarray = x.reshape(-1, columns)
    jac:np.ndarray = np.zeros(shape=arr.shape)
    jac[:,2:] = arr[:,[3,2]]
    return -jac.flatten()
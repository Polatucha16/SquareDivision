import numpy as np
from numpy import linalg as LA

def dist_fun(arg:np.ndarray, clinched_rectangles:np.ndarray=np.array([])):
    arr = arg.reshape(-1,4)
    dist = LA.norm(arr - clinched_rectangles)**2
    jac = (2*(arr - clinched_rectangles)).flatten()
    return dist, jac

# def ratio_demand_cost(arg:np.ndarray, clinched_rectangles:np.ndarray=np.array([])):
#     #### COST
#     arr = arg.reshape(-1,4).astype(float)
#     dist_2 = LA.norm(arr - clinched_rectangles)**2
#     origin__widths, recip__widths = clinched_rectangles[:, 2], np.reciprocal(arr[:,2])
#     origin_heights, recip_heights = clinched_rectangles[:, 3], np.reciprocal(arr[:,3])
#     ones = np.ones(shape=origin__widths.shape)
#     penalty__width = LA.norm((origin__widths * recip__widths) - ones)**2
#     penatly_height = LA.norm((origin_heights * recip_heights) - ones)**2
#     mul_penalty = (1 + penalty__width + penatly_height)
#     cost = dist_2 * mul_penalty
#     #### JACOBIAN (fg)' = f'g + fg'
#     ## dist_2
#     d_dist_2 = (2*(arr - clinched_rectangles))
#     df_g = d_dist_2 * mul_penalty
#     ## mul_penalty
#     d_mul_penalty = np.zeros(shape=arr.shape)
#     d_norm_at__widths = 2*((origin__widths * recip__widths) - ones)
#     d_norm_at_heights = 2*((origin_heights * recip_heights) - ones)
#     d_recip__widths = (-1) * origin__widths * (recip__widths**2)
#     d_recip_heights = (-1) * origin_heights * (recip_heights**2)
#     d_mul_penalty_at__widths = d_norm_at__widths * d_recip__widths
#     d_mul_penalty_at_heights = d_norm_at_heights * d_recip_heights
#     d_mul_penalty[:, 2] = d_mul_penalty_at__widths
#     d_mul_penalty[:, 3] = d_mul_penalty_at_heights
#     f_dg = dist_2 * d_mul_penalty
#     jac = (df_g + f_dg).flatten()
#     return cost, jac
import numpy as np
from scipy.optimize import Bounds

def bounds_trust_constr(clinched_rectangles:np.ndarray, keep_feasible=True):
    shape = clinched_rectangles.shape
    lb = np.zeros(shape=shape).flatten()
    ub = np.ones(shape=shape).flatten()
    return Bounds(lb=lb, ub=ub, keep_feasible=keep_feasible)
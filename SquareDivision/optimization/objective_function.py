import numpy as np
from numpy import linalg as LA

def dist_fun(arg:np.ndarray, clinched_rectangles:np.ndarray=np.array([])):
    arr = arg.reshape(-1,4)
    dist = LA.norm(arr - clinched_rectangles)
    return dist

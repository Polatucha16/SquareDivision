import numpy as np
from numpy import linalg as LA
import networkx as nx
from scipy.optimize import root

from SquareDivision.optimization.constraints import (
    low_boundary_constraint_args,
    high_boundary_constraint_args,
    contact_constraint_args,
    closing_holes_4_way,
    area_constraint_fun
)

def system(x:np.ndarray, idxs_to_close, east_graph:nx.Graph, north_graph:nx.Graph):


def search_x0(x, clinched_rectangles, radius):
    """ for c = sqrt(np.prod(shape)) search for solution with closed in L^inf sphere of radius <radius> around 
    clinched_rectangles when failed search in 2*<radius>, then 3*<radius> etc.
    """

    pass
import numpy as np
import networkx as nx
from networkx.algorithms.cycles import simple_cycles

def graph_from_neighbours_arrays(east_neighbours, north_neighbours):
    graph_matrix = east_neighbours + north_neighbours
    G = nx.from_numpy_array(graph_matrix)
    return G

def find_holes(clinched_rectangles, east_neighbours, north_neighbours):
    """ Return lists of indices of rectangles (rows in clinched_rectangles)
        that sorround hole in clinched_rectangles.
        """
    G = graph_from_neighbours_arrays(east_neighbours, north_neighbours)
    cycles_4 = [cycle for cycle in simple_cycles(G, 4) if len(cycle)==4]
    cycles_3 = [cycle for cycle in simple_cycles(G, 3) if len(cycle)==3]
    holes = []
    for cycle in cycles_4:
        num_of_3_cycles_in = 0
        cycle_set = set(cycle)
        for tri in cycles_3:
            tri_set = set(tri)
            if tri_set.intersection(cycle_set) == tri_set:
                num_of_3_cycles_in += 1
                break #this cycle is not a hole
        if num_of_3_cycles_in == 0:
            holes.append(cycle)
    return holes

def hole_closing_idxs(hole_path, clinched_rectangles):
    """ Return two pairs of indices [[i_X, j_X], [n_Y,m_Y]] representing 
        two of possible ways of closing a hole in hole_path in clinched_rectangles.
        Return
            [i_X, j_X]  reprezent horizontal squeeze of a hole i.e.
            in configuration without holes(after squeezing) rectangle i_X is touching rectangle j_X:
                arr[i_X, 0] + arr[i_X, 2] = arr[j_X, 0].

            [i_X, j_X]  reprezent vertical squeeze of a hole i.e.
            in configuration without holes(after squeezing) rectangle n_Y is touching rectangle m_Y:
                arr[n_Y, 1] + arr[n_Y, 3] = arr[m_Y, 1].
        """
    arg_of_top_Y = np.argmax(clinched_rectangles[hole_path][:,1])
    idx_of_top   = hole_path[arg_of_top_Y]
    idx_of_bottom= hole_path[(arg_of_top_Y + 2) % 4]
    arg_of_top_X = np.argmax(clinched_rectangles[hole_path][:,0])
    idx_of_right = hole_path[arg_of_top_X]
    idx_of__left = hole_path[(arg_of_top_X + 2) % 4]
    return [[idx_of__left, idx_of_right], [idx_of_bottom, idx_of_top ]]
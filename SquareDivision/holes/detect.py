import networkx as nx

def find_holes(clinched_rectangles, east_neighbours, north_neighbours):
    graph_matrix = east_neighbours + north_neighbours
    G = nx.from_numpy_array(graph_matrix)
    return G
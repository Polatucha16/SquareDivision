import numpy as np

from SquareDivision.src.regions import (
    opposing_walls_in_half_plane_in_dir,
    intersect_intervals,
    wall_axis,
    perp_axis
    )

# FIX: function takin a number arr and direction  and returning a wall: [a,b]
def wall_of(rect_num:int, clinched_rectangles:np.ndarray, dir):
    rect = clinched_rectangles[rect_num]
    perp_idx = perp_axis(dir)
    s, t = rect[perp_idx], rect[perp_idx] + rect[perp_idx + 2]
    return np.array([s, t])

def numers_of_contacting_rectangles_in_dir(rect_num:int, clinched_rectangles:np.ndarray, dir):
    n, cols = clinched_rectangles.shape
    wall = wall_of(rect_num, clinched_rectangles, dir)
    # potential upper neighbours: 
    idx_rect, suspended_walls = opposing_walls_in_half_plane_in_dir(rect_num, clinched_rectangles, dir)

    potential_barriers = np.arange(n)[idx_rect]
    if len(suspended_walls) == 0:
        # there are no upper neighbours
        return np.zeros(shape=(n,))
    have_intersection =  np.apply_along_axis(
        func1d = (lambda vec : intersect_intervals(*wall, *vec)[0]),
        axis = 1,
        arr = suspended_walls[:, 1:]
        )
    potential_contacts = potential_barriers[have_intersection]
    # for rectangles number in potential_contacts to be contanct 
    # we need rect[wall_axis] + rect[wall_axis + 2] == clinched_rectangles[potential_contacts][wall_axis]
    idx = wall_axis(dir)
    rect = clinched_rectangles[rect_num]
    contact_numbers = []
    for potencial_rectangle_num in potential_contacts:
        pot_rect = clinched_rectangles[potencial_rectangle_num]
        if np.isclose(rect[idx] + rect[idx + 2], pot_rect[idx] ):
            contact_numbers.append(potencial_rectangle_num)
    contact_row = np.zeros(shape=(n,))
    number_of_contacts = len(contact_numbers)
    contact_row[contact_numbers] = np.ones(shape=(number_of_contacts,))
    return contact_row

def contact_graph_incidence_matrix(clinched_rectangles, dir) -> np.ndarray:
    rows = []
    for i, rect in enumerate(clinched_rectangles):
        rows.append(numers_of_contacting_rectangles_in_dir(
            i, clinched_rectangles, dir
        ))
    return np.r_[rows]
    
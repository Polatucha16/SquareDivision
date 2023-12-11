import numpy as np
from typing import Literal

from SquareDivision.walls.suspended_walls import from_rectangles_family, intersect

# from SquareDivision.src.regions import (
#     opposing_walls_in_half_plane_in_dir,
#     intersect_intervals,
#     wall_axis,
#     perp_axis,
#     direction,
# )


# FIX: function takin a number arr and direction  and returning a wall: [a,b]
# def wall_of(rect_num: int, clinched_rectangles: np.ndarray, dir: direction):
#     rect = clinched_rectangles[rect_num]
#     perp_idx = perp_axis(dir)
#     s, t = rect[perp_idx], rect[perp_idx] + rect[perp_idx + 2]
#     return np.array([s, t])


# def numers_of_contacting_rectangles_in_dir(
#     rect_num: int, clinched_rectangles: np.ndarray, dir: direction
# ):
#     """Return row (of 0s and 1s) of incidence matrix of upper contact graphin in direction dir of clinched_rectangles"""
#     n, cols = clinched_rectangles.shape
#     wall = wall_of(rect_num, clinched_rectangles, dir)
#     # potential upper neighbours:
#     idx_rect, suspended_walls = opposing_walls_in_half_plane_in_dir(
#         rect_num, clinched_rectangles, dir
#     )

#     potential_barriers = np.arange(n)[idx_rect]
#     if len(suspended_walls) == 0:
#         # there are no upper neighbours
#         return np.zeros(shape=(n,))
#     have_intersection = np.apply_along_axis(
#         func1d=(lambda vec: intersect_intervals(*wall, *vec)[0]),
#         axis=1,
#         arr=suspended_walls[:, 1:],
#     )
#     potential_contacts = potential_barriers[have_intersection]
#     # for rectangles number in potential_contacts to be contanct
#     # we need rect[wall_axis] + rect[wall_axis + 2] == clinched_rectangles[potential_contacts][wall_axis]
#     idx = wall_axis(dir)
#     rect = clinched_rectangles[rect_num]
#     contact_numbers = []
#     for potencial_rectangle_num in potential_contacts:
#         pot_rect = clinched_rectangles[potencial_rectangle_num]
#         if np.isclose(rect[idx] + rect[idx + 2], pot_rect[idx]):
#             contact_numbers.append(potencial_rectangle_num)
#     contact_row = np.zeros(shape=(n,))
#     number_of_contacts = len(contact_numbers)
#     contact_row[contact_numbers] = np.ones(shape=(number_of_contacts,))
#     return contact_row


# def contact_graph_incidence_matrix(clinched_rectangles, dir: direction) -> np.ndarray:
#     rows = []
#     for i, rect in enumerate(clinched_rectangles):
#         rows.append(numers_of_contacting_rectangles_in_dir(i, clinched_rectangles, dir))
#     out: np.ndarray = np.r_[rows]
#     return out.astype(int)

from SquareDivision.walls.suspended_walls import from_rectangles_family, intersect

def idxs_of_touching_rectangles_in_dir(
    rect_num: int, clinched_rectangles: np.ndarray, direction: Literal['l', 'r', 'u', 'd']
):
    """
    Return row (of 0s and 1s) of rectangles contacating rect_num
    in direction dir in clinched_rectangles family

    for rectangle family rects.clinched_rectangles use:
        from SquareDivision.contact_graph.incidence_matrix import idxs_of_touching_rectangles_in_dir
        idxs_of_touching_rectangles_in_dir(6, rects.clinched_rectangles, 'd')
    """
    opp_dir = {'r' : 'l', 'u' : 'd', 'l' : 'r','d' : 'u'}
    curr_rect:np.ndarray = clinched_rectangles[rect_num]
    current_rect_wall = from_rectangles_family(a=curr_rect.reshape(1, -1), dir=direction)
    opposing_walls = from_rectangles_family(a=clinched_rectangles, dir=opp_dir[direction])
    # [FIX] SuspendedWalls should have had methods of intersecting like below:
    # same_anchor_idxs is array of True False:
    same_anchor_Q = np.isclose(opposing_walls[:, 0], current_rect_wall[:, 0])
    same_anchor_idxs = np.nonzero(same_anchor_Q)[0]
    if same_anchor_Q.astype(int).sum() == 0:
        # there are no neighbours in direction
        return np.array([]).astype(int)
    possible_opp_wals = opposing_walls[same_anchor_Q]
    intersect_with_curr_Q = np.apply_along_axis( 
        func1d = (lambda other_wall: intersect(current_rect_wall[0, 1:], other_wall)[0]),
        axis=1,
        arr=possible_opp_wals[:, 1:]
    )
    return same_anchor_idxs[intersect_with_curr_Q]

def idxs_to_incidence_row(idxs:np.ndarray, clinched_rectangles:np.ndarray):
    """
    Return array of shape (n,) such that
    they are 1 at indices in idxs and zero elsewhere"""
    shape = clinched_rectangles.shape
    incidence_row = np.zeros(shape=(shape[0],))
    incidence_row[idxs] = 1
    return incidence_row

def contact_graph_incidence_matrix(clinched_rectangles, dir: Literal['l', 'r', 'u', 'd']) -> np.ndarray:
    rows = []
    for i, rect in enumerate(clinched_rectangles):
        idxs = idxs_of_touching_rectangles_in_dir(i, clinched_rectangles, dir)
        rows.append(idxs_to_incidence_row(idxs, clinched_rectangles))
    out: np.ndarray = np.r_[rows]
    return out.astype(int)
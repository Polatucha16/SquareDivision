import numpy as np
from typing import Literal

def intersect(a: np.ndarray, b: np.ndarray) -> tuple:
    """
    Intersect open intervals a, b. Ignoring order of points in a and b.
    Return tuple:
       0 :  True or False if intervals intersect or not respectively;
       1 :  intersection : (start of intersection, end of intersection)
                or
            gap backwards : (end of gap, start of gap).
    """
    a_min, a_max = min(a), max(a)
    b_min, b_max = min(b), max(b)
    min_max = min(a_max, b_max)
    max_min = max(a_min, b_min)
    dx = min_max - max_min
    return dx > 0, np.array([max_min, min_max])

def interval_scaling(dist, interval, focal_length) -> np.ndarray:
    """
    Return image of interval (a, b) under homogeneus scaling with focal_length from the center at the distance.
    (ivl = interval)
                                            + stop
                            + max(ivl)      |
        +<--focal_length--->|<--- dist  --->|
                            + min(ivl)      |
                                            + start

    """
    interval_min, interval_max = min(interval), max(interval)
    half_interval_length = (interval_max - interval_min) / 2
    slope = half_interval_length / focal_length
    start = -slope * dist + interval_min
    stop = slope * dist + interval_max
    return np.array([start, stop])

def from_rectangles_family(a, dir:Literal['l','r','d','u']):
    """
    build suspended walls from family of rectangles stored in 
    array a of shape (N, 4)"""
    direction_dict = {'l' : 'vertical'  , 'r' : 'vertical',
                      'd' : 'horizontal', 'u' : 'horizontal'  }
    direction = direction_dict[dir]
    if dir == 'l':
        anchors = a[:, 0]
        wall_0, wall_1 = a[:, 1], a[:, 1] + a[:, 3]
    elif dir == 'r':
        anchors = a[:, 0] + a[:, 2]
        wall_0, wall_1 = a[:, 1], a[:, 1] + a[:, 3]
    elif dir == 'd':
        anchors = a[:, 1]
        wall_0, wall_1 = a[:, 0], a[:, 0] + a[:, 2]
    elif dir == 'u':
        anchors = a[:, 1] + a[:,3]
        wall_0, wall_1 = a[:, 0], a[:, 0] + a[:, 2]
    else:
        raise Exception(f'dir = {dir} is not one of ["l", "r", d", "u"]')
    data = np.c_[anchors, wall_0, wall_1]
    return SuspendedWalls(data, direction)



class SuspendedWalls:
    """ 
    Store and query families of axial supsended walls in [0,1]^2 
    suspended wall in this class np.ndarray : [anchor, start, stop]"""
    _anchor_axis_dir = {'horizontal' : 1, 'vertical' : 0}
    _wall___axis_dir = {'horizontal' : 0, 'vertical' : 1}

    def __init__(
        self, data: np.ndarray, direction: Literal["horizontal", "vertical"]
    ) -> None:
        self.data = data
        self.direction = direction
        self.anchor_ax = SuspendedWalls._anchor_axis_dir[direction]
        self.wall___ax = SuspendedWalls._wall___axis_dir[direction]

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, newvalue):
        self.data[key] = newvalue

    def anchors(self) -> np.ndarray:
        return self.data[:, 0]

    def walls(self) -> np.ndarray:
        return self.data[:, 1:]

    def walls_in_half_plane(
        self, suspended_wall: np.ndarray, leq_or_geq: Literal["leq", "geq"]
    ) -> np.ndarray:
        """
        For a triple [anchor*, start, stop] in suspended_wall argument,
        Return suspended walls that have anchors
        'less or equal than' or 'greater or equal than' (depending on leq_or_geq)
        anchor*."""

        if leq_or_geq == "leq":
            indices = np.asarray(self.anchors() <= suspended_wall[0]).nonzero()
        elif leq_or_geq == "geq":
            indices = np.asarray(self.anchors() >= suspended_wall[0]).nonzero()
        else:
            raise Exception(f'leq_or_geq = {leq_or_geq} is not one of ["leq", "geq"]')
        data = self.data[indices]
        return data, indices[0]

    def potential_bariers(self, suspended_wall, leq_or_geq) -> np.ndarray:
        """
        Return the indices of those walls from self.data that can be potential stoping
        places for suspended_wall in direction leq_or_geq
        ordered in anchors starting from the closesed to suspended_wall"""
        bariers, indices = self.walls_in_half_plane(suspended_wall, leq_or_geq)
        anchors_order = np.argsort(bariers[:, 0])
        anchors_order = anchors_order if leq_or_geq == "geq" else np.flip(anchors_order)
        return indices[anchors_order]

    def first_barrier_in_wall_push(self, suspended_wall, leq_or_geq) -> np.ndarray:
        """
        Return suspended wall that is the intersection of prallel projection of suspended_wall in the direction leq_or_geq
        onto first suspended wall from the family self.data
        """
        rectangle_wall = suspended_wall[1:]
        anchors_order = self.potential_bariers(suspended_wall, leq_or_geq)
        for i, wall_number in enumerate(anchors_order):
            current_wall = self.walls()[wall_number]
            intersectQ, start_stop = intersect(current_wall, rectangle_wall)
            if intersectQ == True:
                data = np.r_[self.anchors()[wall_number], start_stop]
                return data
            else:
                continue
        # HERE no intersection wos found -> Go to 0 or 1 depending on leq_or_geq
        if leq_or_geq == "leq":
            data = np.r_[0, rectangle_wall]
        else:
            data = np.r_[1, rectangle_wall]
        return data

    def first_barrier_in_wall_scale(
        self, suspended_wall, leq_or_geq, focal_length
    ) -> np.ndarray:
        """
        Return suspended wall that is intersection of 
        1. projection of suspended_wall in direction leq_or_geq from the point focal_length away of suspended_wall
        and 
        2. first (in direction leq_or_geq) suspended wall from family self.data
        """
        rectangle_wall = suspended_wall[1:]
        anchors_order = self.potential_bariers(suspended_wall, leq_or_geq)
        for i, wall_number in enumerate(anchors_order):
            current_wall = self.walls()[wall_number]
            # scaling here
            dist = abs(suspended_wall[0] - self.anchors()[wall_number])
            scaled_start_stop = interval_scaling(dist, rectangle_wall, focal_length)
            intersectQ, start_stop = intersect(current_wall, scaled_start_stop)
            if intersectQ == True:
                data = np.r_[self.anchors()[wall_number], start_stop]
                return data
            else:
                continue
        # HERE no intersection wos found -> Go to 0 or 1 depending on leq_or_geq
        if leq_or_geq == "leq":
            dist = abs(suspended_wall[0] - 0)
            scaled_start_stop = interval_scaling(dist, rectangle_wall, focal_length)
            data = np.r_[0, scaled_start_stop]
        else:
            dist = abs(suspended_wall[0] - 1)
            scaled_start_stop = interval_scaling(dist, rectangle_wall, focal_length)
            data = np.r_[1, scaled_start_stop]
        return data

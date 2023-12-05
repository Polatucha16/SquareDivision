import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List, Literal
from SquareDivision.walls.suspended_walls import SuspendedWalls, from_rectangles_family

#### for disjoint family of rentancgle stored in array a, return clinched rectangles


def rectangle_walls(rectangle: np.ndarray) -> List[SuspendedWalls]:
    right_wall = from_rectangles_family(rectangle.reshape(1, -1), "r")
    upper_wall = from_rectangles_family(rectangle.reshape(1, -1), "u")
    left__wall = from_rectangles_family(rectangle.reshape(1, -1), "l")
    down__wall = from_rectangles_family(rectangle.reshape(1, -1), "d")
    return right_wall, upper_wall, left__wall, down__wall


# def find_max_scaling(
#     idx: int,
#     disjoint_rectangles: np.ndarray,
#     right_walls: SuspendedWalls,
#     upper_walls: SuspendedWalls,
#     left__walls: SuspendedWalls,
#     down__walls: SuspendedWalls,
# ):
#     """return maximal obstacle free scaling of an rectangle"""
#     x, y, width, height = disjoint_rectangles[idx]
#     center = np.array([x + width / 2, y + height / 2])
#     right_barrier = left__walls.first_barrier_in_wall_scale(
#         right_walls.data[idx], "geq", width / 2
#     )
#     upper_barrier = down__walls.first_barrier_in_wall_scale(
#         upper_walls.data[idx], "geq", height / 2
#     )
#     left__barrier = right_walls.first_barrier_in_wall_scale(
#         left__walls.data[idx], "leq", width / 2
#     )
#     down__barrier = upper_walls.first_barrier_in_wall_scale(
#         down__walls.data[idx], "leq", height / 2
#     )
#     right_scale = abs(right_barrier[0] - center[0]) / (width / 2)
#     upper_scale = abs(upper_barrier[0] - center[1]) / (height / 2)
#     left__scale = abs(left__barrier[0] - center[0]) / (width / 2)
#     down__scale = abs(down__barrier[0] - center[1]) / (height / 2)
#     scales = np.array([right_scale, upper_scale, left__scale, down__scale])
#     return min(scales)

def push_rectangle(rectangle: np.ndarray, barrier:np.ndarray, direction:Literal['r','u','l','d']):
    x, y, width, height = rectangle
    if direction == 'r':
        new_width = barrier[0] - rectangle[0]
        rectangle[2] = new_width
        return rectangle
    elif direction == 'u':
        new_height = barrier[0] - rectangle[1]
        rectangle[3] = new_height
        return rectangle
    elif direction == 'l':
        new_x = barrier[0]
        new_width = width + x - new_x
        rectangle[0], rectangle[2] = new_x, new_width
        return rectangle
    elif direction == 'd':
        new_y = barrier[0]
        new_height = height + y -new_y
        rectangle[1], rectangle[3] = new_y, new_height
        return rectangle
    
## FIX : move to class and make sure scaled agree with stopping walls
# this method gives : 
# right wall : [0.59418404 0.5226999  0.6490047 ]
# left  wall : [0.59749117 0.4939744  0.63159795]
def scale_rectangle(rectangle: np.ndarray, scale):# barrier:np.ndarray, direction:Literal['r','u','l','d']):
    """scale from the center""" 
    x, y, width, height = rectangle
    xy = np.array([x, y])
    center = np.array([x + width / 2, y + height / 2])
    new_xy = center + scale * (xy - center)
    new_width, new_height = scale * width, scale * height
    new_rectangle: np.ndarray = np.r_[new_xy, new_width, new_height]
    return new_rectangle


def rectangles_idx_update(
    idx: int,
    disjoint_rectangles: np.ndarray,
    right_walls: SuspendedWalls,
    upper_walls: SuspendedWalls,
    left__walls: SuspendedWalls,
    down__walls: SuspendedWalls,
):
    """Order : right, up, left, down"""
    scale = find_max_scaling(
        idx,
        disjoint_rectangles,
        right_walls,
        upper_walls,
        left__walls,
        down__walls,
    )
    # scaling
    new_rectangle = scale_rectangle(disjoint_rectangles[idx], scale)
    disjoint_rectangles[idx] = new_rectangle
    (right_wall, upper_wall, left__wall, down__wall) = rectangle_walls(new_rectangle)
    right_walls[idx] = right_wall[0]
    upper_walls[idx] = upper_wall[0]
    left__walls[idx] = left__wall[0]
    down__walls[idx] = down__wall[0]
    # pushing
    # right_barrier = left__walls.first_barrier_in_wall_push(right_wall.data[0], "geq")
    # new_width = abs(new_rectangle[0] - right_barrier[0])
    # new_rectangle[2] = new_width
    # (right_wall, upper_wall, left__wall, down__wall) = rectangle_walls(new_rectangle)

    # upper_barrier = down__walls.first_barrier_in_wall_push(upper_wall.data[0], "geq")
    # new_height = abs(new_rectangle[1] - upper_barrier[0])
    # new_rectangle[3] = new_height
    # (right_wall, upper_wall, left__wall, down__wall) = rectangle_walls(new_rectangle)

    # left__barrier = right_walls.first_barrier_in_wall_push(left__wall.data[0], "leq")
    # new_x = left__barrier[0]
    # new_width = new_rectangle[0] + new_rectangle[2] - new_x
    # new_rectangle[0], new_rectangle[2] = new_x, new_width
    # (right_wall, upper_wall, left__wall, down__wall) = rectangle_walls(new_rectangle)

    # down__barrier = upper_walls.first_barrier_in_wall_push(down__wall.data[0], "leq")
    # new_y = down__barrier[0]
    # new_height = new_rectangle[1] + new_rectangle[3] - new_y
    # new_rectangle[1], new_rectangle[3] = new_y, new_height
    # (right_wall, upper_wall, left__wall, down__wall) = rectangle_walls(new_rectangle)

    # update return
    # disjoint_rectangles[idx] = new_rectangle
    # right_walls.data[idx] = right_wall.data[0]
    # upper_walls.data[idx] = upper_wall.data[0]
    # left__walls.data[idx] = left__wall.data[0]
    # down__walls.data[idx] = down__wall.data[0]
    return (
        disjoint_rectangles,
        right_walls,
        upper_walls,
        left__walls,
        down__walls,
    )


# rectangles centers strategies
class InflateStrategy(ABC):
    @abstractmethod
    def inflate(self, rectangles_sample, **kwargs) -> np.ndarray:
        # this method changes disjoint rectangles to clinched rectangles
        pass


class MaxHomThenMaxPushFromOrder(InflateStrategy):
    def __init__(self, disjoint_rectangles, order=None) -> None:
        self.rectangles = disjoint_rectangles
        if order is None:
            widths, heights = disjoint_rectangles[:, 2], disjoint_rectangles[:, 3]
            self.order = np.flip(np.argsort(widths * heights))

    def update_walls(self):
        self.right_walls = from_rectangles_family(self.rectangles, dir="r")
        self.upper_walls = from_rectangles_family(self.rectangles, dir="u")
        self.left__walls = from_rectangles_family(self.rectangles, dir="l")
        self.down__walls = from_rectangles_family(self.rectangles, dir="d")

    def find_max_scaling(self, idx):
        """return maximal, obstacle free, homogeneous scaling of an rectangle"""
        x, y, width, height = self.rectangles[idx]
        center = np.array([x + width / 2, y + height / 2])
        right_barrier = self.left__walls.first_barrier_in_wall_scale(
            self.right_walls[idx], "geq", width / 2
        )
        upper_barrier = self.down__walls.first_barrier_in_wall_scale(
            self.upper_walls[idx], "geq", height / 2
        )
        left__barrier = self.right_walls.first_barrier_in_wall_scale(
            self.left__walls[idx], "leq", width / 2
        )
        down__barrier = self.upper_walls.first_barrier_in_wall_scale(
            self.down__walls[idx], "leq", height / 2
        )
        right_scale = abs(right_barrier[0] - center[0]) / (width / 2)
        upper_scale = abs(upper_barrier[0] - center[1]) / (height / 2)
        left__scale = abs(left__barrier[0] - center[0]) / (width / 2)
        down__scale = abs(down__barrier[0] - center[1]) / (height / 2)
        scales = np.array([right_scale, upper_scale, left__scale, down__scale])
        return min(scales)
    
    # def inflate_idx(self, idx):
    #     scale = self.find_max_scaling(idx)
    #     if np.isclose(scale, 1) == False:
    #         self.rectangles[idx] = scale_rectangle(self.rectangles[idx], scale)
    #         self.update_walls()
    #     else: 
    #         return

    def inflate(self) -> np.ndarray:
        """Return clinched rectangles"""
        self.update_walls()

        for idx in self.order:
            scale = self.find_max_scaling(idx)
            if np.isclose(scale, 1) == False:
                self.rectangles[idx] = scale_rectangle(self.rectangles[idx], scale)
                self.update_walls()
        
        return self.rectangles
        

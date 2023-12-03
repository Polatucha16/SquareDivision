import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List
from SquareDivision.walls.suspended_walls import SuspendedWalls, from_rectangles_family

#### for disjoint family of rentancgle stored in array a, return clinched rectangles


def rectangle_walls(rectangle: np.ndarray) -> List[SuspendedWalls]:
    right_wall = from_rectangles_family(rectangle.reshape(1, -1), "r")
    upper_wall = from_rectangles_family(rectangle.reshape(1, -1), "u")
    left__wall = from_rectangles_family(rectangle.reshape(1, -1), "l")
    down__wall = from_rectangles_family(rectangle.reshape(1, -1), "d")
    return right_wall, upper_wall, left__wall, down__wall


def find_max_scaling(
    idx: int,
    disjoint_rectangles: np.ndarray,
    left__walls: SuspendedWalls,
    right_walls: SuspendedWalls,
    down__walls: SuspendedWalls,
    upper_walls: SuspendedWalls,
):
    """return maximal obstacle free scaling of an rectangle"""
    x, y, width, height = disjoint_rectangles[idx]
    center = np.array([x + width / 2, y + height / 2])
    right_barrier = left__walls.first_barrier_in_wall_scale(
        right_walls.data[idx], "geq", width / 2
    )
    upper_barrier = down__walls.first_barrier_in_wall_scale(
        upper_walls.data[idx], "geq", height / 2
    )
    left__barrier = right_walls.first_barrier_in_wall_scale(
        left__walls.data[idx], "leq", width / 2
    )
    down__barrier = upper_walls.first_barrier_in_wall_scale(
        down__walls.data[idx], "leq", height / 2
    )
    right_scale = abs(right_barrier[0] - center[0]) / (width / 2)
    upper_scale = abs(upper_barrier[0] - center[1]) / (height / 2)
    left__scale = abs(left__barrier[0] - center[0]) / (width / 2)
    down__scale = abs(down__barrier[0] - center[1]) / (height / 2)
    scales = np.array([right_scale, upper_scale, left__scale, down__scale])
    scale = min(scales)
    return scale


def scale_rectangle(rectangle: np.ndarray, scale: float):
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
    (right_wall, upper_wall, left__wall, down__wall) = rectangle_walls(new_rectangle)
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
    disjoint_rectangles[idx] = new_rectangle
    right_walls.data[idx] = right_wall.data[0]
    upper_walls.data[idx] = upper_wall.data[0]
    left__walls.data[idx] = left__wall.data[0]
    down__walls.data[idx] = down__wall.data[0]
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
    def __init__(self, order=None) -> None:
        self.order: np.ndarray = order

    def inflate(self, disjoint_rectangles) -> np.ndarray:
        # disjoint_rectangles = np.copy(disjoint_rectangles)
        if self.order is None:
            widths, heights = disjoint_rectangles[:, 2], disjoint_rectangles[:, 3]
            self.order = np.flip(np.argsort(widths * heights))
        right_walls = from_rectangles_family(disjoint_rectangles, dir="r")
        upper_walls = from_rectangles_family(disjoint_rectangles, dir="u")
        left__walls = from_rectangles_family(disjoint_rectangles, dir="l")
        down__walls = from_rectangles_family(disjoint_rectangles, dir="d")
        for idx in self.order:
            (
                disjoint_rectangles,
                right_walls,
                upper_walls,
                left__walls,
                down__walls,
            ) = rectangles_idx_update(
                idx,
                disjoint_rectangles,
                right_walls,
                upper_walls,
                left__walls,
                down__walls,
            )
        return disjoint_rectangles
        

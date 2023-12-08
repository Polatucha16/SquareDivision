import numpy as np
from abc import ABC, abstractmethod
from typing import Literal
from SquareDivision.walls.suspended_walls import SuspendedWalls, from_rectangles_family

class InflateStrategy(ABC):
    @abstractmethod
    def inflate(self, disjoint_rectangles, **kwargs) -> np.ndarray:
        # this method changes disjoint rectangles to clinched rectangles
        pass


class MaxHomThenMaxPushFromOrder(InflateStrategy):

    def update_walls(self, idx=None):
        if idx == None:
            self.right_walls = from_rectangles_family(self.rectangles, dir="r")
            self.upper_walls = from_rectangles_family(self.rectangles, dir="u")
            self.left__walls = from_rectangles_family(self.rectangles, dir="l")
            self.down__walls = from_rectangles_family(self.rectangles, dir="d")
        else:
            rectangle = self.rectangles[idx].reshape(1, -1)
            self.right_walls[idx] = from_rectangles_family(rectangle, dir="r").data[0]
            self.upper_walls[idx] = from_rectangles_family(rectangle, dir="u").data[0]
            self.left__walls[idx] = from_rectangles_family(rectangle, dir="l").data[0]
            self.down__walls[idx] = from_rectangles_family(rectangle, dir="d").data[0]

    def find_max_scaling(self, idx):
        """Return maximal, obstacle free, homogeneous scaling of an rectangle"""
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
        barriers = {
            "r": right_barrier,
            "u": upper_barrier,
            "l": left__barrier,
            "d": down__barrier,
        }
        right_scale = abs(right_barrier[0] - center[0]) / (width / 2)
        upper_scale = abs(upper_barrier[0] - center[1]) / (height / 2)
        left__scale = abs(left__barrier[0] - center[0]) / (width / 2)
        down__scale = abs(down__barrier[0] - center[1]) / (height / 2)
        scales = np.array([right_scale, upper_scale, left__scale, down__scale])
        scale = scales[np.argmin(scales)]
        direction = ["r", "u", "l", "d"][np.argmin(scales)]
        return scale, direction, barriers

    def scale_rectangle(
        self,
        idx: int,
        scale: float,
        direction: Literal["r", "u", "l", "d"],
        barriers: dict,
    ) -> np.ndarray:
        """
        Return homogeneously scaled rectangle.
        Arguments:
            idx         : index of rectangle from self.rectangles to scale
            scale       : how much to scale
            direction   : direction where is the first barrier
            barriers    : dictionary of bariers {direction : suspended wall}
                            it contain: {'r' : np.array([anchor, start, stop]), ...}
        """
        if np.isclose(scale, 1.0) is True:
            return self.rectangles[idx]
        x, y, width, height = self.rectangles[idx]
        center = np.array([x + width / 2, y + height / 2])
        if direction == "r":
            new_width = scale * width
            new_x = barriers["r"][0] - new_width
            new_height = scale * height
            new_y = center[1] - new_height / 2
            return np.r_[new_x, new_y, new_width, new_height]
        elif direction == "u":
            new_height = scale * height
            new_y = barriers["u"][0] - new_height
            new_width = scale * width
            new_x = center[0] - new_width / 2
            return np.r_[new_x, new_y, new_width, new_height]
        elif direction == "l":
            new_x = barriers["l"][0]
            new_width = scale * width
            new_height = scale * height
            new_y = center[1] - new_height / 2
            return np.r_[new_x, new_y, new_width, new_height]
        elif direction == "d":
            new_y = barriers["d"][0]
            new_width = scale * width
            new_height = scale * height
            new_x = center[0] - new_width / 2
            return np.r_[new_x, new_y, new_width, new_height]
        else:
            raise Exception(f"direction = {direction} not in ['r', 'u', 'l', 'd']")

    def find_max_push(
        self, idx: int, direction: Literal["r", "u", "l", "d"]
    ) -> np.ndarray:
        """
        Return first barier in rectangles on the push way"""
        if direction == "r":
            barrier = self.left__walls.first_barrier_in_wall_push(
                self.right_walls[idx], "geq"
            )
            return barrier
        elif direction == "u":
            barrier = self.down__walls.first_barrier_in_wall_push(
                self.upper_walls[idx], "geq"
            )
            return barrier
        elif direction == "l":
            barrier = self.right_walls.first_barrier_in_wall_push(
                self.left__walls[idx], "leq"
            )
            return barrier
        elif direction == "d":
            barrier = self.upper_walls.first_barrier_in_wall_push(
                self.down__walls[idx], "leq"
            )
            return barrier
        else:
            raise Exception(f"direction = {direction} not in ['r', 'u', 'l', 'd']")

    def push_rectangle(
        self, idx, barrier, direction: Literal["r", "u", "l", "d"]
    ) -> np.ndarray:
        x, y, width, height = self.rectangles[idx]
        if direction == "r":
            new_width = barrier[0] - x
            return np.r_[x, y, new_width, height]
        elif direction == "u":
            new_height = barrier[0] - y
            return np.r_[x, y, width, new_height]
        elif direction == "l":
            right_boundary = x + width
            new_x = barrier[0]
            new_width = right_boundary - new_x
            return np.r_[new_x, y, new_width, height]
        elif direction == "d":
            upper_boundary = y + height
            new_y = barrier[0]
            new_height = upper_boundary - new_y
            return np.r_[x, new_y, width, new_height]
        else:
            raise Exception(f"direction = {direction} not in ['r', 'u', 'l', 'd']")

    def inflate(self, disjoint_rectangles, order=None) -> np.ndarray:
        """Return clinched rectangles"""
        self.rectangles: np.ndarray = disjoint_rectangles
        if order is None:
            widths, heights = self.rectangles[:, 2], self.rectangles[:, 3]
            self.order = np.flip(np.argsort(widths * heights))
        self.update_walls()
        for idx in self.order:
            scale, direction, barriers = self.find_max_scaling(idx)
            self.rectangles[idx] = self.scale_rectangle(idx, scale, direction, barriers)
            self.update_walls(idx)

            for direction in ["r", "u", "l", "d"]:
                barrier = self.find_max_push(idx, direction)
                self.rectangles[idx] = self.push_rectangle(idx, barrier, direction)
                self.update_walls(idx)
        return self.rectangles

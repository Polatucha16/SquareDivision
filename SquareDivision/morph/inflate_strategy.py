import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
from SquareDivision.walls.suspended_walls import SuspendedWalls, from_rectangles_family
from SquareDivision.morph.inflate_rectangle import MaxHomThenMaxPush_idx_update

#### for disjoint family of rentancgle stored in array a, return clinched rectangles


# rectangles centers strategies
class InflateStrategy(ABC):
    @abstractmethod
    def inflate(self, rectangles_sample, **kwargs) -> np.ndarray:
        # change from rectangles sample to clinched_rectangles
        pass


class MaxHomThenMaxPushFromOrder(InflateStrategy):
    def __init__(self, order) -> None:
        self.order: np.ndarray = order

    def inflate(self, rectangles_sample) -> np.ndarray:
        left__walls = from_rectangles_family(rectangles_sample, dir="l")
        right_walls = from_rectangles_family(rectangles_sample, dir="r")
        down__walls = from_rectangles_family(rectangles_sample, dir="d")
        upper_walls = from_rectangles_family(rectangles_sample, dir="u")
        for idx in self.order:
            (
                rectangles_sample,
                left__walls,
                right_walls,
                down__walls,
                upper_walls,
            ) = MaxHomThenMaxPush_idx_update(
                idx,
                rectangles_sample,
                left__walls,
                right_walls,
                down__walls,
                upper_walls,
            )

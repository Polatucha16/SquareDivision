import numpy as np
from numpy.random._generator import Generator
from scipy.optimize import LinearConstraint
import networkx as nx

import matplotlib.pyplot as plt

from SquareDivision.src.distributions import SizeStrategy
from SquareDivision.src.dataflow import (
    find_anchors_and_crop,
    sort_by_area,
    remove_smaller,
    inflate_rectangles,
)
from SquareDivision.contact_graph.incidence_matrix import contact_graph_incidence_matrix
from SquareDivision.holes.detect import find_holes, holes_idxs, check_holes
from SquareDivision.projection.orthogonal import orth_proj_onto_affine_L
from SquareDivision.optimization.constraints import linear_constraint
from SquareDivision.draw.draw import draw_rectangles, rectangle_numbers
from SquareDivision.config import config


class Rectangulation:
    def __init__(self, config=config):
        self.rng: Generator = np.random.default_rng(config["seed"])

    def uniform_centers(self, num):
        self.centers = self.rng.uniform([0, 0], [1, 1], (num, 2))

    def sample_centers_from(self, distribution, **kwargs):
        self.centers = self.rng.__getattribute__(distribution)(**kwargs)

    def edit_centers(self, center_edit_strategy):
        # self.centers = center_edit_strategy.edit(self.centers)
        pass

    def sample_widths(self, startegy: SizeStrategy, **kwargs):
        """Execute strategy to generate widths sample"""
        self.widths = startegy.generate(**kwargs)

    def sample_heights(self, startegy: SizeStrategy, **kwargs):
        """Execute strategy to generate heights sample"""
        self.heights = startegy.generate(**kwargs)

    def sample_rectangles(
        self,
        num,
        widths_strategy: SizeStrategy,
        heights_strategy: SizeStrategy,
    ):
        """Default path of creating rectangles at uniformly drawn centers"""
        self.uniform_centers(num)
        self.sample_widths(
            widths_strategy,
            centers=self.centers,
        )
        self.sample_heights(
            heights_strategy,
            centers=self.centers,
        )
        self.rectangles_sample = np.c_[self.centers, self.widths, self.heights]

    def sample_rectangles_from(
        self,
        centers_distribution,
        centers_kwargs,
        # center_edit_strategy,
        widths_strategy,
        widths_kwargs,
        heights_strategy,
        heights_kwargs,
    ):
        """Advanced path of creating rectangles"""
        self.sample_centers_from(centers_distribution, **centers_kwargs)
        # self.edit_centers(center_edit_strategy)
        self.sample_widths(widths_strategy, **widths_kwargs)
        self.sample_heights(heights_strategy, **heights_kwargs)
        pass

    # FIX pass algoritm how to create disjoint rectangles from a rectangles_sample
    def find_disjoint_family(self):
        self.arr = find_anchors_and_crop(self.rectangles_sample)
        self.arr = sort_by_area(self.arr)
        self.arr = remove_smaller(self.arr)

    def inflate(self):
        self.clinched_rectangles = inflate_rectangles(self.arr)
        self.clinched_rectangles = np.maximum(0, self.clinched_rectangles)

    def graph_processing(self, rectangles=None):
        rectangles = self.clinched_rectangles if rectangles is None else rectangles
        self.east_neighbours = contact_graph_incidence_matrix(rectangles, "r").astype(
            int
        )
        self.north_neighbours = contact_graph_incidence_matrix(rectangles, "u").astype(
            int
        )
        self.east_graph = nx.from_numpy_array(self.east_neighbours)
        self.north_graph = nx.from_numpy_array(self.north_neighbours)
        self.holes = find_holes(self.east_neighbours, self.north_neighbours)
        self.possible_holes_idxs = holes_idxs(rectangles, self.holes)
        self.holes_idxs = check_holes(rectangles, self.possible_holes_idxs)

    def execute(self, num, **kwargs):
        self.sample_rectangles(num, **kwargs)
        self.find_disjoint_family()
        self.inflate()
        self.graph_processing()

    def prepare_closing(self, keep_feasible=True):
        self.x0 = self.clinched_rectangles.flatten()
        self.constraint: LinearConstraint = linear_constraint(
            self.clinched_rectangles,
            self.east_neighbours,
            self.north_neighbours,
            self.holes_idxs,
            keep_feasible=keep_feasible,
        )

    def close_holes(self):
        self.sol = orth_proj_onto_affine_L(
            self.x0, self.constraint.A, self.constraint.lb
        )
        self.closed = self.sol.reshape(-1, 4)

    def report(self, tol=0.03, digits=3):
        """Report rectangles relative ratio change between clinched_rectangles and closed"""

        def relative_change(before: np.ndarray, after: np.ndarray):
            """Return value of function on arrays given by formula:
            relative_shape_change( before = {x, y}, after = {u, v}) =
                (1 + x/u + y/v ) * Norm[{x, y} - {u, v}] ** 2
            """
            shape = before.shape
            x_ratios, y_ratios = before[:, 0] / after[:, 0], before[:, 1] / after[:, 1]
            ones = np.ones(shape=shape[0])
            coef = np.linalg.norm(before - after, axis=1) ** 2
            return (ones + x_ratios + y_ratios) * coef

        relative_diff = relative_change(
            self.clinched_rectangles[:, 2:], self.closed[:, 2:]
        )
        order = np.argsort(relative_diff)[::-1]
        rel_diff_ord = relative_diff[order]
        touched_beyoned_tolerance = np.where(rel_diff_ord > tol)[0]
        if len(touched_beyoned_tolerance) > 0:
            for no in touched_beyoned_tolerance:
                print(
                    f"rectangle no.{order[no]:>3} relatively changed by {rel_diff_ord[no]:>{digits+3}.{digits}f} "
                )
        else:
            print(f"All rectangles within tolerace")

    def draw(
        self,
        disjoint: bool = False,
        disjoint_nums: bool = True,
        inflated: bool = False,
        inflated_nums: bool = True,
        closed: bool = False,
        closed_nums: bool = True,
        size: int = 5,
    ):
        num_of_axes = np.array([disjoint, inflated, closed]).sum()
        fig, axes = plt.subplots(
            nrows=1, ncols=num_of_axes, figsize=(num_of_axes * size, size)
        )
        axes = [axes] if num_of_axes == 1 else axes
        i = 0
        if disjoint is True:
            axes[i] = draw_rectangles(axes[i], self.arr)
            if disjoint_nums is True:
                axes[i] = rectangle_numbers(axes[i], self.arr)
            i += 1
        if inflated is True:
            axes[i] = draw_rectangles(axes[i], self.clinched_rectangles)
            if inflated_nums is True:
                axes[i] = rectangle_numbers(axes[i], self.clinched_rectangles)
            i += 1
        if closed is True:
            axes[i] = draw_rectangles(axes[i], self.closed)
            if closed_nums is True:
                axes[i] = rectangle_numbers(axes[i], self.closed)
            i += 1
        plt.show()

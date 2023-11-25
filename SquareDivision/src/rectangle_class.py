import functools

import numpy as np
from numpy.random._generator import Generator
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint
import networkx as nx

import matplotlib.pyplot as plt

from SquareDivision.src.distributions import (
    x_plus_y_func, 
    tepui, 
    CentersGenerationStrategy,
    RngDistribution,
    WidthHeightStrategy,)
from SquareDivision.src.generators import uniform_pts
from SquareDivision.src.dataflow import (
    find_anchors_and_crop,
    sort_by_area,
    remove_smaller,
    inflate_rectangles,
    arg_rect_list,
    rects_from_distributions)
from SquareDivision.contact_graph.incidence_matrix import (
    contact_graph_incidence_matrix
)
from SquareDivision.holes.detect import find_holes, holes_idxs, check_holes
from SquareDivision.projection.orthogonal import orth_proj_onto_affine_L
from SquareDivision.optimization.constraints import linear_constraint #linear_constraints, hole_closing_constraints, linear_constraint
from SquareDivision.optimization.objective_function import objective#, ratio_demand_cost
from SquareDivision.optimization.bounds import bounds_trust_constr
from SquareDivision.optimization.initial_guess import contact_universal_x0
from SquareDivision.draw.draw import draw_rectangles, rectangle_numbers
from SquareDivision.config import config



class Rectangulation():
    def __init__(self, config=config):
        self.rng:Generator = np.random.default_rng(config['seed'])
    
    def sample_centers(self, strategy:CentersGenerationStrategy, **kwargs):
        """ Execute strategy to generate centers sample """
        self.centers = strategy.generate(**kwargs)
    
    # def sample_size(self, startegy:WidthHeightStrategy, **kwargs):
    #     return startegy.generate(**kwargs)
    
    # def sample_widths(self, **kwargs):
    #     self.widths = self.sample_size(**kwargs)

    # def sample_heights(self, **kwargs):
    #     self.heights = self.sample_size(**kwargs)

    def sample_widths(self, startegy:WidthHeightStrategy, **kwargs):
        """ Execute strategy to generate widths sample """
        self.widths = startegy.generate(**kwargs)

    def sample_heights(self, startegy:WidthHeightStrategy, **kwargs):
        """ Execute strategy to generate heights sample """
        self.heights = startegy.generate(**kwargs)
    
    def build_rectangles(self, outside_rectangles:np.ndarray=None):
        """ Joins: self.centers, self.widths, self.heights into self.rectangles,
                or
            if outside_rectangles is not None loads is into self.rectangles """
        if outside_rectangles is None:
            self.rectangles = np.c_[self.centers, self.widths, self.heights]
        else:
            self.rectangles = outside_rectangles

    def load_distributions(self, fun = x_plus_y_func):
        # this needs to be changed to accept two functions maybe to strategy?
        self.func = functools.partial(fun,
                         min_00=0.025, max_00=0.03,
                         min_11=0.2, max_11=0.3,
                         rng= self.rng)
        
    def sample(self, num = 10):
        # this needs to be changed maybe to strategy?
        # self.arr_sample = arg_rect_list(num, uniform_pts, self.func, rng=self.rng)
        self.arr_sample = rects_from_distributions(
            num,
            pts_func=uniform_pts,
            width_distribution=tepui(base=0.05, top=0.5, slope=5, vertex=1.5),
            height_distribution=tepui(base=0.05, top=0.5, slope=5, vertex=1.5),
            rng=self.rng)

    def find_disjoint_family(self):
        self.arr = find_anchors_and_crop(self.arr_sample)
        self.arr = sort_by_area(self.arr)
        self.arr = remove_smaller(self.arr)

    def inflate(self):
        self.clinched_rectangles = inflate_rectangles(self.arr)
        self.clinched_rectangles = np.maximum(0, self.clinched_rectangles)

    def graph_processing(self, rectangles=None):
        rectangles = self.clinched_rectangles if rectangles is None else rectangles
        self.east_neighbours = contact_graph_incidence_matrix(rectangles, 'r').astype(int)
        self.north_neighbours=contact_graph_incidence_matrix(rectangles, 'u').astype(int)
        self.east_graph = nx.from_numpy_array(self.east_neighbours)
        self.north_graph= nx.from_numpy_array(self.north_neighbours)
        self.holes = find_holes(self.east_neighbours, self.north_neighbours)
        self.possible_holes_idxs = holes_idxs(rectangles, self.holes)
        self.holes_idxs = check_holes(rectangles, self.possible_holes_idxs)
    
    def execute(self, **kwargs):
        self.load_distributions()
        self.sample(**kwargs)
        self.find_disjoint_family()
        self.inflate()
        self.graph_processing()
    
    def prepare_constraints(self, keep_feasible=True):
        self.x0 = self.clinched_rectangles.flatten()

        self.bounds = bounds_trust_constr(self.clinched_rectangles, keep_feasible=keep_feasible)
        # self.linear____constr = linear_constraints(
        #     self.clinched_rectangles, 
        #     self.east_neighbours, 
        #     self.north_neighbours,
        #     keep_feasible=keep_feasible
        #     )
        # self.holes_constr = hole_closing_constraints(
        #     self.holes_idxs, 
        #     self.clinched_rectangles, 
        #     keep_feasible=keep_feasible
        #     )
        self.constraint = linear_constraint(
            self.clinched_rectangles, 
            self.east_neighbours, 
            self.north_neighbours,
            self.holes_idxs,
            keep_feasible=keep_feasible
        )

    def close_holes(self):
        self.sol = orth_proj_onto_affine_L(self.x0, self.constraint.A, self.constraint.lb)
        self.closed = self.sol.reshape(-1,4)

    def report(self, tol=0.03, digits = 3):
        """ report relative distances between rectangles in clinched_rectangles and closed"""
        relative_diff = np.linalg.norm(self.clinched_rectangles-self.closed, axis=1)/np.linalg.norm(self.clinched_rectangles, axis=1)
        order = np.argsort(relative_diff)[::-1]
        rel_diff_ord = relative_diff[order] # sorted form the higiest to lowest 
        touched_beyoned_tolerance = np.where(rel_diff_ord > tol)[0]
        if len(touched_beyoned_tolerance) > 0:
            for no in touched_beyoned_tolerance:
                print(f'rectangle no.{order[no]:>3} relatively changed by {rel_diff_ord[no]:>{digits+3}.{digits}f} ')
        else:
            print(f'All rectangles within tolerace')

    def draw(self, disjoint:bool, inflated:bool, closed:bool, size:int=5):
        num_of_axes = np.array([disjoint, inflated, closed]).sum()
        fig, axes = plt.subplots(
            nrows=1,
            ncols=num_of_axes, 
            figsize=(num_of_axes * size, size))
        axes = [axes] if num_of_axes == 1 else axes
        i = 0 
        if disjoint is True:
            axes[i] = draw_rectangles(axes[i], self.arr)
            axes[i] = rectangle_numbers(axes[i],  self.arr)
            i += 1
        if inflated is True:
            axes[i] = draw_rectangles(axes[i],  self.clinched_rectangles)
            axes[i] = rectangle_numbers(axes[i],  self.clinched_rectangles)
            i += 1
        if closed is True:
            axes[i] = draw_rectangles(axes[i], self.closed)
            axes[i] = rectangle_numbers(axes[i],  self.closed)
            i += 1
        plt.show()


import functools

import numpy as np
from numpy.random._generator import Generator
from scipy.optimize import minimize
import networkx as nx

import matplotlib.pyplot as plt

from SquareDivision.src.distributions import x_plus_y_func, tepui
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
from SquareDivision.optimization.constraints import linear_constraints, nonlinear_constraints
from SquareDivision.optimization.objective_function import dist_fun#, ratio_demand_cost
from SquareDivision.optimization.bounds import bounds_trust_constr
from SquareDivision.optimization.initial_guess import contact_universal_x0
from SquareDivision.draw.draw import draw_rectangles, rectangle_numbers
from SquareDivision.config import config


class Rectangulation():
    def __init__(self, config=config):
        self.rng:Generator = np.random.default_rng(config['seed'])
    
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

    def graph_processing(self):
        self.east_neighbours = contact_graph_incidence_matrix(self.clinched_rectangles, 'r').astype(int)
        self.north_neighbours=contact_graph_incidence_matrix(self.clinched_rectangles, 'u').astype(int)
        self.east_graph = nx.from_numpy_array(self.east_neighbours)
        self.north_graph= nx.from_numpy_array(self.north_neighbours)
        self.holes = find_holes(self.east_neighbours, self.north_neighbours)
        self.possible_holes_idxs = holes_idxs(self.clinched_rectangles, self.holes)
        self.holes_idxs = check_holes(self.clinched_rectangles, self.possible_holes_idxs)
    
    def execute(self, **kwargs):
        self.load_distributions()
        self.sample(**kwargs)
        self.find_disjoint_family()
        self.inflate()
        self.graph_processing()
    
    def prepare_constraints(self):
        # self.x0 = contact_universal_x0(self.clinched_rectangles[:, :4])
        self.x0 = self.clinched_rectangles.flatten()
        self.bounds = bounds_trust_constr(self.clinched_rectangles)
        self.linear____constr = linear_constraints(
            self.clinched_rectangles, 
            self.east_neighbours, 
            self.north_neighbours,
            )
        self.nonlinear_constr = nonlinear_constraints(
            self.east_graph, 
            self.north_graph, 
            self.holes_idxs
            )
    def close_holes(self):
        self.sol = minimize(
            fun= lambda x : dist_fun(x, clinched_rectangles=self.clinched_rectangles), 
            x0=self.x0,
            jac=True, 
            method='trust-constr', 
            constraints=self.linear____constr + self.nonlinear_constr,
            bounds = self.bounds)
        self.closed = self.sol.x.reshape(-1,4)

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


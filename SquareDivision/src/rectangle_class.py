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
from SquareDivision.optimization.constraints import constraints_trust_constr
from SquareDivision.optimization.objective_function import dist_fun, ratio_demand_cost
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
    
    def execute(self, **kwargs):
        self.load_distributions()
        self.sample(**kwargs)
        self.find_disjoint_family()
        self.inflate()
        self.graph_processing()
    
    def draw(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        axes[0] = draw_rectangles(axes[0], self.arr[:,:4])
        axes[0] = rectangle_numbers(axes[0],  self.arr[:,:4])
        axes[1] = draw_rectangles(axes[1],  self.clinched_rectangles[:,:4])
        axes[1] = rectangle_numbers(axes[1],  self.clinched_rectangles[:,:4])
        plt.show()
    
    def prepare_constraints(self):
        self.x0 = contact_universal_x0(self.clinched_rectangles[:, :4])
        self.bounds = bounds_trust_constr(self.clinched_rectangles[:, :4])
        self.holes_idxs = holes_idxs(self.clinched_rectangles, self.holes)
        self.holes_idxs = check_holes(self.clinched_rectangles,self.holes_idxs)
        self.const_trust = constraints_trust_constr(
            self.clinched_rectangles[:,:4], 
            self.east_neighbours, 
            self.north_neighbours,
            self.holes_idxs
        )
    def close_holes(self):
        self.sol = minimize(
            # ratio_demand_cost,
            fun=lambda x : dist_fun(x, clinched_rectangles=self.clinched_rectangles[:,:4]), 
            x0=self.clinched_rectangles[:,:4].flatten(), #self.x0,#
            # args=(self.clinched_rectangles[:,:4]), 
            jac=True, 
            method='trust-constr', 
            constraints=self.const_trust)
            # bounds = self.bounds)
    
    def draw_closed(self):
        self.closed = self.sol.x.reshape(-1,4)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        axes[0] = draw_rectangles(axes[0], self.arr[:,:4])
        axes[0] = rectangle_numbers(axes[0],  self.arr[:,:4])
        axes[1] = draw_rectangles(axes[1],  self.clinched_rectangles[:,:4])
        axes[1] = rectangle_numbers(axes[1],  self.clinched_rectangles[:,:4])
        axes[2] = draw_rectangles(axes[2], self.closed)
        axes[2] = rectangle_numbers(axes[2],  self.closed)
        plt.show()

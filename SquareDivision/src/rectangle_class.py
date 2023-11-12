import functools

import numpy as np
from numpy.random._generator import Generator

import matplotlib.pyplot as plt

from SquareDivision.src.distributions import x_plus_y_func
from SquareDivision.src.generators import uniform_pts
from SquareDivision.src.dataflow import (
    find_anchors_and_crop,
    sort_by_area,
    remove_smaller,
    inflate_rectangles,
    arg_rect_list)
from SquareDivision.contact_graph.incidence_matrix import (
    contact_graph_incidence_matrix
)
from SquareDivision.holes.detect import find_holes, holes_idxs
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
        self.arr_sample = arg_rect_list(num, uniform_pts, self.func, rng=self.rng)

    def find_disjoint_family(self):
        self.arr = find_anchors_and_crop(self.arr_sample)
        self.arr = sort_by_area(self.arr)
        self.arr = remove_smaller(self.arr)

    def inflate(self):
        self.clinched_rectangles = inflate_rectangles(self.arr)
    
    def graph_processing(self):
        self.east_neighbours = contact_graph_incidence_matrix(self.clinched_rectangles, 'r').astype(int)
        self.north_neighbours=contact_graph_incidence_matrix(self.clinched_rectangles, 'u').astype(int)
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
        self.holes_idxs = holes_idxs(self.clinched_rectangles, self.holes)


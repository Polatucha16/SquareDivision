import numpy as np

def rectangles_disjoint_in_axis_Q(r0: np.ndarray, r1: np.ndarray, axis: int):
    """
    ri is [x,y,width,height] for i in 0,1
    checking if
        r0 is below r1 OR r1 is below r0
    """
    r0_r1 = r0[axis] + r0[axis + 2] - r1[axis]
    r1_r0 = r1[axis] + r1[axis + 2] - r0[axis]
    r0_r1_Q = r0_r1 <= 0 or np.isclose(r0_r1, 0)
    r1_r0_Q = r1_r0 <= 0 or np.isclose(r1_r0, 0)
    return r0_r1_Q or r1_r0_Q

def rectangles_disjoint_Q(r0: np.ndarray, r1: np.ndarray):
    """ 
    if rectangles are disjoint in one of the axies they are disjoint"""
    disjoint_on_axis_0 = rectangles_disjoint_in_axis_Q(r0, r1, 0)
    disjoint_on_axis_1 = rectangles_disjoint_in_axis_Q(r0, r1, 1)
    return disjoint_on_axis_0 or disjoint_on_axis_1

def disjoint_family(rect_family: np.ndarray):
    """ 
    check if rectangles in rect_family (N, 4) is 
    family of disjoint rectangles or not
    """
    shape = (len(rect_family), len(rect_family))
    disjoint_Q = np.zeros(shape=shape)
    for i, r0 in enumerate(rect_family):
        for j, r1 in enumerate(rect_family):
            if i != j:
                disjoint_Q[i,j] = rectangles_disjoint_Q(r0, r1)
            else:
                continue
    all_disjoint_Q = np.all(disjoint_Q == 1 - np.eye(shape[0],shape[1]))
    return all_disjoint_Q

from SquareDivision.src.rectangle_class import Rectangulation
from SquareDivision.src.distributions import  BetweenFunctions,tepui

tepui_kwargs_width = {
    'top' : 0.3, 
    'bottom' : 0.02, 
    'slope' : 3, 
    'vertex' : 0.5, 
    'pts' : np.array([[0.25, 0.25],[0.75, 0.75]])
    }
tepui_kwargs_height = {
    'top' : 0.3, 
    'bottom' : 0.02, 
    'slope' : 3, 
    'vertex' : 0.5, 
    'pts' : np.array([[0.75, 0.25],[0.25, 0.75]])
    }

width_0 = lambda mid_pt: tepui(mid_pt, **tepui_kwargs_width)
width_1 = width_0
height_0 = lambda mid_pt: tepui(mid_pt, **tepui_kwargs_height)
height_1 = height_0

rngrects = Rectangulation(config={"seed": 12345678})
kwargs = {'num' : 20,
          'widths_strategy' : BetweenFunctions(func_0=width_0, func_1=width_1, rng=rngrects.rng),
          'heights_strategy': BetweenFunctions(func_0=height_0, func_1=height_1, rng=rngrects.rng)}

def test_stability(n=20, seed=0, kwargs=kwargs) -> None:
    rng = np.random.default_rng(seed=seed)
    rect_seeds = rng.integers(low=1_000_000, high=9_999_999, size=n)
    result_record = []
    for i in range(n):
        rects = Rectangulation(config={"seed": rect_seeds[i]})
        rects.sample_rectangles(**kwargs) 
        rects.perform()
        result_record.append(disjoint_family(rects.clinched_rectangles))
    assert result_record == [True] * n


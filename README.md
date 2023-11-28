# Retrangulation from random family of rectangles

3 stages of family of rectangles:\
<img src="SquareDivision\output_example.png" alt="example"/>
Widths and heights of initial rectangles were drawn from the function called tepui:\
<img src="SquareDivision\tepui_distribution.png" alt="tepui_distribution"/>

How to use:\
```python
import numpy as np
from SquareDivision.src.rectangle_class import Rectangulation
from SquareDivision.src.distributions import BetweenFunctions

def surface_perp_to(pt, vect: np.ndarray, val_at_0: float):
    return -vect[:2].dot(pt) / vect[2] + val_at_0

rects = Rectangulation(config={"seed": 123567})
# define sizes of 
width_0 = lambda mid_pt: surface_perp_to(mid_pt, vect = np.array([0, -1, 5]), val_at_0 = 0.005)
width_1 = lambda mid_pt: surface_perp_to(mid_pt, vect = np.array([0, -2, 10]), val_at_0 = 0.01)
height_0 = lambda mid_pt: surface_perp_to(mid_pt, vect = np.array([-1, 0, 5]), val_at_0 = 0.005)
height_1 = lambda mid_pt: surface_perp_to(mid_pt, vect = np.array([-2, 0, 10]), val_at_0 = 0.01)

rects.execute(
    num=1500, 
    widths_strategy = BetweenFunctions(func_0=width_0, func_1=width_1, rng=rects.rng), 
    heights_strategy= BetweenFunctions(func_0=height_0, func_1=height_1, rng=rects.rng), 
)
rects.prepare_closing()
rects.close_holes()
rects.draw(disjoint=True, inflated=True, inflated_nums=True, closed=True, closed_nums=False)
```
then the output should be:\
<img src="SquareDivision\output_after_codebox.png" alt="example"/>
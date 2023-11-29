# Retrangulation from random family of rectangles

3 stages of family of rectangles:\
<img src="SquareDivision\output_example.png" alt="example"/>
Widths and heights of initial rectangles were set by values of the following function:
```python
from SquareDivision.src.distributions import tepui
from SquareDivision.draw.draw import draw_func
tepui_kwargs = {'bottom': 0.1, 'top': 0.45, 'vertex': 0.6, 'slope': 2}
draw_func(tepui, func_kwargs = tepui_kwargs )
```
<img src="SquareDivision\tepui_distribution.png" alt="tepui_distribution"/>

# How to use:
```python
import numpy as np
from SquareDivision.src.rectangle_class import Rectangulation
from SquareDivision.src.distributions import BetweenFunctions, surface_perp_to

rects = Rectangulation(config={"seed": 123567})


width_0 = lambda mid_pt: surface_perp_to(mid_pt, vect = np.array([0, -1, 5]), val_at_0 = 0.005)
width_1 = lambda mid_pt: surface_perp_to(mid_pt, vect = np.array([0, -2, 10]), val_at_0 = 0.01)

height_0 = lambda mid_pt: surface_perp_to(mid_pt, vect = np.array([-1, 0, 5]), val_at_0 = 0.005)
height_1 = lambda mid_pt: surface_perp_to(mid_pt, vect = np.array([-2, 0, 10]), val_at_0 = 0.01)

rects.sample_rectangles(
    num=500, 
    widths_strategy = BetweenFunctions(func_0=width_0, func_1=width_1, rng=rects.rng), 
    heights_strategy= BetweenFunctions(func_0=height_0, func_1=height_1, rng=rects.rng), 
)
rects.find_disjoint_family()
rects.clinch()
rects.close_holes()

rects.report(tol=0.0005, digits=4, limit_list=20)
rects.draw(disjoint=True, inflated=True, inflated_nums=True, closed=True, closed_nums=False)
```
Then the output should be:
```
rectangle no. 23 relatively changed by  0.0014
rectangle no. 12 relatively changed by  0.0011 
rectangle no.  0 relatively changed by  0.0007 
rectangle no. 17 relatively changed by  0.0006 
rectangle no.  6 relatively changed by  0.0006 
rectangle no.  2 relatively changed by  0.0005
```
<img src="SquareDivision\output_after_codebox.png" alt="example"/>

This time the distribution of width and height is uniform between two linear functions.
An example of one of those functions we plot below:
```python
import numpy as np
from SquareDivision.src.distributions import surface_perp_to
from SquareDivision.draw.draw import draw_func
surface_perp_to_kwargs = {'vect' : np.array([0, -1, 5]), 'val_at_0' : 0.005}
draw_func(surface_perp_to, func_kwargs = surface_perp_to_kwargs)
```
<img src="SquareDivision\surface_perp_to_boundary.png" alt="example perp"/>


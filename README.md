# Retrangulation from random family of rectangles

3 stages of family of rectangles:\
<img src="README_pictures\output_example.png" alt="example"/>
Widths and heights of initial rectangles were set by values of the following function:
```python
from SquareDivision.src.distributions import tepui
from SquareDivision.draw.draw import draw_func
tepui_kwargs = {'bottom': 0.1, 'top': 0.45, 'vertex': 0.6, 'slope': 2}
draw_func(tepui, func_kwargs = tepui_kwargs )
```
<img src="README_pictures\tepui_distribution.png" alt="tepui_distribution"/>

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
<img src="README_pictures\output_after_codebox.png" alt="example"/>

This time the distribution of width and height is uniform between two linear functions.
An example of one of those functions we plot below:
```python
import numpy as np
from SquareDivision.src.distributions import surface_perp_to
from SquareDivision.draw.draw import draw_func
surface_perp_to_kwargs = {'vect' : np.array([0, -1, 5]), 'val_at_0' : 0.005}
draw_func(surface_perp_to, func_kwargs = surface_perp_to_kwargs)
```
<img src="README_pictures\surface_perp_to_boundary.png" alt="example perp"/>

## Contact graph

After the rectangles are clinched we can produce contact graphs, this is done by\
```python graph_processing()``` method and the results are stored in :\
```python self.east_neighbours``` and ```python self.north_neighbours``` - incidence matrices of from left to right contacts and 
from bottom to up constacts respectively.\
```python self.east_graph``` and  ```python self.north_graph``` - XNetwork graphs objects build from incidence matrices.\
```python self.holes_idxs``` - the list of rectangle indecies bounding holes. Each element is pair of pairs
representing [left and right] and [bottom and upper] bound of a hole in clinched rectangles.\
\
```python draw_contact_graph(i)``` method draw contacts graphs for ```python i``` equal to:
0 - disjoint sample;\
1 - clinched;\
2 - closed.\

Notice the hole [[3,0], [2,1]] in ```python self.holes_idxs``` notation, that hole in rectangles is
represented as hole(chordless cycle) in upper right corner of the graph below.
```python
rects.draw_contact_graph(1)
```
<img src="README_pictures\contact_graph_linear_distribution.png" alt="example contact graph"/>


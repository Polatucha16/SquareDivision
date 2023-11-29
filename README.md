# Retrangulation from random family of rectangles

3 stages of family of rectangles:\
<img src="README_pictures\output_example.png" alt="example"/>

Widths and heights of initial rectangles in **Rectangles sample** were set by values of the following function:
```python
from SquareDivision.src.distributions import tepui
from SquareDivision.draw.draw import draw_func
tepui_kwargs = {'bottom': 0.1, 'top': 0.45, 'vertex': 0.6, 'slope': 2}
draw_func(tepui, func_kwargs = tepui_kwargs )
```
<img src="README_pictures\tepui_distribution.png" alt="tepui_distribution"/>

# How to use and what is going on:
The idea is to define distributions for: centers, widths, and heights of rectangles.\
The centers are drawn from uniform distribution on `[0,1)^2`.
1. The strategy `BetweenFunctions` does the following:\
for the point `(x, y)` evaluate `func_0(x,y)`, `func_1(x,y)` and draw a number `w` from the uniform distribution:
`U(func_0(x,y), func_1(x,y))` say it will be width.\
Do the similar for height `h` with possibly different functions or different way of sampling,\
then create a rectangle with center `(x,y)`, width `w` & height `h` and add it to primordial set of rectangles.
2. After initial sampling is done `find_disjoint_family()` method pick disjoint family of rectangles.
3. For the disjoint family method `clinch()` inflate rectangles so that every single one is touching other rectangles or
boundary of square `[0,1]^2`.
4. Finally `close_holes()` decide how and moddify clinched rectangle in such a way to remove holes.

Example in code:

```python
import numpy as np
from SquareDivision.src.rectangle_class import Rectangulation
from SquareDivision.src.distributions import FromFunction, BetweenFunctions, tepui, surface_perp_to

rects = Rectangulation(config={"seed": 123567})

# Define boundaries of width & height distributions here:
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

rects.report(tol=0.01, digits=4, limit_list=3)
rects.draw(disjoint=True, inflated=True, inflated_nums=True, closed=True, closed_nums=False)
```
Then the output should be:
```
rectangle no. 23 relatively changed by  0.0627 
rectangle no. 12 relatively changed by  0.0560 
rectangle no.  0 relatively changed by  0.0452 
```
<img src="README_pictures\output_after_codebox.png" alt="example"/>

This time, the distribution of widths and heights is uniform between two linear functions.
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
```graph_processing()``` method and the results are stored in :\
```self.east_neighbours``` and ```self.north_neighbours``` - incidence matrices of from left to right contacts and 
from bottom to up constacts respectively.\
```self.east_graph``` and  ```self.north_graph``` - NetworkX graphs build from the above incidence matrices.\
```self.holes_idxs``` - list of indecies of rectangles bounding holes. Each element is pair of pairs
representing [left and right] and [bottom and upper] bound of a hole in clinched rectangles.\
\
```draw_contact_graph(i)``` method draws contacts graphs. Depending on  ```i``` it draws:\
0 - disjoint sample;\
1 - clinched;\
2 - closed.

Notice in upper right corner of the picture below that the hole ```[[3,0], [2,1]]``` in ```self.holes_idxs``` notation is a hole in clinched rectangles\
 and it is also a hole (chordless cycle) in the graph constructed from the union of ```self.east_graph``` and  ```self.north_graph```.
```python
rects.draw_contact_graph(1)
```
<img src="README_pictures\contact_graph_linear_distribution.png" alt="example contact graph"/>


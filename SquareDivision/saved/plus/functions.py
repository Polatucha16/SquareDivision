import numpy as np
def distToIntervalAB(pt, ax=0, ay=0, bx=1, by=1):
   # Define the points as numpy arrays
   p = np.array(pt)
   a = np.array([ax, ay])
   b = np.array([bx, by])

   # Calculate the normalized tangent vector
   d = np.divide(b - a, np.linalg.norm(b - a))

   # Calculate the signed parallel distance components
   s = np.dot(a - p, d)
   t = np.dot(p - b, d)

   # Calculate the clamped parallel distance
   h = np.maximum.reduce([s, t, 0])

   # Calculate the perpendicular distance component
   c = np.cross(p - a, d)

   # Return the Euclidean distance
   return np.hypot(h, np.linalg.norm(c))

def cross_ABCD(pt, bottom=0, slope=1, ax=0, ay=0, bx=1, by=1,cx=0, cy=1, dx=1, dy=0):
    a = slope * distToIntervalAB(pt, ax, ay, bx, by)
    b = slope * distToIntervalAB(pt, cx, cy, dx, dy)
    return bottom + min(a, b)

cross_ABCD_kwargs = {'bottom':0.02,'slope':0.3,'ax':0.25, 'ay':0.5, 'bx':0.75, 'by':0.5, 'cx':0.5, "cy":0.25, 'dx':0.5, 'dy':0.75}
width_0 = lambda mid_pt: cross_ABCD(mid_pt, **cross_ABCD_kwargs)
width_1 = width_0
height_0 = lambda mid_pt: cross_ABCD(mid_pt, **cross_ABCD_kwargs)
height_1 = height_0

num =4000
seed = 12345378
import numpy as np

def contact_universal_x0(clinched_rectangles:np.ndarray):
    """Return `x0` argument the initial guess.
        In order to fullfil complicated contact conditions we : 
        1) Find rectangle such that it has (x,y) = (0,0) this rectacnge
            and therefore that row will become the row: `[0, 0, 1, 1, 1]`
        2) Rectangles on the most left are degerated to 
            the interval from (0,1) to (1,1) and rows become: `[0, 1, 1, 0, 0]`
        3) Rectangles on the bottom are degerated to 
            the interval from (1,0) to (1,1) and rows become: `[1, 0, 0, 1, 0]`
        4) All other are squished to a point (1,1) and rows become: `[1, 1, 0, 0, 0]`
    """
    x0=np.zeros(shape=clinched_rectangles.shape)
    for num, row in enumerate(clinched_rectangles):
        if np.isclose(row[0], 0) and np.isclose(row[1], 0):
            x0[num] = np.array([0, 0, 1, 1])
            continue
        elif np.isclose(row[0], 0) and (not np.isclose(row[1], 0)):
            x0[num] = np.array([0, 1, 1, 0])
            continue
        elif (not np.isclose(row[0], 0)) and np.isclose(row[1], 0):
            x0[num] = np.array([1, 0, 0, 1])
            continue
        else:
            x0[num] = np.array([1, 1, 0, 0])
    return x0.flatten()
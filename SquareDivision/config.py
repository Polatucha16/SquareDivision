config = {
    "seed": 1234, 
    "fig_eps": 0.05
    }

figure_settings = {
    "figsize": (4, 4)
    }

axis_settings = {
    "xlim": (0 - config["fig_eps"], 1 + config["fig_eps"]),
    "ylim": (0 - config["fig_eps"], 1 + config["fig_eps"]),
    "aspect": "equal",
    "frame_on": False,
    "xticks" : [],
    "yticks" : []
}

#### Boundaries of height and width linearly depend on position ####

# import numpy as np
# from SquareDivision.src.distributions import surface_perp_to

# surface_perp_to_kwargs_width_0 = {
#     'vect'      : np.array([0, -1,  5]),
#     'val_at_0'  : 0.015
# }
# surface_perp_to_kwargs_width_1 = {
#     'vect'      : np.array([0, -2, 10]),
#     'val_at_0'  : 0.02
# }
# surface_perp_to_kwargs_height_0 = {
#     'vect'      : np.array([-1, 0,  5]),
#     'val_at_0'  : 0.015
# }
# surface_perp_to_kwargs_height_1 = {
#     'vect'      : np.array([-2, 0, 10]),
#     'val_at_0'  : 0.02
# }
# width_0 = lambda mid_pt: surface_perp_to(mid_pt, **surface_perp_to_kwargs_width_0)
# width_1 = lambda mid_pt: surface_perp_to(mid_pt, **surface_perp_to_kwargs_width_1)
# height_0 = lambda mid_pt: surface_perp_to(mid_pt, **surface_perp_to_kwargs_height_0)
# height_1 = lambda mid_pt: surface_perp_to(mid_pt, **surface_perp_to_kwargs_height_1)

#### 4 bigger cirles of alternately elongated rectangles ####

# import numpy as np
# from SquareDivision.src.distributions import tepui

# tepui_kwargs_width = {
#     'top' : 0.3, 
#     'bottom' : 0.02, 
#     'slope' : 3, 
#     'vertex' : 0.5, 
#     'pts' : np.array([[0.25, 0.25],[0.75, 0.75]])
#     }
# tepui_kwargs_height = {
#     'top' : 0.3, 
#     'bottom' : 0.02, 
#     'slope' : 3, 
#     'vertex' : 0.5, 
#     'pts' : np.array([[0.75, 0.25],[0.25, 0.75]])
#     }

# width_0 = lambda mid_pt: tepui(mid_pt, **tepui_kwargs_width)
# width_1 = width_0
# height_0 = lambda mid_pt: tepui(mid_pt, **tepui_kwargs_height)
# height_1 = height_0
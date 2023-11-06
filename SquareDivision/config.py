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
}

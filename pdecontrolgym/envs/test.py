import numpy as np
import matplotlib.pyplot as plt
from figure import PlotGenerator

x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)
figSettings = {
    "show_error_figure": True,
    "show_fig": False,
    "save_fig": False,
    "fig_xlabel": "x",
    "fig_ylabel": "t",
    "fig_zlabel": "$u(x, t)$",
    "rstride": 1,
    "cstride": 1,
    "fig_size": 516,
}
fig = plt.figure()
figSettings["save_fig"] = True
figSettings["file_name"] = "tester.png"
figObj = PlotGenerator(figSettings)
u = np.random.uniform(-10, 10, size=(len(t), len(x)))
figObj.makeFigure(x, t, u)

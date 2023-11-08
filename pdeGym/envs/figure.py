# This file provides all the code for figure and animation generators

# Typically, the methods here are only expected to be called from rendering environments
# but they are public and so can be accessed individually from outside the environment. 
# See bottom of the file for details on creating custom figures

# Handles all the Figure creation, animation creations
# And comparison figures.
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from .rewards import NormReward

# Update tex fonts
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "times",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

plt.rcParams.update(tex_fonts)

# Build Default Settings for Plotting 


class PlotGenerator():
    def __init__(self, figSettings=None):
        self.fig = plt.figure(figsize=self.set_figure_size(516, 0.99, (1, 1), height_add=1)) 
        self.figObject = FigureObj(self.fig, figSettings)

    def makeFigure(self, x, t, u):
        self.figObject.plotData(x, t, u)

    def set_figure_size(self, width, fraction=1, subplots=(1, 1), height_add=0):
        """Set figure dimensions to avoid scaling in LaTeX.

        Parameters
        ----------
        width: float or string
                Document width in points, or string of predined document type
        fraction: float, optional
                Fraction of the width which you wish the figure to occupy
        subplots: array-like, optional
                The number of rows and columns of subplots.
        Returns
        -------
        fig_dim: tuple
                Dimensions of figure in inches
        """
        if width == 'thesis':
            width_pt = 426.79135
        elif width == 'beamer':
            width_pt = 307.28987
        else:
            width_pt = width

        # Width of figure (in pts)
        fig_width_pt = width_pt * fraction
        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        golden_ratio = (5**.5 - 1) / 2

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = height_add + fig_width_in * golden_ratio * (subplots[0] / subplots[1])

        return (fig_width_in, fig_height_in)

# Builds a single figure on the image fig according to figSettings
class FigureObj():
    # Static variables for default plotting settings
    default_reward = NormReward(2, "temporal")
    default_fig_settings = {
        "show_error_figure": False,
        "save_fig": False,
        "show_fig": True,
        "fig_xlabel": "x",
        "fig_ylabel": "t", 
        "fig_zlabel": "$u(x, t)$", 
        "fig_title": "PDE with Control",
        "error_fig_xlabel": "t",  
        "error_fig_ylabel": "Error", 
        "error_fig_title": "Reward",
        "reward_func": default_reward, 
        "rstride": 10, 
        "cstride": 100
        }

    
    def __init__(self, fig, figSettings=None):
        if figSettings is None:
            self.fig_settings = FigureObj.default_fig_settings
        else: 
            self.fig_settings = {**(FigureObj.default_fig_settings), **figSettings}
        self.fig = fig

    def plotData(self, x, t, u):
        if self.fig_settings["show_error_figure"]:
            self._plotFigAndErrorFig(x, t, u)
        else:
            self._plotFig(self.fig, x, t, u)
        if self.fig_settings["save_fig"]:
            self._savefig()
        if self.fig_settings["show_fig"]:
            plt.show()

    def _plotFig(self, fig, x, fullt, u):
        # These settings create beautiful PDE figures. Modification not recommended. 
        ax = fig.subplots(nrows=1, ncols=1, subplot_kw={"projection": "3d", "computed_zorder": False})
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo['axisline']['linewidth'] = 1
            axis._axinfo['axisline']['color'] = "b"
            axis._axinfo['grid']['linewidth'] = 0.2
            axis._axinfo['grid']['linestyle'] = "--"
            axis._axinfo['grid']['color'] = "#d1d1d1"
            axis.set_pane_color((1, 1, 1))
        ax.set_xlabel(self.fig_settings["fig_xlabel"])
        ax.set_ylabel(self.fig_settings["fig_ylabel"])
        ax.set_zlabel(self.fig_settings["fig_zlabel"], rotation=90)
        ax.zaxis.set_rotate_label(False)
        meshx, mesht = np.meshgrid(x, fullt)
        ax.plot_surface(meshx, mesht, u, edgecolor="black", lw=0.2, rstride=int(len(fullt)/self.fig_settings["rstride"]), cstride=int((len(x)/self.fig_settings["cstride"])), alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
        ax.view_init(10, 15)
        ax.set_yticks(np.linspace(min(fullt), max(fullt), 5))

    def _plotFigAndErrorFig(self, x, fullt, u):
        subfigs = self.fig.subfigures(nrows=2, ncols=1, hspace=0)
        ax = subfigs[1].subplots(nrows=1, ncols=1)
        subfigs[1].subplots_adjust(left=0.17, bottom=0.18, right=0.87, top=0.9)
        ax.set_xlabel(self.fig_settings["error_fig_xlabel"])
        ax.set_ylabel(self.fig_settings["error_fig_ylabel"])
        ax.set_xticks(np.linspace(min(fullt), max(fullt), 5))
        ax.plot(fullt[0: len(u)], [self.fig_settings["reward_func"].reward(uval) for uval in u])
        subfigs[0].suptitle(self.fig_settings["fig_title"])
        subfigs[1].suptitle(self.fig_settings["error_fig_title"])
        subfigs[0].subplots_adjust(left=0.01, bottom=0, right=1, top=1.15)
        self._plotFig(subfigs[0], x, fullt, u)

    def _savefig(self):
        self.fig.savefig(self.fig_settings["file_name"], dpi=300)


#class AnimationObj():
#    def __init__

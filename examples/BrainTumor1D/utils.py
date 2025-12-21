import matplotlib.pyplot as plt
import numpy as np

def set_size(width, fraction=1, subplots=(1, 1), height_add=0):
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

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "times",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 12,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize":10
}

plt.rcParams.update(tex_fonts)

def load_csv(filename):
    timesteps = []
    rewards = []
    with open(filename, "r") as f:
        # remove header
        line = f.readline()
        line = f.readline()
        while line:
            s = line.split(",")
            timesteps.append(int(s[1]))
            rewards.append(float(s[2]))
            line = f.readline()
    return timesteps, rewards

def load_csv_all(filename):
    walltime = []
    timesteps = []
    rewards = []
    with open(filename, "r") as f:
        # remove header
        line = f.readline()
        line = f.readline()
        while line:
            s = line.split(",")
            walltime.append(float(s[0]))
            timesteps.append(int(s[1]))
            rewards.append(float(s[2]))
            line = f.readline()
    return walltime, timesteps, rewards



linestyle_tuple = [
             ('loosely dotted',        (0, (1, 10))),
                  ('dotted',                (0, (1, 1))),
                       ('densely dotted',        (0, (1, 1))),
                            ('long dash with offset', (5, (10, 3))),
                                 ('loosely dashed',        (0, (5, 10))),
                                      ('dashed',                (0, (5, 5))),
                                           ('densely dashed',        (0, (5, 1))),

                                                ('loosely dashdotted',    (0, (3, 10, 1, 10))),
                                                     ('dashdotted',            (0, (3, 5, 1, 5))),
                                                          ('densely dashdotted',    (0, (3, 1, 1, 1))),

                                                               ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
                                                                    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                                                                         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


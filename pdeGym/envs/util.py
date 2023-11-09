import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp

def central_difference_x(f, dx=0.01):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (
        f[1:-1, 2:  ]
        -
        f[1:-1, 0:-2]
    ) / (
        2 * dx
    )
    return diff

def central_difference_y(f, dy=0.01):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (
        f[2:  , 1:-1]
        -
        f[0:-2, 1:-1]
    ) / (
        2 * dy
    )
    return diff

def laplace(f, dx=0.01, dy=0.01):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (
        f[1:-1, 0:-2]
        +
        f[0:-2, 1:-1]
        -
        4
        *
        f[1:-1, 1:-1]
        +
        f[1:-1, 2:  ]
        +
        f[2:  , 1:-1]
    ) / (
        dx * dy
    )
    return diff


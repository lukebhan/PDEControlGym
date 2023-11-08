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


def laplacian_operator(Nx, Ny, dx, dy):
    # Construct the Laplacian operator using finite differences
    diagonal = -4.0 * np.ones(Nx * Ny)
    off_diagonal_x = np.ones(Nx * Ny - 1)
    off_diagonal_y = np.ones(Nx * Ny - Nx)
    diagonals = [diagonal, off_diagonal_x, off_diagonal_x, off_diagonal_y, off_diagonal_y]
    laplacian = diags(diagonals, [0, -1, 1, -Nx, Nx], format="csr")
    # Account for grid spacing
    laplacian /= dx*dy
    return laplacian



def Diff_mat_1D(Nx):
    # First derivative
    D_1d = sp.diags([-1, 1], [-1, 1], shape = (Nx,Nx)) 
    D_1d = sp.lil_matrix(D_1d)
    D_1d[0,[0,1,2]] = [-3, 4, -1]    # 2rd order
    D_1d[Nx-1,[Nx-3, Nx-2, Nx-1]] = [1, -4, 3] 
    D2_1d =  sp.diags([1, -2, 1], [-1,0,1], shape = (Nx, Nx)) 
    D2_1d = sp.lil_matrix(D2_1d)                  
    D2_1d[0,[0,1,2,3]] = [2, -5, 4, -1] 
    D2_1d[Nx-1,[Nx-4, Nx-3, Nx-2, Nx-1]] = [-1, 4, -5, 2] 
    return D_1d, D2_1d


def Diff_mat_2D(Nx,Ny):
    Dx_1d, D2x_1d = Diff_mat_1D(Nx)
    Dy_1d, D2y_1d = Diff_mat_1D(Ny)
    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)
    Dx_2d = sp.kron(Iy,Dx_1d)
    Dy_2d = sp.kron(Dy_1d,Ix)
    D2x_2d = sp.kron(Iy,D2x_1d)
    D2y_2d = sp.kron(D2y_1d,Ix)
    return Dx_2d.tocsr(), Dy_2d.tocsr(), D2x_2d.tocsr(), D2y_2d.tocsr()
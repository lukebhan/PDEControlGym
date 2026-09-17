from setuptools import setup

setup(name="pdecontrolgym",
        version="0.0.1",
        install_requires=["gymnasium",
            "numpy",
            "matplotlib"],
        # The 3D data-center environment vendors the Han et al. (2021) FFD-Upwind
        # solver (pde_control_gym/src/environments3d/{ffd_upwind,datacenter}).
        # Kept out of the base install so the core library stays lightweight:
        #   pip install pdecontrolgym[datacenter]
        # scipy: required (sparse pressure Poisson solve, cKDTree, brentq).
        # numba: JIT for the hot Jacobi sweeps.
        # pyamg: the DEFAULT pressure-solver backend. Optional at runtime -- if it
        #   is not installed the solver falls back to the scipy-only sparse LU
        #   ('lu_mmd'), so it is a recommended-but-not-required part of the extra.
        extras_require={
            "datacenter": ["scipy", "numba", "pyamg"],
        },
        )

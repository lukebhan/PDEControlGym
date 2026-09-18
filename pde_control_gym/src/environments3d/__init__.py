from pde_control_gym.src.environments3d.base_env_3d import PDEEnv3D

__all__ = ["PDEEnv3D", "DataCenter3D"]


def __getattr__(name):
    # DataCenter3D pulls in the optional solver deps (scipy/numba/pyamg), so it
    # is imported lazily -- `import pde_control_gym` and the lightweight envs
    # never trigger it, but `from ...environments3d import DataCenter3D` works.
    if name == "DataCenter3D":
        from pde_control_gym.src.environments3d.datacenter3d import DataCenter3D
        return DataCenter3D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

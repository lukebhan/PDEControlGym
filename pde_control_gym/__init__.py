from gymnasium.envs.registration import register

register(
    id="PDEControlGym-TransportPDE1D", entry_point="pde_control_gym.src:TransportPDE1D"
)

register(
    id="PDEControlGym-ReactionDiffusionPDE1D", entry_point="pde_control_gym.src:ReactionDiffusionPDE1D"
)

register(
    id="PDEControlGym-BrainTumor1D", entry_point="pde_control_gym.src:BrainTumor1D"
)

register(
    id="PDEControlGym-TrafficPDE1D", entry_point="pde_control_gym.src:TrafficPDE1D"
)

register(
    id="PDEControlGym-NavierStokes2D", entry_point="pde_control_gym.src:NavierStokes2D"
)

# Deep entry point (not via pde_control_gym.src) so the optional solver deps
# (scipy/numba/pyamg) load only when this environment is actually made.
register(
    id="PDEControlGym-DataCenter3D", entry_point="pde_control_gym.src.environments3d.datacenter3d:DataCenter3D"
)

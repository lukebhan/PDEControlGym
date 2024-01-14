from gymnasium.envs.registration import register

register(
    id="PDEControlGym-HyperbolicPDE1D", entry_point="pde_control_gym.src:HyperbolicPDE1D"
)

register(
    id="PDEControlGym-ParabolicPDE1D", entry_point="pde_control_gym.src:ParabolicPDE1D"
)

register(
    id="PDEControlGym-NavierStokes2D", entry_point="pde_control_gym.src:NavierStokes2D"
)

from gymnasium.envs.registration import register

register(
    id="PDEControlGym-HyperbolicPDE1D", entry_point="pdecontrolgym.envs:HyperbolicPDE1D"
)

register(
    id="PDEControlGym-ParabolicPDE1D", entry_point="pdecontrolgym.envs:ParabolicPDE1D"
)

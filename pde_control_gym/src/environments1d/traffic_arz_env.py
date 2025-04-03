import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional
# from gym.envs.toy_text import discrete

from pde_control_gym.src.environments1d.base_env_1d import PDEEnv1D
from pde_control_gym.src.environments1d.traffic_arz_utils import Veq, F_r, F_y

class TrafficPDE(PDEEnv1D):
    r""" 
    Transport PDE 1D

    This class implements the 1D Transport PDE and inhertis from the class :class:`PDEEnv1D`. Thus, for a full list of of arguments, first see the class :class:`PDEEnv1D` in conjunction with the arguments presented here

    :param sensing_noise_func: Takes in a function that can add sensing noise into the system. Must return the same sensing vector as given as a parameter.
    :param reset_init_condition_func: Takes in a function used during the reset method for setting the initial PDE condition :math:`u(x, 0)`.
    :param reset_recirculation_func: Takes in a function used during the reset method for setting the initial plant parameter :math:`\beta` vector at the start of each epsiode.
    :param sensing_loc: Sets the sensing location as either ``"full"``, ``"collocated"``, or ``"opposite"`` which indicates whether the full state, the boundary at the same side of the control, or boundary at the opposite side of control is given as the observation at each time step.
    :param control_type: The control location can either be given as a ``"Dirchilet"`` or ``"Neumann"`` boundary conditions and is always at the ``X`` point. 
    :param sensing_type: Only used when ``sensing_loc`` is set to ``opposite``. In this case, the sensing can be either given as ``"Dirchilet"`` or ``"Neumann"`` and is given at the ``0`` point.
    :param limit_pde_state_size: This is a boolean which will terminate the episode early if :math:`\|u(x, t)\|_{L_2} \geq` ``max_state_value``.
    :param max_state_value: Only used when ``limit_pde_state_size`` is ``True``. Then, this sets the value for which the :math:`L_2` norm of the PDE will be compared to at each step asin ``limit_pde_state_size``.
    :param max_control_value: Sets the maximum control value input as between [``-max_control_value``, ``max_control_value``] and is used in the normalization of action inputs.
    :param control_sample_rate: Sets the sample rate at which the controller is applied to the PDE. This allows the PDE to be simulated at a smaller resolution then the controller.
    :param v_steady: Sets the steady state velocity of the PDE. This is used in the initial condition function.
    :param ro_steady: Sets the steady state density of the PDE. This is used in the initial condition function.
    :param v_max: Sets the maximum velocity of the PDE. This is used in the initial condition function.
    :param ro_max: Sets the maximum density of the PDE. This is used in the initial condition function.
    :param tau: Sets the relaxation time
    """
    def __init__(self, #sensing_noise_func: Callable[[np.ndarray], np.ndarray],
                #  reset_init_condition_func: Callable[[int], np.ndarray],
                #  reset_recirculation_func: Callable[[int], np.ndarray], 
                #  sensing_loc: str = "full",
                #  control_type: str= "Dirchilet", 
                #  sensing_type: str = "Dirchilet",
                 simulation_type: int = 1, 
                #  limit_pde_state_size: bool = False, 
                #  max_state_value: float = 1e10, 
                #  max_control_value: float = 20, 
                #  control_sample_rate: float=0.1,
                 v_steady: float = 10,
                 ro_steady: float = 0.12,
                 v_max: float = 40,
                 ro_max: float = 0.16,
                 v_desired: float = 10,
                 ro_desired: float = 0.12,
                 tau: float = 30,
                 **kwargs):
        super().__init__(**kwargs)
        # self.sensing_noise_func = sensing_noise_func
        # # self.reset_init_condition_func = reset_init_condition_func 
        # self.reset_recirculation_func = reset_recirculation_func
        # self.sensing_loc = sensing_loc
        # self.control_type = control_type
        # self.sensing_type = sensing_type
        # self.limit_pde_state_size = limit_pde_state_size
        # self.max_state_value = max_state_value
        # self.max_control_value = max_control_value
        # self.control_sample_rate = control_sample_rate
        # self.vs = v_steady
        # self.rs = ro_steady
        
        self.cont_scenario = simulation_type
        self.vm = v_max
        self.rm = ro_max
        self.qm = v_max * ro_max/4
        self.tau = tau

        
        if self.cont_scenario == 1:
            print('Case 1: Outlet Boundary Control')
        elif self.cont_scenario == 2:
            print('Case 2: Inlet Boundary Control')
        elif self.cont_scenario == 3:
            print('Case 3: Outlet & Inlet Boundary Control')
        elif self.cont_scenario == 4:
            print('Case 4: Stochastic Outlet Boundary Control for Training')
        elif self.cont_scenario == 5:
            print('Case 5: Stochastic Inlet Boundary Control for Training')
        elif self.cont_scenario == 6:
            print('Case 6: Stochastic Outlet Boundary Control for Validation // Lighter Traffic')
        elif self.cont_scenario == 7:
            print('Case 7: Stochastic Inlet Boundary Control for Validation // Denser Traffic')
        else:
            raise ValueError('Case is not chosen. Please check the settings_file.py')

        if self.cont_scenario == 1 or self.cont_scenario == 2 or self.cont_scenario == 3:  
            print('Determinstic Env.')
            self.vs = v_steady
            self.vs_desired = v_desired
            self.rs = ro_steady
            self.rs_desired = ro_desired
            self.qs = v_steady * ro_steady
            self.qs_desired = v_desired * ro_desired
            self.ps = self.vm/self.rm * self.qs/self.vs
        
        x = np.arange(0,self.X+self.dx,self.dx)
        self.M = len(x)
        self.t = 0
        self.discrete = True  # Set to True or False based on your requirement

        self.qs = self.qs
        self.qs_input = np.linspace(self.qs/2,2*self.qs,40)

        self.r = np.zeros([self.M,1])
        self.y = np.zeros([self.M,1])


        #Initial condition of the PDE
        print('Initial condition of rs: {}'.format(self.rs))

        self.r = self.rs * np.transpose(np.sin(3 * x / self.L * np.pi ) * 0.1 + np.ones([1,self.M]))
        self.y = self.qs * np.ones([self.M,1]) - self.vm * self.r + self.vm / self.rm * (self.r)**(2)
        self.v = self.y/self.r + Veq(self.vm, self.rm, self.r)
        
        self.info = dict()
        self.info['V'] = self.v

	    # Observation space changes depending on sensing

        self.observation_space = spaces.Box(
            np.full(self.nx, -self.max_state_value, dtype="float32"),
            np.full(self.nx, self.max_state_value, dtype="float32"),
        )

        # match self.sensing_loc:
        #     case "full":
        #         self.observation_space = spaces.Box(
        #             np.full(self.nx, -self.max_state_value, dtype="float32"),
        #             np.full(self.nx, self.max_state_value, dtype="float32"),
        #         )
        #     case "collocated" | "opposite":
        #         self.observation_space = spaces.Box(
        #             np.full(1, -self.max_state_value, dtype="float32"),
        #             np.full(1, self.max_state_value, dtype="float32"),
        #         )
        #     case _:
        #         raise Exception(
        #             "Invalid sensing_loc parameter. Please use 'full', 'collocated', or 'opposite'. See documentation for details."
        #         )

        # Setup configurations for control and sensing. Messy, but done once, explicitly before runtime to setup return and control functions
        # There is a trick here where noise is a function call itself. Important that noise is a single argument function that returns a single argument

        # self.control_update = lambda control, state, dt: control
        # self.sensing_update = lambda state, dx, noise: noise(state)


        # match self.control_type:
        #     case "Neumann":
        #         self.control_update = lambda control, state, dx: control * dx + state
        #         match self.sensing_loc:
        #             # Neumann control u_x(1), full state measurement
        #             case "full":
        #                 self.sensing_update = lambda state, dx, noise: noise(state)
        #             # Neumann control u_x(1), Dirchilet sensing u(1)
        #             case "collocated":
        #                 self.sensing_update = lambda state, dx, noise: noise(state[-1])
        #             case "opposite":
        #                 match self.sensing_type:
        #                     # Neumann control u_x(1), Neumann sensing u_x(0)
        #                     case "Neumann":   
        #                         self.sensing_update = lambda state, dx, noise: noise(
        #                             (state[1] - state[0]) / dx
        #                         )
        #                     # Neumann control u_x(1), Dirchilet sensing u(0)
        #                     case "Dirchilet":
        #                         self.sensing_update = lambda state, dx, noise: noise(state[0])
        #                     case _:
        #                         raise Exception(
        #                             "Invalid sensing_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
        #                         )
        #             case _:
        #                 raise Exception(
        #                     "Invalid sensing_loc parameter. Please use 'full', 'collocated', or 'opposite'. See documentation for details."
        #                 )
        #     case "Dirchilet":
        #         self.control_update = lambda control, state, dt: control
        #         match self.sensing_loc:
        #             # Dichilet control u(1), full state measurement
        #             case "full":
        #                 self.sensing_update = lambda state, dx, noise: noise(state)
        #             # Dichilet control u(1), Neumann sensing u_x(1)
        #             case "collocated":
        #                 self.sensing_update = lambda state, dx, noise: noise(
        #                     (state[-1] - state[-2]) / dx
        #                 )
        #             case "opposite":
        #                 match self.sensing_type:
        #                     # Dichilet control u(1), Neumann sensing u_x(0)
        #                     case "Neumann":
        #                         self.sensing_update = lambda state, dx, noise: noise(
        #                             (state[1] - state[0]) / dx
        #                         )
        #                     # Dirchilet control u(1), Dirchilet sensing u(0)
        #                     case "Dirchilet":
        #                         self.sensing_update = lambda state, dx, noise: noise(
        #                             state[0]
        #                         )
        #                     case _:
        #                         raise Exception(
        #                             "Invalid sensing_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
        #                         )
        #     case _:
        #         raise Exception(
        #             "Invalid control_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
        #         )

    @property
    def observation_space(self):
        return spaces.Box(low=-10, high=10, shape=(2 * self.M,), dtype=np.float32)
        # return spaces.Box(low=-2, high=2*qm, shape=(2 * M,), dtype=np.float32)

    @property
    def action_space(self):
        if self.discrete:
            return spaces.Discrete(20)
        else:
            # Specify the input shape.
            if self.cont_scenario == 1 or self.cont_scenario == 2 or self.cont_scenario == 4 or self.cont_scenario == 5 or self.cont_scenario == 6 or self.cont_scenario == 7:
                return_box = spaces.Box(dtype=np.float32, low=self.qs * 0.8, high=1.2 * self.qs, shape=(1,))
            elif self.cont_scenario == 3:
                return_box = spaces.Box(dtype=np.float32, low=self.qs * 0.8, high=1.2 * self.qs, shape=(2,))
            
        return return_box

    def step(self, action: float):
        """
        step

        Moves the PDE with control action forward ``control_sample_rate*dt`` steps.

        :param control: The control input to apply to the PDE at the boundary.
        """
        Nx = self.nx
        dx = self.dx
        dt = self.dt

        self.t += dt

        if self.discrete:
            qs_input = self.qs_input[action]
        else:
            qs_input = action
            if self.cont_scenario == 1 or self.cont_scenario == 2 or self.cont_scenario == 4 or self.cont_scenario == 5 or self.cont_scenario == 6 or self.cont_scenario == 7:
                qs_input = np.clip(qs_input, a_min=self.action_space.low, a_max=self.action_space.high)[0]
            else:
                qs_input = np.clip(qs_input, a_min=self.action_space.low, a_max=self.action_space.high)
                q_inlet_input = qs_input[0]
                q_outlet_input = qs_input[1]


        if self.cont_scenario == 1 or self.cont_scenario == 4 or self.cont_scenario == 6:
			# Fixed inlet boundary input
            self.q_inlet = self.qs

        elif self.cont_scenario == 2 or self.cont_scenario == 5 or self.cont_scenario == 7:
			# Control inlet boundary input (single-input)
            self.q_inlet = qs_input

        elif self.cont_scenario == 3:
            # Control inlet boundary input (Multi-input)
            self.q_inlet = q_inlet_input

        # ------------------------------------------------------------------

        # Boundary conditions
        self.r[0] = self.r[1]
        self.y[0] = self.q_inlet - self.r[0] * Veq(self.vm, self.rm, self.r[0])

        # Ghost condition
        # M-1 means boundary
        self.r[self.M-1] = self.r[self.M-2]

        # PDE control part.------------------------------------------------------------------
        # Outlet

        if self.cont_scenario == 1 or self.cont_scenario == 4 or self.cont_scenario == 6:
            # Control outlet boundary input
            self.y[self.M-1] = qs_input - self.r[self.M-1]* Veq(self.vm, self.rm, self.r[self.M-1])
        
        elif self.cont_scenario == 2  or self.cont_scenario == 5 or self.cont_scenario == 7:
            # Fixed outlet boundary input
            self.y[self.M-1] = self.qs - self.r[self.M-1]* Veq(self.vm, self.rm, self.r[self.M-1])
        
        elif self.cont_scenario == 3:
            # Control outlet boundary input (Multi-input)
            self.y[self.M-1] = q_outlet_input - self.r[self.M-1]* Veq(self.vm, self.rm, self.r[self.M-1])
        
        
        # ------------------------------------------------------------------

        for j in range(1,self.M-1) :

            r_pmid = 1/2 * (self.r[j+1] + self.r[j]) - dt/(2 * dx) * ( F_r(self.vm, self.rm, self.r[j+1], self.y[j+1]) - F_r(self.vm, self.rm, self.r[j], self.y[j]) )

            y_pmid = 1/2 * (self.y[j+1] + self.y[j]) - dt/(2 * dx) * ( F_y(self.vm, self.rm, self.r[j+1], self.y[j+1]) - F_y(self.vm, self.rm, self.r[j], self.y[j])) - 1/4 * dt / self.tau * (self.y[j+1]+self.y[j])

            r_mmid = 1/2 * (self.r[j-1] + self.r[j]) - dt/(2 * dx) * ( F_r(self.vm, self.rm, self.r[j], self.y[j]) - F_r(self.vm, self.rm, self.r[j-1], self.y[j-1]))

            y_mmid = 1/2 * (self.y[j-1] + self.y[j]) - dt/(2 * dx) * ( F_y(self.vm, self.rm, self.r[j], self.y[j]) - F_y(self.vm, self.rm, self.r[j-1], self.y[j-1])) - 1/4 * dt / self.tau * (self.y[j-1]+self.y[j])

            self.r[j] = self.r[j] - dt/dx * (F_r(self.vm, self.rm, r_pmid, y_pmid) - F_r(self.vm, self.rm, r_mmid, y_mmid))
            self.y[j] = self.y[j] - dt/dx * (F_y(self.vm, self.rm, r_pmid, y_pmid) - F_y(self.vm, self.rm, r_mmid, y_mmid)) - 1/2 * dt/self.tau * (y_pmid + y_mmid)

        # Calculate Velocity
        self.v = self.y/self.r + Veq(self.vm, self.rm, self.r)

        # Reward
        v_desired = self.vs_desired  # vs
        r_desired = self.rs_desired  # rs
        reward = -(np.linalg.norm(self.v - v_desired, ord=None) / (v_desired) + np.linalg.norm(self.r - r_desired, ord=None) / (r_desired))

        # Done mask
        is_done = False
        if all(self.r - r_desired == 0) and all(self.v - v_desired == 0):
            is_done = True

        if self.t >= self.T / self.dt:
            print("Time over..")
            is_done = True

        # Return
        return np.reshape(np.concatenate(((self.r - self.rs_desired) / self.rs_desired, (self.v - self.vs_desired) / self.vs_desired)), -1), reward, is_done, self.info

    def terminate(self):
        """
        terminate

        Determines whether the episode should end if the ``T`` timesteps are reached
        """
        if self.time_index >= self.nt - 1:
            return True
        else:
            return False

    def truncate(self):
        """
        truncate 

        Determines whether to truncate the episode based on the PDE state size and the vairable ``limit_pde_state_size`` given in the PDE environment intialization.
        """
        if (
            self.limit_pde_state_size
            and np.linalg.norm(self.u[self.time_index], 2)  >= self.max_state_value
        ):
            return True
        else:
            return False
         
    # def reset_init_condition_func(self, nx: int):
    #     """
    #     reset_init_condition_func

    #     Sets the initial condition of the PDE at the start of each episode. Must be passed in as a function during environment initialization.

    #     :param nx: The number of spatial points in the PDE simulation.
    #     """
    #     v_i = np.linspace(0, self.nx, 100)
    #     v_i = -0.1 * np.sin((3 * np.pi * v_i) / self.nx) * self.v_steady + self.v_steady

    #     ro_i = np.linspace(0, self.nx, 100)
    #     ro_i = 0.1 * np.sin((3 * np.pi * ro_i) / self.nx) * self.ro_steady + self.ro_steady
        
    #     return v_i, ro_i

    # Resets the system state
    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """
        reset 

        :param seed: Allows a seed for initialization of the envioronment to be set for RL algorithms.
        :param options: Allows a set of options for the initialization of the environment to be set for RL algorithms.

        Resets the PDE at the start of each environment according to the parameters given during the PDE environment intialization

        """
        return np.reshape(np.concatenate(((self.r-self.rs_desired)/self.rs_desired, (self.v-self.vs_desired)/self.vs_desired)), -1)
        # try:
        #     init_condition = self.reset_init_condition_func(self.nx)
        #     beta = self.reset_recirculation_func(self.nx)
        # except:
        #     raise Exception(
        #         "Please pass both an initial condition and a recirculation function in the parameters dictionary. See documentation for more details"
        #         )
        # self.u_v = np.zeros(
        #     (self.nt, self.nx), dtype=np.float32
        # )
        # self.u_ro = np.zeros(
        #     (self.nt, self.nx), dtype=np.float32
        # )
        # self.u[0] = init_condition
        # self.time_index = 0
        # self.beta = beta
        # return (
        #     self.sensing_update(
        #         self.u[self.time_index], 
        #         self.dx,
        #         self.sensing_noise_func,
        #     ),
        #     {},
        # )

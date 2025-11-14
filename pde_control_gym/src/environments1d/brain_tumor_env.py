import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional

from pde_control_gym.src.environments1d.base_env_1d import PDEEnv1D

class BrainTumor1D(PDEEnv1D):
    r"""
    Brain Tumor 1D PDE

    This class implements the 1D Brain Tumor DPR PDE and inherits from the class :class:`PDEEnv1D`. Thus, for a full list of of arguments, first see the class :class:`PDEEnv1D` in conjunction with the arguments presented here

    This implementation was inspired by prior work:

      - Rockne et al. (2009): `A mathematical model for brain tumor response to radiation therapy <https://pubmed.ncbi.nlm.nih.gov/18815786>`_
      - Rockne et al. (2010): `Predicting efficacy of radiotherapy in individual glioblastoma patients in vivo: a mathematical modeling approach <https://pubmed.ncbi.nlm.nih.gov/20484781>`_
      - Hathout et al. (2016): `Modeling the efficacy of the extent of surgical resection in the setting of radiation therapy for glioblastoma <https://pmc.ncbi.nlm.nih.gov/articles/PMC4982585>`_

    :param t1_detection_threshold: Fraction of carrying capacity k used to define the T1 region
    :param t2_detection_threshold: Fraction of carrying capacity k used to define the T2 region
    :param D: Diffusion coefficient in units :math:`(mm^2/day)`
    :param rho: Proliferation rate in units :math:`(1/day)`
    :param alpha: Radio-sensitivity parameter in units :math:`(Gy^-1)`
    :param alpha_beta_ratio: Radio-biologic parameter measuring fractionation sensitivity in units :math:`(Gy)`
    :param k: Carrying capacity in units :math:`(cells/mm^3)`
    :param t1_detection_radius: Radius of tumor at detection in units :math:`(mm)`
    :param t1_death_radius: Radius of tumor at patient death :math:`(mm)`
    :param reset_init_condition_func: Function that resets initial PDE condition :math:`u(x, 0)`
    :param total_dosage: Total radiation dosage in units :math:`(Gy)`
    :param verbose: Toggles print statements
    """
    def __init__(self,
                 t1_detection_threshold: float = 0.8,
                 t2_detection_threshold: float = 0.16,
                 D: float = 0.2,
                 rho: float = 0.03,
                 alpha: float = 0.04,
                 alpha_beta_ratio: int = 10,
                 k: float = 1e5,
                 t1_detection_radius: int = 15,
                 t1_death_radius: int = 35,
                 reset_init_condition_func = None,
                 total_dosage = None,
                 verbose = True,
                 **kwargs     
    ):
        super().__init__(**kwargs)
        self.verbose = verbose

        self.nx = int(round(self.X/self.dx)+1) # total grid points = last observable index + 1
        self.u = np.zeros((self.nt, self.nx))
        self.dosage_vs_time = np.zeros(self.nt)
        self.xScale = np.linspace(0, self.X, self.nx) # index -> space mapping
        if self.verbose:
            print(f"nx: {self.nx}, nt: {self.nt}")
            print(f"u.shape: {self.u.shape}")

        # Override base env action_space to [0, 1]
        self.action_space = spaces.Box(
            np.full(1, 0, dtype="float32"),
            np.full(1, 1, dtype="float32")
        )

        # Define observation_space
        self.observation_space = spaces.Box(
            np.full(self.nx, 0, dtype="float64"),
            np.full(self.nx, k, dtype="float64"),
            dtype=np.float64
        )

        # Constant simulation parameters
        self.t1_detection_threshold = t1_detection_threshold
        self.t2_detection_threshold = t2_detection_threshold
        self.reset_init_condition_func = reset_init_condition_func
        self.D = D
        self.rho = rho
        self.alpha = alpha
        self.alphaBetaRatio = alpha_beta_ratio
        self.k = k
        self.t1_detection_radius = t1_detection_radius
        self.t1_death_radius = t1_death_radius

        # Action controlled parameters
        self.total_dosage = float(total_dosage) # Constant
        self.remaining_dosage = float(total_dosage) # Changed after each action

        # Simulation recorded metrics
        self.stage = "Growth"
        self.simulationDays = 0
        self.growthDays = 0
        self.therapyDays = 0
        self.postTherapyDays = 0
        self.firstTherapyDay = None
        self.firstPostTherapyDay = None
        self.cDeathDay = None
        self.t_benchmark = None

    
    def getTumorRadius(self, time_index, detectionRatio):
        """
        getTumorRadius

        Helper function determining T1/T2 tumor radius at given time_index

        :param time_index: Time step to calculate radius for
        :param detectionRatio: Proportion of carrying capacity to set as detection threshold
        """
        densities = self.u[time_index]
        threshold = detectionRatio * self.k
        binaryMask = densities >= threshold

        if not binaryMask.any(): # tumor invisible to simulated MRI scan
            return None
        
        rightmostIdx = binaryMask.size - 1 - np.argmax(binaryMask[::-1])
        return rightmostIdx * self.dx

    def step(self, control: float):
        """
        step

        Moves the PDE with control action forward dt steps

        :param control: Control input. Proportion of total_dosage to apply
        """

        if self.verbose:
            print(f"\tEnvironment: Call step(). Perform dimensionalized finite differencing for time_index={self.time_index+1}")

        if (self.time_index < self.nt-1):
            self.time_index += 1
            currU = self.u[self.time_index-1]
            if not hasattr(self, "_next_buffer"):
                self._next_buffer = np.empty_like(currU)
            nextU = self._update_fd(currU, R=None) # default: no radiation

            # stage actions and transitions
            reward = 0.0
            treatmentRadius = None
            applied_dosage = 0.0

            if self.stage == "Growth":
                self.u[self.time_index] = nextU
                T1, T2 = self._log_radii()
                self.growthDays = self.time_index
                if (T1 is not None and T1 >= self.t1_detection_radius):
                    self.firstTherapyDay = self.time_index + 1
                    if self.verbose:
                        print(f"\n\tGrowth break when time_index={self.growthDays}. Therapy stage begins when time_index={self.firstTherapyDay}. T1TumorRadius={T1:.2f}\n")
                    self.stage = "Therapy"

            elif self.stage == "Therapy":
                control = float(np.asarray(control).squeeze())
                applied_dosage = min(control * self.total_dosage, self.remaining_dosage)
                self.dosage_vs_time[self.time_index] = applied_dosage
                self.remaining_dosage -= applied_dosage
                if self.verbose:
                    print(f"\tAction = {control}. Remaining dosage = {self.remaining_dosage}. Dosage for current timestep = {applied_dosage}")

                R, treatmentRadius = self._compute_radiation_field(applied_dosage)
                nextU = self._update_fd(currU, R=R)
                self.u[self.time_index] = nextU

                T1, T2 = self._log_radii()
                # end therapy when remaining dosage exhausted or almost exhausted
                if (self.remaining_dosage < 0.1):
                    self.therapyDays = self.time_index - self.growthDays
                    self.firstPostTherapyDay = self.time_index + 1
                    if self.verbose:
                        print(f"\n\tTherapy completed. Begin Post-Therapy starting time_index={self.firstPostTherapyDay}\n")
                    self.stage = "Post-Therapy"
                
                # reward during therapy
                terminate = self.terminate()
                truncate = self.truncate()
                reward = self.reward_class.reward(
                    uVec=self.u, time_index=self.time_index, terminate=terminate, truncate=truncate, 
                    action=control, verbose=self.verbose, t_benchmark=self.t_benchmark, 
                    tumor_radius=T1, treatment_radius=treatmentRadius, applied_dosage=applied_dosage, 
                    total_dosage=self.total_dosage
                )
                return (
                  self.u[self.time_index],
                  reward, 
                  terminate,
                  truncate,
                  {"stage": self.stage},
                )

            elif self.stage == "Post-Therapy":
                self.u[self.time_index] = nextU
                T1, T2 = self._log_radii()
            
            # handle termination or truncation
            terminate = self.terminate()
            truncate  = self.truncate()

            # post-therapy may return a reward on termination or truncation
            if self.stage == "Post-Therapy" and (terminate or truncate):
                reward = self.reward_class.reward(
                    uVec=self.u, time_index=self.time_index, terminate=terminate, truncate=truncate, 
                    action=control, verbose=self.verbose, t_benchmark = self.t_benchmark
                )
            else:
                reward = 0.0

            return (
                self.u[self.time_index],
                reward, 
                terminate,
                truncate,
                {"stage": self.stage},
            )
                

    def _update_fd(self, currU, R=None):
        """
        _update_fd

        Helper function that performs finite differencing to calculate next state. Takes optional R parameter for radiation

        :param currU: current state array
        :param R: radiation field array
        """

        u = currU
        u_c = u[1:-1]
        u_l = u[:-2]
        u_r = u[2:]

        diffusion = self.D * ((u_r - 2.0 * u_c + u_l) / (self.dx ** 2))
        proliferation = self.rho * u_c * (1.0 - (u_c / self.k))
        radiation = 0.0 if R is None else (R[1:-1] * u_c * (1.0 - (u_c / self.k)))

        nextU = self._next_buffer
        nextU[1:-1] = u_c + self.dt * (diffusion + proliferation - radiation)
        nextU[0] = nextU[1]
        nextU[-1] = nextU[-2]
        nextU = np.clip(nextU, 0, self.k)
        return nextU
    
    def _compute_radiation_field(self, applied_dosage):
        """
        _compute_radiation_field

        Helper function that computes radiation field for current timestep given applied_dosage and current state's tumor size

        :param applied_dosage: dosage in Gy for current timestep
        """

        dArray = np.zeros_like(self.u[0])
        # use previous day's radius metrics
        T2TumorRadius = self.getTumorRadius(self.time_index - 1, self.t2_detection_threshold)
        treatmentRadius = 0.0 if T2TumorRadius is None else (T2TumorRadius + 25)
        dArray[self.xScale <= treatmentRadius] = applied_dosage

        BED = dArray + ((dArray ** 2) / (self.alphaBetaRatio))
        S = np.exp(-self.alpha * BED)
        R = 1.0 - S
        return R, treatmentRadius
    
    def _log_radii(self):
        """
        _log_radii

        Helper function that computes T1 and T2 tumor radii, and optionally logs information
        """

        T1TumorRadius = self.getTumorRadius(self.time_index, self.t1_detection_threshold)
        T2TumorRadius = self.getTumorRadius(self.time_index, self.t2_detection_threshold)
        if self.verbose:
            t1r = float('nan') if T1TumorRadius is None else T1TumorRadius
            t2r = float('nan') if T2TumorRadius is None else T2TumorRadius
            print(f"\t{self.stage:<15} {self.time_index:<5} {t1r:<15.2f} {t2r:<15.2f}\n")
        return T1TumorRadius, T2TumorRadius

        
    def terminate(self):
        """
        terminate

        Determines whether episode should end if the T timesteps are reached
        """
        if self.time_index < self.nt - 1:
            return False
        
        if self.stage == "Therapy":
            self.therapyDays = self.time_index - self.growthDays
            self.simulationDays = self.growthDays + self.therapyDays
        elif self.stage == "Post-Therapy":
            self.postTherapyDays = self.time_index - self.therapyDays - self.growthDays
            self.simulationDays = self.growthDays + self.therapyDays + self.postTherapyDays

        if self.verbose:
            print(f"\nTerminate: self.time_index is at or exceeds {self.nt - 1}\n")
            print(f"self.simulationDays {self.simulationDays}")
            print(f"self.growthDays {self.growthDays}")
            print(f"self.therapyDays {self.therapyDays}")
            print(f"self.postTherapyDays {self.postTherapyDays}")
            print(f"self.firstTherapyDay {self.firstTherapyDay}")
            print(f"self.firstPostTherapyDay {self.firstPostTherapyDay}")
            print(f"self.cDeathDay {self.cDeathDay}")

        return True
        
    def truncate(self):
        """
        truncate

        Truncates episode if patient death conditions reached
        """
        T1TumorRadius = self.getTumorRadius(self.time_index, self.t1_detection_threshold)
        lethal = (T1TumorRadius is not None) and (T1TumorRadius >= self.t1_death_radius)
        
        if not lethal:
            return False
        
        if self.cDeathDay is None:
            self.cDeathDay = self.time_index

            if self.stage == "Therapy":
                self.therapyDays = self.time_index - self.growthDays
                self.simulationDays = self.growthDays + self.therapyDays
            elif self.stage == "Post-Therapy":
                self.postTherapyDays = self.time_index - self.therapyDays - self.growthDays
                self.simulationDays = self.growthDays + self.therapyDays + self.postTherapyDays

            if self.verbose:
                print(f"\nTruncate: Tumor T1TumorRadius {T1TumorRadius} is at or exceeds self.t1_death_radius {self.t1_death_radius}mm\n")
                print(f"self.simulationDays {self.simulationDays}")
                print(f"self.growthDays {self.growthDays}")
                print(f"self.therapyDays {self.therapyDays}")
                print(f"self.postTherapyDays {self.postTherapyDays}")
                print(f"self.firstTherapyDay {self.firstTherapyDay}")
                print(f"self.firstPostTherapyDay {self.firstPostTherapyDay}")
                print(f"self.cDeathDay {self.cDeathDay}")
                
        return True
        
    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """
        reset

        Resets the PDE environment at start of each environment setting according to parameters given during PDE environment initialization

        :param seed: Allows a seed for initialization of the envioronment to be set for RL algorithms
        :param options: Allows a set of options for the initialization of the environment to be set for RL algorithms
        """
        try:
            init_condition = self.reset_init_condition_func(self.X, self.nx)
        except:
            raise Exception(
                "Please pass an initial condition function"
            )

        # Reset parameters
        self.time_index = 0
        self.u = np.zeros((self.nt, self.nx))
        self.dosage_vs_time = np.zeros(self.nt)
        self.u[0] = init_condition
        self.stage = "Growth"
        self.total_dosage = self.total_dosage
        self.remaining_dosage = self.total_dosage

        # Reset recorded metrics
        self.simulationDays = 0
        self.growthDays = 0
        self.therapyDays = 0
        self.postTherapyDays = 0
        self.firstTherapyDay = None
        self.firstPostTherapyDay = None
        self.cDeathDay = None

        return (
            self.u[0],
            {},
        )

class TherapyWrapper(gym.Wrapper):
    """
    Therapy Wrapper

    This class implements a custom wrapper inerhting from the class gym.Wrapper. The wrapper abstracte growth stage and post-therapy stage environment activity, exposing the environment only during the treatment stage when the RL acts

    :param weekends: Determines whether or not we take weekend breaks during treatment
    :param verbose: Toggles print statements
    """
    def __init__(self, env: BrainTumor1D, weekends=False, verbose=True):
        super().__init__(env)

        self.verbose = verbose
        self.weekends = weekends

        # tracking proportion of soft constrant violations
        self.treatment_calls = 0
        self.soft_constraint_violations = 0

        # tracking for weekends
        self.consecutive_treatment_days = 0
    
    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """
        reset

        Calls wrapped environment's reset method and runs entire growth stage
        """
        if self.verbose:
            print(f"Wrapper: Reset environment")
        self.consecutive_treatment_days = 0
        obs, info = self.env.reset()

        if self.verbose:
            print(f"Wrapper: Start Growth Stage")
        while self.env.unwrapped.stage == "Growth":
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                break
        if self.verbose:
            print(f"Wrapper: End Growth Stage\n")
        return obs, info
    
    def step(self, control: float):
        """
        step

        Runs 1 treatment step if during treatment stage. If during post-therapy stage, simulate until truncation or termination

        :param control: Control input. Proportion of total_dosage to apply
        """
        # Case 1: Internally simulate Post-Therapy until terminate or truncate
        if self.env.unwrapped.stage == "Post-Therapy":
            if self.verbose:
                print(f"Wrapper: Post-Therapy step()")
            terminated, truncated = False, False
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = self.env.step(0)

            if self.verbose:
                print(f"[Episode Reward] {reward}\n")
                print(f"Soft constraint violation rate: {(self.soft_constraint_violations / self.treatment_calls) * 100}%")
            return obs, reward, terminated, truncated, info

        # Case 2: Run single therapy step
        if self.verbose:
            print("Wrapper: Therapy step()")

        obs, reward, terminated, truncated, info = self.env.step(control)

        self.treatment_calls += 1
        if reward < 0.0:
            self.soft_constraint_violations += 1

        if self.weekends:
            if control > 0:
                self.consecutive_treatment_days += 1
            else:
                self.consecutive_treatment_days = 0

        if self.weekends and self.consecutive_treatment_days >= 5:
            self.consecutive_treatment_days = 0
            if self.verbose:
              print("Wrapper: Force weekend")
            for i in range(2):
                _ = self.env.step(0)
                if terminated or truncated:
                    return obs, reward, terminated, truncated, info
            
        if self.verbose:
            print(f"[Therapy Reward] {reward}\n")

        return obs, reward, terminated, truncated, info
    
    def benchmark(self):
        """
        benchmark

        Runs open-loop episode under the hood to set t_benchmark.
        RUN THIS METHOD BEFORE MODEL TRAINING AND RUNNING AN EPISODE
        """
        obs, info = self.env.reset()
        if self.verbose:
            print(f"Wrapper: Benchmark (episode run with no action and no reward):")
        
        terminated = False
        truncated = False

        while not (terminated or truncated):
            obs, _, terminated, truncated, info = self.env.step(0)

        # set t_benchmark and reset environment
        t_benchmark = self.env.unwrapped.simulationDays
        self.env.unwrapped.t_benchmark = t_benchmark
        if self.verbose:
            print(f"Set t_benchmark = {t_benchmark}\n\n\n")
        obs, info = self.env.reset()

        return t_benchmark


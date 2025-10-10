import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional

from pde_control_gym.src.environments1d.base_env_1d import PDEEnv1D

class BrainTumor1D(PDEEnv1D):
    def __init__(self,
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
        # Override base env 1d state size
        self.nt = int(round(self.T/self.dt)+1)
        self.nx = int(round(self.X/self.dx)+1)
        self.u = np.zeros((self.nt, self.nx))
        self.xScale = np.linspace(0, self.X, self.nx) # index -> space mapping
        if self.verbose:
            print(f"nx: {self.nx}, nt: {self.nt}")
            print(f"u.shape: {self.u.shape}")

        # Override base env action space to [0, 1]
        self.action_space = spaces.Box(
            np.full(1, 0, dtype="float32"),
            np.full(1, 1, dtype="float32")
        )

        # Simulation parameters
        self.T1_DETECTION_THRESHOLD = 0.8
        self.T2_DETECTION_THRESHOLD = 0.16
        self.reset_init_condition_func = reset_init_condition_func
        self.D = D
        self.rho = rho
        self.alpha = alpha
        self.alphaBetaRatio = alpha_beta_ratio
        self.k = k
        self.t1_detection_radius = t1_detection_radius
        self.t1_death_radius = t1_death_radius
        self.observation_space = spaces.Box(
            np.full(self.nx, 0, dtype="float64"),
            np.full(self.nx, k, dtype="float64"),
            dtype=np.float64
        )

        # Action controlled parameters
        self.total_dosage = total_dosage
        self.remaining_dosage = total_dosage #deducted after each action

        # Simulation metrics
        self.stage = "Growth"
        self.simulationDays = 0
        self.growthDays = 0
        self.therapyDays = 0
        self.postTherapyDays = 0
        self.firstTherapyDay = None
        self.firstPostTherapyDay = None
        #self.DOT = 0 #excludes weekends
        #self.DIT = 0 #includes weekends
        self.cDeathDay = None
        self.t_benchmark = None

    
    def getTumorRadius(self, time_index, detectionRatio):
        densities = self.u[time_index]
        threshold = detectionRatio * self.k
        binaryMask = (densities >= threshold).astype(int)
        rightmostIdx = np.where(binaryMask == 1)[0].max() if np.any(binaryMask == 1) else None
        if rightmostIdx is not None:
          xScale = np.linspace(0, self.X, self.nx)
          tumorRadius = xScale[rightmostIdx]
          return tumorRadius
        return None
    
    # Diffusion-Proliferation-Radiation with Action and Reward
    def step(self, control: float):
        if self.verbose:
            print(f"\tEnvironment: Call step(). Perform dimensionalized finite differencing for time_index={self.time_index+1}")

        if (self.time_index < self.nt-1):
            self.time_index += 1
            currU = self.u[self.time_index-1]
            nextU = np.zeros(self.nx)

            # growth stage
            if (self.stage == "Growth"):
                # perform finite differencing
                nextU[1:self.nx-1] = ( currU[1:self.nx-1] +
                                      self.dt * (self.D * (((currU[2:self.nx] - 2 * currU[1:self.nx-1] + currU[0:self.nx-2]) / (self.dx ** 2)))
                                                  + self.rho * currU[1:self.nx-1] * (np.ones_like(currU)[1:self.nx-1] - (currU[1:self.nx-1] / self.k))
                                                  # no radiation term for free growth
                                                )
                                      )
                nextU[0] = nextU[1]
                nextU[self.nx-1] = nextU[self.nx-2]
                nextU = np.clip(nextU, 0, self.k)
                self.u[self.time_index] = nextU

                # log after current time_index calculated
                T1TumorRadius = self.getTumorRadius(self.time_index, 0.8)
                T2TumorRadius = self.getTumorRadius(self.time_index, 0.16)
                if self.verbose:
                    print(f"\t{self.stage:<15} {self.time_index:<5} {f'{T1TumorRadius:.2f}' if T1TumorRadius is not None else 'None':<15} {f'{T2TumorRadius:.2f}' if T2TumorRadius is not None else 'None':<15}\n")

                # check if detection radius reached
                T1TumorRadius = self.getTumorRadius(self.time_index, self.T1_DETECTION_THRESHOLD)
                self.growthDays = self.time_index
                if (T1TumorRadius is not None and T1TumorRadius >= self.t1_detection_radius):
                    self.firstTherapyDay = self.time_index + 1
                    if self.verbose:
                        print(f"\n\tGrowth break when time_index={self.growthDays}. Therapy stage begins when time_index={self.firstTherapyDay}. T1TumorRadius={T1TumorRadius:.2f}\n")
                    self.stage = "Therapy"
                
                # check termination conditions
                terminate = self.terminate()
                truncate = self.truncate()
                return (
                  self.u[self.time_index],
                  0, #no reward during growth stage
                  terminate,
                  truncate,
                  {"stage": self.stage},
                )
            
            # therapy stage
            if (self.stage == "Therapy"):
                if self.verbose:
                    print(f"\tTherapy: Perform dimensionalized finite differencing for time_index={self.time_index}")

                break_therapy = False
                applied_dosage = control * self.total_dosage
                if (applied_dosage >= self.remaining_dosage):
                    break_therapy = True
                    applied_dosage = self.remaining_dosage
                if self.verbose:
                    print(f"\tAction = {control}. Remaining dosage = {self.remaining_dosage}. Dosage for current timestep = {applied_dosage}")
                self.remaining_dosage -= applied_dosage

                dArray = np.zeros_like(self.u[0])
                # use previous day's radius metrics
                T2TumorRadius = self.getTumorRadius(self.time_index - 1, self.T2_DETECTION_THRESHOLD)
                treatmentRadius = T2TumorRadius + 25
                dArray[self.xScale <= treatmentRadius] = applied_dosage
                R = np.ones_like(self.u[0])
                BED = (dArray + ((dArray ** 2) / (self.alphaBetaRatio)) )
                S = np.exp(-self.alpha * BED)
                R = R - S

                # perform finite differencing with radiation
                nextU[1:self.nx-1] = ( currU[1:self.nx-1] +
                                      self.dt * (self.D * (((currU[2:self.nx] - 2 * currU[1:self.nx-1] + currU[0:self.nx-2]) / (self.dx ** 2)))
                                                  + self.rho * currU[1:self.nx-1] * (np.ones_like(currU)[1:self.nx-1] - (currU[1:self.nx-1] / self.k))
                                                  - R[1:self.nx-1] * currU[1:self.nx-1] * (np.ones_like(currU)[1:self.nx-1] - (currU[1:self.nx-1] / self.k))
                                                )
                                      )
                nextU[0] = nextU[1]
                nextU[self.nx-1] = nextU[self.nx-2]
                nextU = np.clip(nextU, 0, self.k)
                self.u[self.time_index] = nextU

                # log after current time_index calculated
                T1TumorRadius = self.getTumorRadius(self.time_index, 0.8)
                T2TumorRadius = self.getTumorRadius(self.time_index, 0.16)
                if self.verbose:
                    print(f"\t{self.stage:<15} {self.time_index:<5} {f'{T1TumorRadius:.2f}' if T1TumorRadius is not None else 'None':<15} {f'{T2TumorRadius:.2f}' if T2TumorRadius is not None else 'None':<15}")

                # check if therapy completed: if we clamp applied_dosage
                if break_therapy or self.remaining_dosage < 0.1:
                    self.therapyDays = self.time_index - self.growthDays
                    self.firstPostTherapyDay = self.time_index + 1
                    if self.verbose:
                        print(f"\n\tTherapy completed. Begin Post-Therapy starting time_index={self.firstPostTherapyDay}\n")
                    self.stage = "Post-Therapy"
                    

                # check termination conditions
                terminate = self.terminate()
                truncate = self.truncate()
                return (
                  self.u[self.time_index],
                  self.reward_class.reward(uVec=self.u, time_index=self.time_index, terminate=terminate, truncate=truncate, action=control, verbose=self.verbose, t_benchmark = self.t_benchmark, tumor_radius=T1TumorRadius, treatment_radius=treatmentRadius, applied_dosage=applied_dosage), 
                  terminate,
                  truncate,
                  {"stage": self.stage},
                )
            
            # post-therapy stage
            if (self.stage == "Post-Therapy"):
                if self.verbose:
                    print(f"\tPost-Therapy: Perform dimensionalized finite differencing for time_index={self.time_index}")

                # perform finite differencing
                nextU[1:self.nx-1] = ( currU[1:self.nx-1] +
                                      self.dt * (self.D * (((currU[2:self.nx] - 2 * currU[1:self.nx-1] + currU[0:self.nx-2]) / (self.dx ** 2)))
                                                  + self.rho * currU[1:self.nx-1] * (np.ones_like(currU)[1:self.nx-1] - (currU[1:self.nx-1] / self.k))
                                                  # no radiation term for free growth
                                                )
                                      )
                nextU[0] = nextU[1]
                nextU[self.nx-1] = nextU[self.nx-2]
                nextU = np.clip(nextU, 0, self.k)
                self.u[self.time_index] = nextU

                # log after current time_index calculated
                T1TumorRadius = self.getTumorRadius(self.time_index, 0.8)
                T2TumorRadius = self.getTumorRadius(self.time_index, 0.16)
                if self.verbose:
                    print(f"{self.stage:<15} {self.time_index:<5} {f'{T1TumorRadius:.2f}' if T1TumorRadius is not None else 'None':<15} {f'{T2TumorRadius:.2f}' if T2TumorRadius is not None else 'None':<15}\n")

                # check termination conditions
                terminate = self.terminate()
                truncate = self.truncate()

                if (terminate or truncate):
                    return (
                      self.u[self.time_index],
                      self.reward_class.reward(uVec=self.u, time_index=self.time_index, terminate=terminate, truncate=truncate, action=control, verbose=self.verbose, t_benchmark = self.t_benchmark),
                      terminate,
                      truncate,
                      {"stage": self.stage},
                    )
                else:
                    return (
                      self.u[self.time_index],
                      0, #no reward if not terminate or truncate during post-therapy
                      terminate,
                      truncate,
                      {"stage": self.stage},
                    )


        
    def terminate(self):
        if self.time_index >= self.nt - 1:
            if (self.stage == "Therapy"):
                self.therapyDays = self.time_index - self.growthDays
                self.simulationDays = self.growthDays + self.therapyDays
            if (self.stage == "Post-Therapy"):
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
        else:
            return False
        
    def truncate(self):
        T1TumorRadius = self.getTumorRadius(self.time_index, self.T1_DETECTION_THRESHOLD)

        # alert when tumor reaches deadly radius
        if (T1TumorRadius is not None and T1TumorRadius >= self.t1_death_radius):
            if (self.cDeathDay is None):
                self.cDeathDay = self.time_index
                if (self.stage == "Therapy"):
                    self.therapyDays = self.time_index - self.growthDays
                    self.simulationDays = self.growthDays + self.therapyDays
                if (self.stage == "Post-Therapy"):
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
        return False
        
    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        try:
            init_condition = self.reset_init_condition_func(self.X, self.nx)
        except:
            raise Exception(
                "Please pass an initial condition function"
            )

        self.time_index = 0
        self.u = np.zeros((self.nt, self.nx))
        self.u[0] = init_condition
        self.stage = "Growth"

        # reset metrics
        self.simulationDays = 0
        self.growthDays = 0
        self.therapyDays = 0
        self.postTherapyDays = 0
        self.firstTherapyDay = None
        self.firstPostTherapyDay = None
        #self.DOT = 0 #excludes weekends
        #self.DIT = 0 #includes weekends
        self.cDeathDay = None
        self.total_dosage = self.total_dosage
        self.remaining_dosage = self.total_dosage #deducted after each action

        return (
            self.u[0],
            {},
        )

class TherapyWrapper(gym.Wrapper):
    def __init__(self, env: BrainTumor1D, verbose=True):
        super().__init__(env)
        self.verbose = verbose
    
    # Run growth stage of simulation until start of therapy
    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        if self.verbose:
            print(f"Wrapper: Reset environment")
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
    
    # Returns therapy reward if normal therapy step. Otherwise return episode reward
    def step(self, control: float):
        if self.verbose:
            print(f"Wrapper: Therapy step()")
        obs, reward, terminated, truncated, info = self.env.step(control)

        # Internally simulate Post-Therapy until terminate or truncate
        if not (terminated or truncated) and self.env.unwrapped.stage == "Post-Therapy":
            if self.verbose:
                print(f"Wrapper: Post-Therapy step()")
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = self.env.step(0)
        
        if (terminated or truncated):
            if self.verbose:
                print(f"[Episode Reward] {reward}\n")
        else:
            if self.verbose:
                print(f"[Therapy Reward] {reward}\n")

        return obs, reward, terminated, truncated, info
    
    def benchmark(self):
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
            print(f"Set t_benchmark = {t_benchmark}")
        obs, info = self.env.reset()

        return t_benchmark


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
                 radiation_schedule = None,
                 **kwargs     
    ):
        super().__init__(**kwargs)
        self.nt = int(round(self.T/self.dt)+1)
        self.nx = int(round(self.X/self.dx)+1)
        self.xScale = np.linspace(0, self.X, self.nx) # index -> space mapping
        print(f"nx: {self.nx}, nt: {self.nt}")

        self.reset_init_condition_func = reset_init_condition_func
        self.radiation_schedule = radiation_schedule

        self.T1_DETECTION_THRESHOLD = 0.8
        self.T2_DETECTION_THRESHOLD = 0.16
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

        self.stage = "Growth"
        self.simulationDays = 0
        self.growthDays = 0
        self.therapyDays = 0
        self.postTherapyDays = 0

        self.firstTherapyDay = None
        self.firstPostTherapyDay = None
        self.DOT = 0 #excludes weekends
        self.DIT = 0 #includes weekends

        self.cDeathDay = None

    
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

    # Diffusion-Proliferation only
    '''
    def step(self, control: float):
        print(f"Call step(). Perform dimensionalized finite differencing for time_index={self.time_index+1}")

        if (self.time_index < self.nt-1):
            self.time_index += 1
            currU = self.u[self.time_index-1]
            nextU = np.zeros(self.nx)

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

            # update state
            self.u[self.time_index] = nextU

        # check termination conditions
        terminate = self.terminate()
        return (
            self.u[self.time_index],
            self.reward_class.reward(),
            terminate,
            False,
            {},
        )
    '''

    # Diffusion-Proliferation-Radiation
    def step(self, control: float):

        if (self.time_index < self.nt-1):
            self.time_index += 1

            currU = self.u[self.time_index-1]
            nextU = np.zeros(self.nx)

            # growth stage
            if (self.stage == "Growth"):
                print(f"Growth: Perform dimensionalized finite differencing for time_index={self.time_index}")
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

                # check if detection radius reached
                T1TumorRadius = self.getTumorRadius(self.time_index, self.T1_DETECTION_THRESHOLD)
                if (T1TumorRadius is not None and T1TumorRadius >= self.t1_detection_radius):
                    self.growthDays = self.time_index
                    self.firstTherapyDay = self.time_index + 1
                    print(f"\nGrowth break when time_index={self.growthDays}. Therapy stage begins when time_index={self.firstTherapyDay}. T1TumorRadius={T1TumorRadius:.2f}\n")
                    self.stage = "Therapy"

                # check termination conditions
                terminate = self.terminate()
                truncate = self.truncate()
                return (
                  self.u[self.time_index],
                  self.reward_class.reward(),
                  terminate,
                  truncate,
                  {},
                )

            # therapy stage
            if (self.stage == "Therapy"):
                print(f"Therapy: Perform dimensionalized finite differencing for time_index={self.time_index}. DIT={self.DIT} DOT={self.DOT}")
                # construct dArray
                isRadiationDay = self.radiation_schedule['days_on_off'][self.DIT % 7]
                dArray = np.zeros_like(self.u[0])
                # use previous day's radius metrics
                T1TumorRadius = self.getTumorRadius(self.time_index - 1, self.T1_DETECTION_THRESHOLD)
                T2TumorRadius = self.getTumorRadius(self.time_index - 1, self.T2_DETECTION_THRESHOLD)

                # weekday -> therapy
                if isRadiationDay:
                    if self.DOT < self.radiation_schedule['t2_therapy_days']:
                        treatmentRadius = T2TumorRadius + 25 if T2TumorRadius is not None else -1
                    else:
                        treatmentRadius = T1TumorRadius + 20 if T1TumorRadius is not None else -1
                    dArray[self.xScale <= treatmentRadius] = self.radiation_schedule['dose_gy']
                    R = np.ones_like(self.u[0])
                    BED = (dArray + ((dArray ** 2) / (self.alphaBetaRatio)) )
                    S = np.exp(-self.alpha * BED)
                    R = R - S
                    self.DOT += 1

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
                # weekend -> no therapy
                else:
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
                
                self.DIT += 1
                # check if therapy completed
                if (self.DOT == self.radiation_schedule['total_therapy_days']):
                    self.therapyDays = self.time_index - self.growthDays
                    self.firstPostTherapyDay = self.time_index + 1
                    print(f"\nTherapy completed. DIT={self.DIT} DOT={self.DOT}. Begin Post-Therapy starting time_index={self.firstPostTherapyDay}\n")
                    self.stage = "Post-Therapy"

                # check termination conditions
                terminate = self.terminate()
                truncate = self.truncate()
                return (
                  self.u[self.time_index],
                  self.reward_class.reward(),
                  terminate,
                  truncate,
                  {},
                )
            
            if (self.stage == "Post-Therapy"):
                print(f"Post-Therapy: Perform dimensionalized finite differencing for time_index={self.time_index}")
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

                # check termination conditions
                terminate = self.terminate()
                truncate = self.truncate()
                return (
                  self.u[self.time_index],
                  self.reward_class.reward(),
                  terminate,
                  truncate,
                  {},
                )
                  

                        



        
    def terminate(self):
        if self.time_index >= self.nt - 1:
            self.postTherapyDays = self.time_index - self.therapyDays - self.growthDays
            self.simulationDays = self.growthDays + self.therapyDays + self.postTherapyDays
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
                print(f"\nTumor reaches T1TumorRadius = {T1TumorRadius} = {self.t1_death_radius} = self.t1_death_radius\n")
                self.cDeathDay = self.time_index
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

        return (
            self.u[0],
            {},
        )

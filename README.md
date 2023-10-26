# PDECG: Partial-Differential-Equation Control Gym

Link Website | Link Paper 

## About this repository

This gym provides implementations of various PDEs including hyperbolic PDEs(Burger's Equation) and parabolic PDEs (Reaction-Diffusion/Heat Equations) that can be easily simulated for control algorithms. 

## Installation

## Running Examples

### Heat Equation

### Burger's Equation

### Advection-Diffusion Equation

## Documentation
All the documentation is available at {Insert link to documentation}

### Current Implementations
- Currently PDEC-G contains implementations for controlling general form 1-D parabolic and hyperbolic PDEs. Particularly, both of these are implemented using traditional finite difference algorithms that require a small time-step. More details about each specific environment are given below and can be found on the website at {Insert Link to documentation}.

## General PDE Arguments 
### PDE Boundary Conditions:	
PDECG supports both Neumann and Dirchilet boundary conditions to specify the non-control boundary. To specify the boundary conditions, one must pass in a functional argument as well as the location and type of boundary condition into the PDECG initialization call. Examples of valid boundary conditions are given below. Please note that boundary condition functions should ideally be continuous to ensure the well-posed formulation of the PDE
### Sensing:
PDECG supports both Neumann and Dirchilet boundary sensing at both endpoints as well as full state sensing. To specify the sensing, provide the {insert argument} when creating the gym and the observation call in the gymnasium enviornment will automatically provide the exact sensing value specified. 
### Control:
As with sensing, PDECG supports both Neumann and Dirchilet boundary control that can be colocated with the sensing. However, the environment will throw an error if the sensing and control arguments are exactly the same. As with sensing, provide the {insert argument} when creating the gym and the action will automatically apply the control at that specific boundary position.
### Rewards:
The PDECG gym supports a variety of awards including the general $L_2$, $L_1$, $L_\infty$ norms of $u(x, t)$. Additionally PDECG allows for rewards to be taken over average times as well as differential rewards. Each reward type is listed below with the exact arguments and behavior specifications. For custom rewards, see the section labeled Tuning Existing PDE Environments. Rewards are specified in the following form 
```python
reward = {type: {"custom", "norm"}, 
          args(optional): {
                            "norm"(optional): {"1", "2", "infty"},
                            "averaging"(optional): {"temporal", "differential", "t-horizon"},
                            "t_avg"(optional): {Integer}
                          }
         }
```
An example of the $L_\infty$ reward average over the past $5$ timesteps looks like:
```python
reward = {type: "norm",
          args: {
                  "norm": "infty",
                  "averaging": "t-horizon",
                  "t_avg": 5
                }
         }
```
#### Pre-implemented reward functions:
- $L_k$ norms
    - Additional Args (given as a tuple): 
		- `norm`(optional, defaults to $L_2$ if not specified): The type of norm. $1$ specifies the Manhattan distannce, $2$ specifies the Euclidean distance, and $\infty$ specifes the max value.
        - `averaging`(optional, defaults to temporal if not specified): 
		    - `temporal`: Returns at each time $t$, the reward $\|\|u(x, t)\|\|_{L_k}$
            - `differential`: Returns at each time $t$, the reward $\|\|u(x, t) - u(x, t-dt)\|\|_{L_k}$ where $u(x, t-dt)$ is the PDE solution at the previous time step
		    - `t-horizon`: Requires an additional argument in the form `t_avg` where $t_{avg}$ is the number of timesteps to average over. If $t_{avg}=1$, this is the same as temporal, but may be slower. For $t_{avg}>1$, this will provide the reward at time $t$ as $$r(t)=\frac{1}{t_{avg}}\sum_{\tau=t-dt*t_{avg}}^t \|\|u(x, \tau)\|\|_{L_k}$$

## Hyperbolic PDE
### General System form:
PDECG supports hyperbolic PDEs of the form
```math
\frac{\partial u}{\partial t} = c \frac{\partial u}{\partial x} + \beta(x)u(0, t)
```
where $u(x, t)$ is the PDE solution at position $x \in [0, 1]$ and $t \in [0, T]$. Additionally, boundary conditions can be specified by the user at either endpoint using Neumann or Dirchilet actuation. More details on boundary conditions is provided in the section below. 



## Controllers
PDECG currently has two preimplemented controllers that can be easily used for controller the PDE. First, an analytical backstepping controller for both hyperbolic and parabolic PDEs has been developed using the algorithm in {Insert Citations of Miroslav's Papers}. Second, any RL library that works with gymansium environments can be easily used with PDECG and currently PPO with stable-baselines3 is provided int he examples tab. Details on the exact implementations for both the backstepping controller and the stable-baselines3 controller is given below. 
### Backstepping
{Add details on the backstepping controller}	
### Reinforcement Learning using Stable Baselines
{Add details on the stable baselines controller}


### Tuning Existing PDE Enviornments
#### Custom Reward Functions
#### Custom Simulation Algorithms
#### Custom figure functions

### Creating a PDE Environment

#### General Setup
#### Required Inheritance

.. _customrewards:

.. automodule:: pde_control_gym.src.rewards.base_reward

Custom Rewards
==============

Custom rewards are super easy to implement with the PDE ContRoL Gym. The only requirement is that one inherits the base class :class:`BaseReward`. 
One then needs to reimplement the reward function and reset functions (in according to the given parameters) and then just pass an instance of the class into the hyperparameters when creating the gym.

Base Reward Class
-----------------

.. autoclass:: BaseReward
   :members:

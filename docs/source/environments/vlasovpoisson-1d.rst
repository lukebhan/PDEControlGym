.. _vlasovpoisson1d:

.. automodule:: pde_control_gym.src.environments1d

Vlasov-Poisson 1D PDE
=====================

This documentation is for the 1D1V Vlasov-Poisson environment, which models a collisionless
electrostatic plasma controlled by an externally applied electric field :math:`H(x,t)`:

.. math::
    :nowrap:

    \begin{eqnarray}
    & \partial_t f(x, v, t) + v \partial_x f(x, v, t) + \left(E(x,t) + H(x,t)\right) \partial_v f(x, v, t) = 0, \\
    & E(x,t) = -\partial_x \Phi(x,t), \quad -\partial_x^2 \Phi(x,t) = \rho(x,t) - \rho_{ion}, \quad \rho(x,t) = \int f(x,v,t) dv,
    \end{eqnarray}

on :math:`x \in [0, X]` with periodic boundary conditions and :math:`v \in [-v_{max}, v_{max}]`.
Here :math:`f` is the particle distribution over phase space, :math:`E` is the
self-consistent field the plasma generates, and :math:`H` is the actuator.

The control objective is to hold the plasma at a chosen equilibrium :math:`\bar f(v)` of
the uncontrolled system, so the quantity that is observed, penalized, and reported is the
perturbation :math:`\delta f = f - \bar f`. The environment ships with the two classic
unstable equilibria, the two-stream distribution and the bump-on-tail distribution, in
``examples/vlasovPoisson/utils.py``.

This environment follows Lu, Wang and Calder, `Dynamical feedback control with operator
learning for the Vlasov-Poisson system <https://arxiv.org/abs/2509.23063>`_.

.. warning::

   Unlike every other environment in the gym, this is **not** a boundary control problem.
   The spatial domain is periodic and has no boundary, so the actuator is an external
   field distributed across the whole domain. The ``action_type`` parameter controls how
   much of that field the agent may shape.

.. autoclass:: VlasovPoisson1D
   :members:
   :exclude-members: truncate, terminate


What the control can and cannot reach
-------------------------------------

Vlasov-Poisson is unusual as a control problem because large parts of the state are
provably out of reach of any :math:`H`, and it is worth being explicit about which.

**Casimirs are invariant under every control.** The Vlasov equation transports :math:`f`
along a divergence-free flow in phase space no matter what :math:`H` is applied.
Consequently :math:`\int\int G(f) dx dv` is conserved for every function :math:`G`: total
mass, every :math:`L^p` norm of :math:`f`, and the entropy are all invariants. At any time
:math:`f(\cdot, t)` is a measure-preserving rearrangement of :math:`f_0`. If :math:`f_0` is
not already a rearrangement of :math:`\bar f`, then no control whatsoever can drive
:math:`f \to \bar f`, and :math:`\|\delta f\|_2` cannot be driven to zero. The
perturbation can only be stirred to finer and finer scales in velocity, which is what
filamentation is. Suppressing the *macroscopic* signature of the instability, the electric
field and the low-order moments, is achievable; erasing the perturbation is not.

**Two scalars are driven purely by the control.** The self-consistent field contributes
nothing to the momentum or energy budgets, so

.. math::
    :nowrap:

    \begin{eqnarray}
    & \frac{d}{dt} \int\int v f \, dx dv = \int H \rho \, dx, \\
    & \frac{d}{dt} \left( \frac{1}{2}\int\int v^2 f \, dx dv + \frac{1}{2}\int E^2 dx \right) = \int H j \, dx, \quad j = \int v f \, dv.
    \end{eqnarray}

These are the two directions in which the actuator has unobstructed authority, and both
are reported every step in the ``info`` dict as ``momentum_rate`` and ``energy_rate``
alongside the quantities they drive. Note that energy authority vanishes when the current
:math:`j` does, so a symmetric plasma at rest cannot be heated or cooled instantaneously.

**The actuator has no velocity resolution.** :math:`H` is a function of :math:`x` only and
enters the dynamics as :math:`H \partial_v f`. Linearized about :math:`\bar f`, the input
operator is :math:`B : H(x) \mapsto -H(x) \partial_v \bar f(v)`, a rank-:math:`n_x` map
into an :math:`n_x n_v` dimensional phase space. Two things follow. First, the same force
is applied to every velocity class at a given :math:`x`, so a bump-on-tail beam cannot be
decelerated without also accelerating the bulk. Second, only the part of :math:`\delta f`
aligned with :math:`\partial_v \bar f` is directly actuated; the orthogonal complement is
reached only indirectly, through free streaming and the nonlinearity.

**Mode by mode, the instability is controllable and the continuum is not.** Around a
homogeneous :math:`\bar f(v)` the linearized system decouples in the spatial Fourier
index, and mode :math:`k` obeys
:math:`\partial_t \delta f_k + i k v \delta f_k + (\delta E_k + H_k)\bar f'(v) = 0`.
The control enters mode :math:`k` only through the fixed velocity profile
:math:`\bar f'(v)`, so its reachable set is the closure of the span of the free-streaming
translates :math:`\{e^{-ikvt}\bar f'(v)\}`. The unstable eigenmodes of the two-stream and
bump-on-tail equilibria live in exactly this reachable subspace, because they are
electrostatic: they have :math:`\delta E_k \ne 0`, and :math:`H_k` can cancel it term for
term. That is precisely why the training-free law of Section 4 of the paper works. What
remains is the van Kampen continuum, which is not reachable, but is also neutrally stable
and phase-mixes on its own.

**Restricting the actuator restricts it further.** With ``action_type="modal"`` the
control is confined to the first ``num_modes`` harmonics and modes above that are exactly
uncontrollable, since the linearized modes do not couple. With
``include_uniform_mode=False`` the uniform component of :math:`H` is projected out; that
component is worth calling out separately, because a periodic self-consistent :math:`E`
always has zero mean, so the uniform accelerating field is the one actuator direction the
plasma can never produce for itself.

**Observability is limited in the same way.** With ``sensing_type="density"`` or
``"field"`` the agent sees only a velocity moment of :math:`\delta f`. The phase-mixing
continuum contributes to :math:`\rho` only through a decaying transient, so it is
essentially unobservable from field diagnostics, while the unstable eigenmodes are both
observable and controllable. Concretely, :math:`-\delta E` in the paper's feedback law is
a functional of :math:`\delta\rho` alone and can be implemented from a density
measurement, but the dissipation term
:math:`\gamma \int \delta f \partial_v \bar f dv` cannot, and needs ``"full"`` sensing.


Numerical Implementation
------------------------

The system is advanced with the Strang-split semi-Lagrangian scheme of the paper, which is
fully explicit and carries no CFL restriction on :math:`\Delta t` because each substage is
an exact solve along characteristics followed by interpolation:

.. math::
    :nowrap:

    \begin{eqnarray}
    f^{(1)}(x, v) &=& f\left(x - \tfrac{\Delta t}{2} v,\; v,\; t^n\right), \\
    f^{(2)}(x, v) &=& f^{(1)}\left(x,\; v - \left(E[f^{(1)}](x) + H(x)\right)\Delta t\right), \\
    f^{n+1}(x, v) &=& f^{(2)}\left(x - \tfrac{\Delta t}{2} v,\; v\right).
    \end{eqnarray}

The Poisson solve is spectral for ``poisson_solver="periodic"``, and uses the explicit
Green's function

.. math::

    G(x, y) = \begin{cases} (x-a)(b-y)/(b-a), & a \le x \le y \le b, \\ (y-a)(b-x)/(b-a), & a \le y \le x \le b, \end{cases}

for ``poisson_solver="dirichlet"``, which reproduces the :math:`\Phi(0) = \Phi(X) = 0`
setup of Section 3.1 of the paper.

.. note::

   The default interpolation is a four point cubic Lagrange stencil rather than the linear
   one of the paper, and this matters for benchmarking. The diffusion a two point stencil
   adds per step is proportional to :math:`\alpha(1-\alpha)\Delta^2` with
   :math:`\alpha \propto \Delta t` the fractional shift, so the diffusion accumulated over
   a fixed horizon is *independent of* :math:`\Delta t` and cannot be refined away in
   time. On the reference two-stream case it removes about 45% of
   :math:`\|\delta f\|_2` over :math:`t \in [0, 40]` all by itself, which a controller
   would otherwise be credited for. Run
   ``examples/vlasovPoisson/vlasovPoisson1DtestSolver.py`` to see the comparison: with the
   cubic stencil the pure cancellation law at :math:`\gamma = 0` conserves
   :math:`\|\delta f\|_2` to 2% over that horizon, as the continuous theory says it must,
   and linear Landau damping rates come out within 3% of the roots of the dispersion
   relation. Set ``interpolation="linear"`` to reproduce the paper exactly, or if
   positivity of :math:`f` matters more than dissipation.


Examples
--------

``examples/vlasovPoisson`` contains:

- ``vlasovPoisson1Dcancellation.py``, which reproduces the two-stream experiment of
  Section 4 with the training-free feedback law
  :math:`H = -\delta E + \gamma \int \delta f \partial_v \bar f dv`,
- ``vlasovPoisson1Dppo.py``, which trains a PPO agent under density-only sensing and modal
  actuation,
- ``vlasovPoisson1DtestSolver.py``, which checks the invariants, the two budget
  identities, linear Landau damping, and the reachability limits described above.

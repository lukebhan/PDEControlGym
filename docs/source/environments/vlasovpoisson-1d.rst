.. _vlasovpoisson1d:

.. automodule:: pde_control_gym.src.environments1d

Vlasov-Poisson 1D PDE
=====================

This documentation provides a detailed description of the Vlasov-Poisson model for
collisionless plasma control and its implementation.

Sustained nuclear fusion requires that a plasma stay confined for long periods, and the
main obstacle is that the confined state is often not a stable one. A small deviation from
equilibrium can be amplified by the electric field the plasma itself generates, growing
exponentially until the plasma is nothing like the state it started in. Two such
instabilities are standard test cases. The **two-stream instability** arises when two
populations of particles counter-stream through one another, and it scatters the beams and
destroys their focus. The **bump-on-tail instability** arises when a small fast population
sits on the tail of a bulk distribution, as happens with runaway electrons or under
radiofrequency heating, and it degrades confinement and deposits energy on the reactor
walls. Both are driven by the same mechanism, energy flowing from the particles into
electrostatic waves. Suppressing them means applying a carefully shaped external electric
field, fast enough to act before the growth compounds.

.. figure:: ../_static/img/vlasovPoissonSchematic.png
   :align: center

The environment implements the one-dimensional, one-velocity (1D1V) Vlasov-Poisson system.
Let :math:`f(x, v, t)` denote the particle distribution over phase space, that is, the
density of particles at position :math:`x` moving with velocity :math:`v` at time
:math:`t`. Let :math:`\rho(x, t)` be the charge density obtained by integrating out
velocity, :math:`E(x,t)` the electric field the plasma generates self-consistently through
the Poisson equation, and :math:`H(x, t)` the external field applied for control. The
dynamics are

.. math::
    :nowrap:

    \begin{align}
    \partial_t f(x,v,t) + v\, \partial_x f(x,v,t) + \big(E(x,t) + H(x,t)\big)\, \partial_v f(x,v,t) &= 0, \tag{1} \\
    E(x,t) = -\partial_x \Phi(x,t), \qquad -\partial_x^2 \Phi(x,t) &= \rho(x,t) - \rho_{ion}, \tag{2} \\
    \rho(x,t) &= \int f(x,v,t)\, dv, \tag{3}
    \end{align}

on a domain :math:`x \in [0, X]` with periodic boundary conditions and a truncated velocity
range :math:`v \in [-v_{max}, v_{max}]`. Here :math:`\rho_{ion}` is a fixed neutralizing
background of ions, taken as constant because the ions are far heavier than the electrons
and barely move on the timescale of interest.

Equation (1) is a transport equation: it says that :math:`f` is carried along the
characteristics of the flow, with particles moving at their velocity and accelerating under
the total field :math:`E + H`. Equations (2) and (3) close the loop by making the field a
functional of the distribution itself, and it is this feedback of the plasma on its own
field that makes some equilibria unstable.

This environment follows Lu, Wang and Calder, `Dynamical feedback control with operator
learning for the Vlasov-Poisson system <https://arxiv.org/abs/2509.23063>`_.

.. warning::

   Unlike every other environment in the gym, this is **not** a boundary control problem.
   The spatial domain is periodic and therefore has no boundary, so the actuator is a field
   distributed across the whole domain. The ``action_type`` parameter controls how much of
   that field the agent is allowed to shape.


Equilibria and the control objective
------------------------------------

Any velocity profile :math:`\bar f(v)` with no spatial dependence is an equilibrium of the
uncontrolled system, since :math:`v \partial_x \bar f = 0` and the resulting field
:math:`\bar E` vanishes when :math:`\int \bar f dv = \rho_{ion}`. Whether that equilibrium
is *stable* is a separate question, decided by the Landau dispersion relation. A single
Maxwellian is stable, and perturbations to it decay by phase mixing, the classical Landau
damping. A mixture of two Gaussians is not.

The two equilibria shipped in ``examples/vlasovPoisson/utils.py`` are

.. math::
    :nowrap:

    \begin{align}
    \bar f_{\text{two-stream}}(v) &= \frac{1}{2\sqrt{2\pi}} e^{-(v - \bar v)^2 / 2} + \frac{1}{2\sqrt{2\pi}} e^{-(v + \bar v)^2 / 2}, \qquad \bar v = 2.4, \tag{4} \\
    \bar f_{\text{bump-on-tail}}(v) &= \frac{\omega_1}{\sqrt{2\pi}} e^{-(v - \bar v_1)^2 / 2} + \frac{\omega_2}{\sqrt{2\pi} v_t} e^{-(v - \bar v_2)^2 / 2 v_t}, \tag{5}
    \end{align}

with :math:`\omega_1 = 0.9`, :math:`\omega_2 = 0.1`, :math:`\bar v_1 = -2`,
:math:`\bar v_2 = 3.5` and :math:`v_t = 0.25`.

.. figure:: ../_static/img/vlasovPoissonEquilibria.png
   :align: center

Instability is wavenumber-dependent, and the right panel above makes the practical point:
for the two-stream case on :math:`X = 10\pi`, only the first two spatial harmonics grow.
Harmonic three and above are Landau damped and need no control at all. Since the linearized
system decouples across spatial harmonics, an actuator restricted to a set of harmonics can
only act on those harmonics, so this measurement is what should determine the action space.

The control objective is to hold the plasma at :math:`\bar f`, so the quantity that is
observed, penalized and reported throughout is the perturbation

.. math::
    :nowrap:

    \begin{align}
    \delta f(x,v,t) = f(x,v,t) - \bar f(v). \tag{6}
    \end{align}

The control action is the external field :math:`H(x, t)`, parameterized in one of two ways
through ``action_type``:

1. ``"modal"``, in which the action holds the coefficients :math:`\alpha_k` of a truncated
   Fourier basis, :math:`H(x) = \sum_k \alpha_k \varphi_k(x)` with
   :math:`\{\varphi_k\} = \{1, \sin(2\pi l x / X), \cos(2\pi l x / X)\}_{l \le K}`. This is
   the parameterization used in the paper, and it makes the actuated wavenumbers explicit.

2. ``"field"``, in which the action holds the nodal values :math:`H(x_i)` directly, giving
   full authority over the field on the grid.

Setting ``include_uniform_mode=False`` additionally projects out the spatially uniform
component, so that :math:`\int_0^X H \, dx = 0` and :math:`H` is the gradient of a periodic
potential.

.. autoclass:: VlasovPoisson1D
   :members:
   :exclude-members: truncate, terminate


What the control can and cannot reach
-------------------------------------

Vlasov-Poisson is unusual as a control problem because large parts of the state are
provably out of reach of any :math:`H`, and it is worth being explicit about which. The
claims below are all checked numerically by
``examples/vlasovPoisson/vlasovPoisson1DtestSolver.py``.

**Casimirs are invariant under every control.** The Vlasov equation transports :math:`f`
along a divergence-free flow in phase space no matter what :math:`H` is applied.
Consequently :math:`\int\int G(f)\, dx\, dv` is conserved for every function :math:`G`:
total mass, every :math:`L^p` norm of :math:`f`, and the entropy are all invariants. At any
time :math:`f(\cdot, t)` is a measure-preserving rearrangement of :math:`f_0`. If
:math:`f_0` is not already a rearrangement of :math:`\bar f`, then no control whatsoever can
drive :math:`f \to \bar f`, and :math:`\|\delta f\|_2` cannot be driven to zero. The
perturbation can only be stirred to finer and finer scales in velocity, which is what
filamentation is. Suppressing the *macroscopic* signature of the instability, the electric
field and the low-order moments, is achievable; erasing the perturbation is not.

**Two scalars are driven purely by the control.** The self-consistent field contributes
nothing to the momentum or energy budgets, so with :math:`j = \int v f\, dv` the current,

.. math::
    :nowrap:

    \begin{align}
    \frac{d}{dt} \int\!\!\int v f \, dx\, dv &= \int H \rho \, dx, \tag{7} \\
    \frac{d}{dt} \left( \frac{1}{2}\int\!\!\int v^2 f \, dx\, dv + \frac{1}{2}\int E^2 dx \right) &= \int H j \, dx. \tag{8}
    \end{align}

These are the two directions in which the actuator has unobstructed authority, and both are
reported every step in the ``info`` dict as ``momentum_rate`` and ``energy_rate`` alongside
the quantities they drive. Note that energy authority vanishes when the current :math:`j`
does, so a symmetric plasma at rest cannot be heated or cooled instantaneously.

**The actuator has no velocity resolution.** :math:`H` is a function of :math:`x` only and
enters the dynamics as :math:`H \partial_v f`. Linearized about :math:`\bar f`, the input
operator is :math:`B : H(x) \mapsto -H(x) \partial_v \bar f(v)`, a rank-:math:`n_x` map into
an :math:`n_x n_v` dimensional phase space, which at the default resolution is under one
percent of the state directions. Two things follow. First, the same force is applied to
every velocity class at a given :math:`x`, so a bump-on-tail beam cannot be decelerated
without also accelerating the bulk. Second, only the part of :math:`\delta f` aligned with
:math:`\partial_v \bar f` is directly actuated; the orthogonal complement is reached only
indirectly, through free streaming and the nonlinearity.

**Mode by mode, the instability is controllable and the continuum is not.** Around a
homogeneous :math:`\bar f(v)` the linearized system decouples in the spatial Fourier index,
and mode :math:`k` obeys

.. math::
    :nowrap:

    \begin{align}
    \partial_t \delta f_k + i k v\, \delta f_k + \big(\delta E_k + H_k\big) \bar f'(v) = 0. \tag{9}
    \end{align}

The control enters mode :math:`k` only through the fixed velocity profile
:math:`\bar f'(v)`, so its reachable set is the closure of the span of the free-streaming
translates :math:`\{e^{-ikvt}\bar f'(v)\}`. The unstable eigenmodes of the two-stream and
bump-on-tail equilibria live in exactly this reachable subspace, because they are
electrostatic: they have :math:`\delta E_k \ne 0`, and :math:`H_k` can cancel it term for
term. That is precisely why the training-free law of Section 4 of the paper works. What
remains is the van Kampen continuum, which is not reachable, but is also neutrally stable
and phase-mixes on its own.

**Restricting the actuator restricts it further.** With ``action_type="modal"`` the control
is confined to the first ``num_modes`` harmonics and modes above that are exactly
uncontrollable, since the linearized modes do not couple. With
``include_uniform_mode=False`` the uniform component of :math:`H` is projected out; that
component is worth calling out separately, because a periodic self-consistent :math:`E`
always has zero mean, so the uniform accelerating field is the one actuator direction the
plasma can never produce for itself.

**Observability is limited in the same way.** With ``sensing_type="density"`` or
``"field"`` the agent sees only a velocity moment of :math:`\delta f`. The phase-mixing
continuum contributes to :math:`\rho` only through a decaying transient, so it is
essentially unobservable from field diagnostics, while the unstable eigenmodes are both
observable and controllable. Concretely, :math:`-\delta E` in the paper's feedback law is a
functional of :math:`\delta\rho` alone and can be implemented from a density measurement,
but the dissipation term :math:`\gamma \int \delta f \partial_v \bar f dv` cannot, and needs
``"full"`` sensing.


Numerical implementation
------------------------

The system is advanced with a Strang-split semi-Lagrangian scheme. Splitting is natural
here because equation (1) separates into two pieces that can each be solved exactly along
characteristics: transport in :math:`x` at fixed :math:`v`, and transport in :math:`v` at
fixed :math:`x`. Each half step is therefore a *shift* of the distribution followed by an
interpolation back onto the grid, which makes the scheme fully explicit and, unlike a
finite difference discretization, free of any CFL restriction on :math:`\Delta t`.

One step from :math:`t^n` to :math:`t^{n+1}` is

.. math::
    :nowrap:

    \begin{align}
    f^{(1)}(x, v) &= f\left(x - \tfrac{\Delta t}{2} v,\; v,\; t^n\right), \tag{10} \\
    f^{(2)}(x, v) &= f^{(1)}\left(x,\; v - \big(E[f^{(1)}](x) + H(x)\big)\Delta t\right), \tag{11} \\
    f^{n+1}(x, v) &= f^{(2)}\left(x - \tfrac{\Delta t}{2} v,\; v\right), \tag{12}
    \end{align}

where the field in stage (11) is recomputed from the half-advected state :math:`f^{(1)}`,
which is what makes the splitting second order in :math:`\Delta t`.

**Interpolation.** Let :math:`x_i = i \Delta x` and :math:`v_j = -v_{max} + j \Delta v`. For
the streaming stages the departure point of node :math:`i` is :math:`x_i - s_j \Delta x`
with :math:`s_j = v_j \Delta t / 2 \Delta x`. Writing
:math:`s_j = \lfloor s_j \rfloor + \{s_j\}`, the departure point lies between nodes
:math:`i - \lfloor s_j \rfloor - 1` and :math:`i - \lfloor s_j \rfloor`, at local coordinate
:math:`\sigma = 1 - \{s_j\}`. The update is a weighted sum over a stencil around that node,

.. math::
    :nowrap:

    \begin{align}
    f^{(1)}_{i,j} = \sum_{m} w_m(\sigma) \; f^{\,n}_{\,(i - \lfloor s_j \rfloor - 1 + m) \bmod n_x,\; j}, \tag{13}
    \end{align}

with the modulo implementing periodicity in :math:`x`. The acceleration stage (11) is the
same construction along :math:`j` with shift :math:`(E_i + H_i)\Delta t / \Delta v`, except
that the velocity domain is a truncation rather than a period, so stencil points falling
outside :math:`[0, n_v)` contribute zero. The weights are either the two point linear set

.. math::
    :nowrap:

    \begin{align}
    w_0 = 1 - \sigma, \qquad w_1 = \sigma, \tag{14}
    \end{align}

or the four point Lagrange set on the stencil :math:`m \in \{-1, 0, 1, 2\}`,

.. math::
    :nowrap:

    \begin{align}
    w_{-1} &= -\tfrac{1}{6}\sigma(\sigma-1)(\sigma-2), &
    w_{0} &= \tfrac{1}{2}(\sigma+1)(\sigma-1)(\sigma-2), \tag{15} \\
    w_{1} &= -\tfrac{1}{2}(\sigma+1)\sigma(\sigma-2), &
    w_{2} &= \tfrac{1}{6}(\sigma+1)\sigma(\sigma-1). \tag{16}
    \end{align}

.. note::

   The default is the cubic stencil (15)-(16) rather than the bilinear scheme of the paper,
   and for benchmarking purposes the difference matters. The numerical diffusion a two
   point stencil introduces per step is proportional to :math:`\sigma(1-\sigma)\Delta^2`,
   and since :math:`\sigma \propto \Delta t` the diffusion accumulated over a fixed horizon
   is *independent of* :math:`\Delta t`. It cannot be refined away in time, only by raising
   the order or the resolution, and it appears as a spurious decay of
   :math:`\|\delta f\|_2` that flatters any controller. On the reference two-stream case it
   removes about 45% of :math:`\|\delta f\|_2` over :math:`t \in [0, 40]` on its own.

   With the cubic stencil, the pure cancellation law at :math:`\gamma = 0` conserves
   :math:`\|\delta f\|_2` to 2% over that horizon, as the continuous theory says it must,
   and linear Landau damping rates come out within 3% of the roots of the dispersion
   relation at :math:`k = 0.4` and :math:`k = 0.5`. Set ``interpolation="linear"`` to
   reproduce the paper exactly, or if positivity of :math:`f` matters more to you than
   dissipation, since the cubic stencil can undershoot below zero at sharp features.

**Poisson solve.** With ``poisson_solver="periodic"`` equation (2) is solved spectrally.
Writing :math:`\widehat{c}` for the discrete Fourier transform of the net charge
:math:`c = \rho - \rho_{ion}`, the field is

.. math::
    :nowrap:

    \begin{align}
    \widehat{E}_k = \frac{\widehat{c}_k}{i k}, \quad k \ne 0, \qquad \widehat{E}_0 = 0, \tag{17}
    \end{align}

the :math:`k = 0` mode being fixed to zero as the only choice consistent with a periodic
potential. With ``poisson_solver="dirichlet"`` the solve instead uses the Green's function
of :math:`-\partial_x^2` on :math:`[a,b]` with zero boundary data,

.. math::
    :nowrap:

    \begin{align}
    G(x, y) = \begin{cases} (x-a)(b-y)/(b-a), & a \le x \le y \le b, \\ (y-a)(b-x)/(b-a), & a \le y \le x \le b, \end{cases} \tag{18}
    \end{align}

assembled once into a dense matrix so that :math:`E = -\int \partial_x G(x,y) c(y) dy`
becomes a single matrix-vector product. This reproduces the :math:`\Phi(0) = \Phi(X) = 0`
setup of Section 3.1 of the paper.

**Diagnostics.** Every call to ``step`` returns an ``info`` dict holding the applied field
``H`` and the resulting field ``E``, the invariants ``mass``, ``momentum``,
``kinetic_energy`` and ``electric_energy``, the two right hand sides of (7) and (8) as
``momentum_rate`` and ``energy_rate``, and ``l2_perturbation`` for
:math:`\|\delta f(t)\|_2`. Together these let a user verify that the solver conserves what
it should and that the control moves only what it can.


Reinforcement learning on this environment
------------------------------------------

Handed to an RL algorithm unmodified, this environment trains poorly, and the reasons are
structural rather than incidental. An unstable perturbation grows by two to three orders of
magnitude within one episode, so the observation spans that range and the quadratic cost
spans its square; a fixed set of network weights cannot respond usefully at both ends. The
useful control amplitude is also small and phase-sensitive, so Gaussian exploration at a
scale comparable to the correct action drives the instability rather than damping it, and
most rollouts come out worse than doing nothing.

``examples/vlasovPoisson/utils.py`` provides ``ScaleFreeVlasov``, a wrapper that addresses
this: it divides the observation by its own magnitude and appends the :math:`\log_{10}` of
that magnitude, interprets the action as a multiple of the same magnitude, and replaces the
reward by the per-step log growth factor. The paper's quadratic cost stays available under
``info["paper_reward"]`` so it remains the reported metric.

Rescaling by the amplitude is not merely a trick. Section 2.1 of the paper shows via the
Pontryagin maximum principle that the optimal control for the linearized system is a
*linear* functional of :math:`\delta f`, hence positively homogeneous in it, so a single set
of weights represents the law at every amplitude once the scale is factored out.

The notebook and ``vlasovPoisson1Dppo.py`` train ``net_arch=[64, 64]`` with tanh at
``lr=3e-4``, reaching a growth factor of about 1.2 after 600k steps against 117 with no
control. The learning rate is tied to the architecture and cannot be carried over from
elsewhere: at ``lr=1e-3`` the same network diverges, reaching growth factors of 74 and 135
by 400k steps on two seeds, ``net_arch=[256, 256]`` settles around 900, which is worse than
applying no control at all, and a ReLU activation in place of tanh is worse again.

If you want a smaller hypothesis class, ``net_arch=[]`` is well matched to the problem, for
the reason given above. It trains stably at ``lr=1e-3`` and reaches the same growth factor
in roughly a third of the samples. Measured over three seeds out to 1.2M steps the
late-phase growth factor averaged 1.33 for the linear policy against 1.27 for the MLP,
which is a tie; the difference between them is conditioning and sample efficiency, not
capacity.

See the `tutorial <../tutorials/vlasovpoisson-1d_tutorial.html>`_ for a full worked
example.


Examples
--------

``examples/vlasovPoisson`` contains:

- ``VlasovPoisson1DExample.ipynb``, a worked notebook that identifies which harmonics are
  actually unstable, builds both baselines, and trains a PPO agent against them,
- ``vlasovPoisson1Dcancellation.py``, which reproduces the two-stream experiment of
  Section 4 with the training-free feedback law
  :math:`H = -\delta E + \gamma \int \delta f \partial_v \bar f dv`,
- ``vlasovPoisson1Dppo.py``, the script form of the notebook's training run,
- ``vlasovPoisson1DtestSolver.py``, which checks the invariants, the two budget identities,
  linear Landau damping, and the reachability limits described above.

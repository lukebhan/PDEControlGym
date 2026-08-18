import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional, Union

from pde_control_gym.src.environments1d.base_env_1d import PDEEnv1D

# np.trapz was renamed to np.trapezoid in numpy 2.0 and removed in later releases.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _stencil_weights(s, order):
    r"""
    Interpolation weights at local coordinate ``s`` in :math:`[0, 1]`, measured from the
    node immediately below the departure point.

    ``"linear"`` returns the two point weights and is the scheme used in the paper.
    ``"cubic"`` returns the four point Lagrange weights on the stencil
    :math:`\{-1, 0, 1, 2\}`, whose leading error is :math:`O(\Delta^4)` rather than
    :math:`O(\Delta^2)`. The distinction matters here: a two point scheme leaks an
    amount of numerical diffusion per step proportional to :math:`\alpha(1-\alpha)\Delta^2`
    with :math:`\alpha` the fractional shift, and since :math:`\alpha \propto \Delta t`
    the diffusion accumulated over a fixed horizon is independent of :math:`\Delta t`. It
    can only be reduced by refining the grid or raising the order, and it shows up as a
    spurious decay of :math:`\|\delta f\|_2` that flatters any controller.

    Returns a list of ``(node offset, weight)`` pairs.
    """
    match order:
        case "linear":
            return [(0, 1.0 - s), (1, s)]
        case "cubic":
            return [
                (-1, -s * (s - 1.0) * (s - 2.0) / 6.0),
                (0, (s + 1.0) * (s - 1.0) * (s - 2.0) / 2.0),
                (1, -(s + 1.0) * s * (s - 2.0) / 2.0),
                (2, (s + 1.0) * s * (s - 1.0) / 6.0),
            ]
        case _:
            raise Exception(
                "Invalid interpolation parameter. Please use 'linear' or 'cubic'. See documentation for details."
            )


def _periodic_shift(f, shift_cells, order="cubic"):
    r"""
    Semi-Lagrangian interpolation of ``f`` along axis 0 with a periodic wrap.

    ``f`` has shape ``(nx, nv)`` and ``shift_cells`` has shape ``(nv,)``, holding the
    (real valued) departure offset :math:`v \Delta t / \Delta x` of every velocity row.
    Returns :math:`f(x - v \Delta t, v)` on the grid.
    """
    nx = f.shape[0]
    i0 = np.floor(shift_cells).astype(np.int64)
    frac = shift_cells - i0
    # Departure point i - shift lies between nodes (i - i0 - 1) and (i - i0), so the
    # local coordinate measured from the lower node is 1 - frac.
    base = (np.arange(nx)[:, None] - i0[None, :] - 1) % nx
    cols = np.arange(f.shape[1])[None, :]
    out = np.zeros_like(f)
    for offset, weight in _stencil_weights(1.0 - frac, order):
        out += weight * f[(base + offset) % nx, cols]
    return out


def _bounded_shift(f, shift_cells, order="cubic"):
    r"""
    Semi-Lagrangian interpolation of ``f`` along axis 1 (velocity) with the
    outflow/zero-extension convention :math:`f \equiv 0` outside :math:`[-v_{max}, v_{max}]`.

    ``f`` has shape ``(nx, nv)`` and ``shift_cells`` has shape ``(nx,)``, holding the
    departure offset :math:`(E + H) \Delta t / \Delta v` at every spatial point.
    Returns :math:`f(x, v - (E+H) \Delta t)`.
    """
    nv = f.shape[1]
    j0 = np.floor(shift_cells).astype(np.int64)
    frac = shift_cells - j0
    base = np.arange(nv)[None, :] - j0[:, None] - 1
    rows = np.arange(f.shape[0])[:, None]
    out = np.zeros_like(f)
    for offset, weight in _stencil_weights((1.0 - frac)[:, None], order):
        node = base + offset
        inside = (node >= 0) & (node < nv)
        out += weight * np.where(inside, f[rows, np.clip(node, 0, nv - 1)], 0.0)
    return out


class VlasovPoisson1D(PDEEnv1D):
    r"""
    Vlasov-Poisson 1D (one spatial dimension, one velocity dimension)

    This class implements the collisionless Vlasov-Poisson system with an external
    (in-domain) electric field as the control input and inherits from :class:`PDEEnv1D`.
    For the full list of shared arguments see :class:`PDEEnv1D` first.

    .. math::

        \begin{eqnarray}
        & \partial_t f + v \partial_x f + (E(x,t) + H(x,t)) \partial_v f = 0 \\
        & E = -\partial_x \Phi, \quad -\partial_x^2 \Phi = \rho - \rho_{ion}, \quad
          \rho(x,t) = \int f(x,v,t) dv
        \end{eqnarray}

    for :math:`x \in [0, X]` (periodic) and :math:`v \in [-v_{max}, v_{max}]`. The
    control :math:`H(x,t)` is an externally applied electric field: unlike every other
    environment in the gym this is a *distributed* (in-domain) actuator, not a boundary
    actuator, because the Vlasov equation is posed on a periodic spatial domain and has
    no controllable boundary.

    The regulation target is an equilibrium :math:`\bar f(v)` of the uncontrolled system
    and the state that is reported and penalized is the perturbation
    :math:`\delta f = f - \bar f`.

    This implementation follows Lu, Wang and Calder, *Dynamical feedback control with
    operator learning for the Vlasov-Poisson system* (arXiv:2509.23063).

    :param reset_init_condition_func: Function ``(x, v) -> f0`` returning the initial
        distribution :math:`f(x, v, 0)` of shape ``(nx, nv)``. ``x`` and ``v`` are the
        1D grids, so use ``np.meshgrid(x, v, indexing="ij")`` inside.
    :param equilibrium_func: Function ``(x, v) -> fbar`` returning the target equilibrium
        :math:`\bar f` of shape ``(nx, nv)``. Must be an equilibrium of the *uncontrolled*
        system, i.e. satisfy :math:`v \partial_x \bar f + \bar E \partial_v \bar f = 0`;
        any :math:`\bar f = \bar f(v)` qualifies.
    :param sensing_noise_func: Function applied to the observation before it is returned.
        Used to model noisy diagnostics. Defaults to the identity.
    :param V: The velocity domain is :math:`[-V, V]`.
    :param dv: The velocity grid spacing.
    :param rho_ion: Constant neutralizing background ion density.
    :param action_type: ``"modal"`` gives the agent the coefficients of the truncated
        Fourier basis :math:`\{1, \sin(2\pi l x/X), \cos(2\pi l x /X)\}_{l \le K}` used in
        the paper, so the action is a vector of length ``2 * num_modes + 1``.
        ``"field"`` gives the agent the nodal values :math:`H(x_i)` directly, so the
        action has length ``nx``.
    :param num_modes: Number of harmonics :math:`K` retained when ``action_type`` is
        ``"modal"``. Modes above :math:`K` are *not* actuated.
    :param include_uniform_mode: When ``False`` the spatially uniform component of
        :math:`H` is projected out, so :math:`\int_0^X H dx = 0` and the control is the
        gradient of a periodic potential. When ``True`` the agent may apply a uniform
        accelerating field, which the self-consistent field :math:`E` can never produce.
    :param sensing_type: What the agent observes. ``"full"`` returns
        :math:`\delta f` of shape ``(nx, nv)``; ``"density"`` returns :math:`\delta\rho(x)`;
        ``"field"`` returns :math:`\delta E(x)`; ``"moments"`` returns the first three
        velocity moments of :math:`\delta f` stacked as shape ``(3, nx)``.
    :param poisson_solver: ``"periodic"`` solves the Poisson equation spectrally with
        periodic boundary conditions on :math:`\Phi`. ``"dirichlet"`` solves it with
        :math:`\Phi(0) = \Phi(X) = 0`, which is the setup used in Section 3.1 of the paper.
    :param interpolation: Order of the semi-Lagrangian interpolation. ``"linear"`` is the
        bilinear scheme of the paper. ``"cubic"`` is a four point Lagrange stencil and is
        the default because the numerical diffusion of the linear scheme does not vanish
        as :math:`\Delta t \to 0` and produces a spurious decay of
        :math:`\|\delta f\|_2` that a controller gets undeserved credit for. ``"cubic"``
        can undershoot below zero in sharp regions; use ``"linear"`` if positivity of
        :math:`f` matters more than dissipation.
    :param max_control_value: Sets the maximum control value as
        [``-max_control_value``, ``max_control_value``] and is used to normalize actions.
    :param limit_pde_state_size: Terminates the episode early if
        :math:`\|\delta f\|_{L_2} \ge` ``max_state_value``.
    :param max_state_value: Threshold used by ``limit_pde_state_size``.
    :param control_sample_rate: The controller is held constant over this many time units
        while the PDE is advanced at resolution ``dt``.
    :param store_history: When ``True`` the full ``(nt, nx, nv)`` trajectory is retained
        for the reward class and for plotting. Set to ``False`` for long runs on fine
        grids, in which case only the two most recent slices are kept.
    """

    def __init__(
        self,
        reset_init_condition_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        equilibrium_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        sensing_noise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        V: float = 8.0,
        dv: float = 0.08,
        rho_ion: float = 1.0,
        action_type: str = "modal",
        num_modes: int = 15,
        include_uniform_mode: bool = True,
        sensing_type: str = "full",
        poisson_solver: str = "periodic",
        interpolation: str = "cubic",
        max_control_value: float = 1.0,
        limit_pde_state_size: bool = False,
        max_state_value: float = 1e10,
        control_sample_rate: float = 0.2,
        store_history: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Recorded so that hand written controllers can undo the action scaling that the
        # base class applies when ``normalize`` is True.
        self.normalize_actions = kwargs.get("normalize", False)
        self.reset_init_condition_func = reset_init_condition_func
        self.equilibrium_func = equilibrium_func
        self.sensing_noise_func = (
            sensing_noise_func
            if sensing_noise_func is not None
            else lambda state: state
        )
        self.V = V
        self.dv = dv
        self.nv = int(round(2 * V / dv)) + 1
        self.rho_ion = rho_ion
        self.action_type = action_type
        self.num_modes = num_modes
        self.include_uniform_mode = include_uniform_mode
        self.sensing_type = sensing_type
        self.poisson_solver = poisson_solver
        self.interpolation = interpolation
        self.max_control_value = max_control_value
        self.limit_pde_state_size = limit_pde_state_size
        self.max_state_value = max_state_value
        self.control_sample_rate = control_sample_rate
        self.store_history = store_history

        # Phase space grid. x is periodic so the endpoint X is the same node as 0 and is
        # excluded, while v is a bounded truncation of the real line and keeps both ends.
        self.x = np.linspace(0, self.X, self.nx, endpoint=False)
        self.v = np.linspace(-V, V, self.nv)

        self._build_control_basis()
        self._build_poisson_solver()
        _stencil_weights(0.5, self.interpolation)

        match self.sensing_type:
            case "full":
                obs_shape = (self.nx, self.nv)
            case "density" | "field":
                obs_shape = (self.nx,)
            case "moments":
                obs_shape = (3, self.nx)
            case _:
                raise Exception(
                    "Invalid sensing_type parameter. Please use 'full', 'density', 'field', or 'moments'. See documentation for details."
                )
        # float64 throughout: the semi-Lagrangian solve and the Poisson inversion are run
        # in double precision and the perturbations of interest are O(1e-4) of the state.
        self.observation_space = spaces.Box(
            np.full(obs_shape, -np.inf, dtype="float64"),
            np.full(obs_shape, np.inf, dtype="float64"),
            dtype=np.float64,
        )

        # The 1D base class assumes a single scalar boundary actuator. Vlasov-Poisson is
        # controlled in the domain, so both the action space and the state buffer that the
        # base class allocated are replaced here.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.u = None
        self.f = None

    def _build_control_basis(self):
        r"""
        Builds the matrix mapping an action to the nodal values of :math:`H(x_i)`.

        For ``action_type == "modal"`` this is the truncated Fourier basis of the paper,
        which caps the actuated wavenumbers at ``num_modes``. For ``action_type ==
        "field"`` it is the identity, giving full authority over :math:`H` on the grid.
        """
        match self.action_type:
            case "modal":
                cols = [np.ones_like(self.x)]
                for l in range(1, self.num_modes + 1):
                    k = 2 * np.pi * l / self.X
                    cols.append(np.sin(k * self.x))
                    cols.append(np.cos(k * self.x))
                self.control_basis = np.stack(cols, axis=1)
            case "field":
                self.control_basis = np.eye(self.nx)
            case _:
                raise Exception(
                    "Invalid action_type parameter. Please use 'modal' or 'field'. See documentation for details."
                )
        if not self.include_uniform_mode:
            # Remove the k=0 component of every basis function so that no action can
            # produce a net uniform force on the plasma. The constant basis function is
            # annihilated by this projection, so it is dropped from the action space
            # rather than left in as an inert coordinate.
            self.control_basis = self.control_basis - self.control_basis.mean(
                axis=0, keepdims=True
            )
            keep = np.linalg.norm(self.control_basis, axis=0) > 1e-12
            self.control_basis = self.control_basis[:, keep]
        self.action_dim = self.control_basis.shape[1]

    def _build_poisson_solver(self):
        r"""
        Precomputes the Poisson solve :math:`\delta\rho \mapsto E`.
        """
        match self.poisson_solver:
            case "periodic":
                kx = 2 * np.pi * np.fft.rfftfreq(self.nx, d=self.dx)
                self._ik_inv = np.zeros_like(kx, dtype=np.complex128)
                self._ik_inv[1:] = 1.0 / (1j * kx[1:])
            case "dirichlet":
                # Green's function of -d^2/dx^2 on [0, X] with zero boundary data,
                # G(x, y) = x (X - y) / X for x <= y and y (X - x) / X otherwise.
                # E = -dPhi/dx is assembled once as a dense nx-by-nx matrix.
                xg = self.x[:, None]
                yg = self.x[None, :]
                dG = np.where(xg <= yg, (self.X - yg) / self.X, -yg / self.X)
                self._efield_matrix = -dG * self.dx
            case _:
                raise Exception(
                    "Invalid poisson_solver parameter. Please use 'periodic' or 'dirichlet'. See documentation for details."
                )

    def solve_poisson(self, charge: np.ndarray):
        r"""
        Solves :math:`-\partial_x^2 \Phi = \text{charge}`, :math:`E = -\partial_x \Phi`.

        :param charge: The net charge density on the spatial grid, normally
            :math:`\rho - \rho_{ion}` for the field :math:`E` or :math:`\delta\rho` for
            the perturbation field :math:`\delta E`.
        """
        match self.poisson_solver:
            case "periodic":
                # In Fourier space -Phi_xx = c gives Phi_k = c_k / k^2 and therefore
                # E_k = -i k Phi_k = c_k / (i k). The k = 0 mode is unconstrained and is
                # set to zero, which is the only choice consistent with a periodic Phi.
                chat = np.fft.rfft(charge - charge.mean())
                return np.fft.irfft(chat * self._ik_inv, n=self.nx)
            case "dirichlet":
                return self._efield_matrix @ charge

    def density(self, f: np.ndarray):
        r"""
        Zeroth velocity moment :math:`\rho(x) = \int f(x, v) dv`.
        """
        return _trapz(f, dx=self.dv, axis=1)

    def current(self, f: np.ndarray):
        r"""
        First velocity moment :math:`j(x) = \int v f(x, v) dv`.
        """
        return _trapz(f * self.v[None, :], dx=self.dv, axis=1)

    def electric_field(self, f: np.ndarray):
        r"""
        Self-consistent electric field :math:`E` generated by the distribution ``f``.
        """
        return self.solve_poisson(self.density(f) - self.rho_ion)

    def build_control(self, action: Union[float, np.ndarray]):
        r"""
        Expands an action into the nodal values of the external field :math:`H(x_i)`.

        :param action: The action produced by the agent, of length ``action_dim``.
        """
        action = np.atleast_1d(np.asarray(action, dtype=np.float64)).ravel()
        if action.shape[0] != self.action_dim:
            raise Exception(
                f"Action of length {action.shape[0]} does not match the action dimension {self.action_dim} of this environment."
            )
        return self.control_basis @ self.normalize(action, self.max_control_value)

    def _advance(self, f: np.ndarray, H: np.ndarray):
        r"""
        One Strang split semi-Lagrangian step of the Vlasov-Poisson system.

        The scheme is unconditionally stable in the CFL sense because each substage is an
        exact characteristic solve followed by linear interpolation.
        """
        dt = self.dt
        order = self.interpolation
        # Half step of free streaming in x.
        f = _periodic_shift(f, self.v * (0.5 * dt) / self.dx, order)
        # Full step of acceleration in v using the field of the half-advected state.
        E = self.electric_field(f)
        f = _bounded_shift(f, (E + H) * dt / self.dv, order)
        # Half step of free streaming in x.
        f = _periodic_shift(f, self.v * (0.5 * dt) / self.dx, order)
        return f, E

    def get_observation(self, f: np.ndarray):
        r"""
        Builds the observation from the current distribution, applying sensing noise.

        :param f: The current distribution :math:`f(x, v)`.
        """
        df = f - self.fbar
        match self.sensing_type:
            case "full":
                obs = df
            case "density":
                obs = self.density(df)
            case "field":
                obs = self.solve_poisson(self.density(df))
            case "moments":
                obs = np.stack(
                    [
                        self.density(df),
                        self.current(df),
                        _trapz(df * self.v[None, :] ** 2, dx=self.dv, axis=1),
                    ]
                )
        return self.sensing_noise_func(obs)

    def step(self, action: Union[float, np.ndarray]):
        r"""
        step

        Holds ``action`` fixed and advances the Vlasov-Poisson system by
        ``control_sample_rate`` time units at the solver resolution ``dt``.

        :param action: The external field, either as modal coefficients or as nodal
            values depending on ``action_type``.
        """
        H = self.build_control(action)
        sample_rate = max(1, int(round(self.control_sample_rate / self.dt)))
        i = 0
        while i < sample_rate and self.time_index < self.nt - 1:
            self.f_current, E = self._advance(self.f_current, H)
            self.time_index += 1
            if self.store_history:
                self.f[self.time_index] = self.f_current
            i += 1

        terminate = self.terminate()
        truncate = self.truncate()
        df = self.f_current - self.fbar
        history = self.f if self.store_history else self.f_current[None, :, :]
        history_index = self.time_index if self.store_history else 0
        reward = self.reward_class.reward(
            history, history_index, terminate, truncate, H
        )
        return (
            self.get_observation(self.f_current),
            reward,
            terminate,
            truncate,
            self.diagnostics(H),
        )

    def diagnostics(self, H: np.ndarray):
        r"""
        Conserved and actuated quantities, returned in the ``info`` dict of every step.

        The Vlasov flow is measure preserving in phase space no matter what :math:`H` is,
        so ``mass`` and every Casimir of :math:`f` are invariants that no control can
        move. Momentum and total energy, by contrast, are driven purely by the control,

        .. math::

            \frac{d}{dt} \int\int v f = \int H \rho dx, \qquad
            \frac{d}{dt} \left( \frac{1}{2}\int\int v^2 f + \frac{1}{2}\int E^2 \right)
            = \int H j dx,

        since the self-consistent field contributes nothing to either budget. The two
        ``_rate`` entries are the right hand sides above and are exactly the two scalar
        directions in which the actuator has unobstructed authority.

        :param H: The external field applied over the step, on the spatial grid.
        """
        f = self.f_current
        E = self.electric_field(f)
        rho = self.density(f)
        j = self.current(f)
        df = f - self.fbar
        return {
            "H": H,
            "E": E,
            "mass": _trapz(rho, dx=self.dx),
            "momentum": _trapz(j, dx=self.dx),
            "kinetic_energy": 0.5
            * _trapz(_trapz(f * self.v[None, :] ** 2, dx=self.dv, axis=1), dx=self.dx),
            "electric_energy": 0.5 * _trapz(E**2, dx=self.dx),
            "momentum_rate": _trapz(H * rho, dx=self.dx),
            "energy_rate": _trapz(H * j, dx=self.dx),
            "l2_perturbation": np.sqrt(
                _trapz(_trapz(df**2, dx=self.dv, axis=1), dx=self.dx)
            ),
        }

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
        r"""
        truncate

        Determines whether to truncate the episode based on the size of the perturbation
        :math:`\delta f` and the variable ``limit_pde_state_size`` given in the PDE
        environment initialization.
        """
        if self.limit_pde_state_size and (
            np.linalg.norm(self.f_current - self.fbar, 2) >= self.max_state_value
            or not np.all(np.isfinite(self.f_current))
        ):
            return True
        else:
            return False

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        reset

        :param seed: Allows a seed for initialization of the envioronment to be set for RL algorithms.
        :param options: Allows a set of options for the initialization of the environment to be set for RL algorithms.

        Resets the PDE at the start of each environment according to the parameters given during the PDE environment intialization
        """
        try:
            init_condition = self.reset_init_condition_func(self.x, self.v)
            fbar = self.equilibrium_func(self.x, self.v)
        except:
            raise Exception(
                "Please pass both an initial condition and an equilibrium function in the parameters dictionary. See documentation for more details"
            )
        expected = (self.nx, self.nv)
        if init_condition.shape != expected or fbar.shape != expected:
            raise Exception(
                f"The initial condition and equilibrium must both have shape {expected}. See documentation for more details"
            )
        self.fbar = np.asarray(fbar, dtype=np.float64)
        self.dvfbar = np.gradient(self.fbar, self.dv, axis=1)
        self.f_current = np.asarray(init_condition, dtype=np.float64).copy()
        if self.store_history:
            self.f = np.zeros((self.nt, self.nx, self.nv))
            self.f[0] = self.f_current
        else:
            self.f = None
        self.time_index = 0
        self.reward_class.reset()
        return self.get_observation(self.f_current), self.diagnostics(np.zeros(self.nx))

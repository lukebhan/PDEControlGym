.. _braintumor:


Brain Tumor PDE
================

This documentation describes the Brain Tumor DPR (Diffusion-Proliferation-Radiation) Environment, which models the growth dynamics of glioblastomas–a fast-growing type of brain cancer–and their response to external beam radiation therapy (XRT). The model is governed by a partial differential equation (PDE) that includes three key terms: diffusion of tumor cells, proliferation, and radiation-induced cell death.

While tumor growth naturally occurs in three spatial dimensions and time, the model assumes spherical symmetry, allowing the simulation to be reduced to a one-dimensional radial domain over time. To further ease computation, the PDE is nondimensionalized, rescaling spatial, temporal, and concentration variables into a unitless form. As a result, the tumor’s evolving state can be represented as a one-dimensional spatial array indexed by time.

The numerical methods and modeling framework are based on the methods proposed in [1]_ and [2]_.


Background
------------------------

Glioblastoma is the most common malignant primary brain tumor, accounting for approximately 16% of all primary brain tumors [3]_. In contrast, the most common primary brain tumor overall–meningioma–makes up over 30% of primary brain tumor cases but tend to be benign and slow-growing [4]_. 

Glioblastomas are fast-growing and highly aggressive, with a median survival time of only 15 months following diagnosis [3]_. Current standard-of-care involves a uniform, protocol-driven approach involving surgical resection (where feasible), external beam radiation therapy (XRT), and concurrent chemotherapy. Magnetic Resonance Imaging (MRI) is used throughout treatment to monitor tumor progression and response to therapy.

Despite the known variability in tumor growth dynamics across patients, clinical radiation therapy regimens remain largely standardized. This “one-size-fits-all” approach limits the ability to tailor treatment to individual tumor dynamics. Computational models such as the DPR framework aim to simulate tumor evolution under different protocols, offering a pathway toward more personalized therapy strategies.



Dimensional Model
------------------------

Let :math:`\mathcal{B} = [0, L]` denote the one-dimensional spatial domain of the brain, where :math:`x=0` corresponds to the tumor center and :math:`x=L` is the outer boundary of the simulation domain. This domain models radial tumor expansion under the assumption of spherical symmetry. 

Let :math:`\mathcal{T} = [0, T]` denote the time domain of the simulation. Let :math:`\mathcal{T}_{\text{therapy}} \subset \mathcal{T}` represent the set of discrete time points at which radiation therapy is administered. The tumor concentration :math:`c(x, t)` evolves according to the following PDE:

.. math::

   \begin{aligned}
   c_t(x, t) &= \overbrace{D\nabla^2 c}^{\text{diffusion}} + \overbrace{\rho c}^{\text{proliferation}} - \overbrace{R(\alpha, \beta, d(x, t), t) c}^{\text{radiation}}\,,  \quad && (x, t) \in \mathcal{B} \times \mathcal{T} \\
   c(x, 0) &= c_0\,, \quad && x \in \mathcal{B}  \\ 
   \mathbf{n} \cdot \nabla c(x, t) &=  0\,,  \quad &&  (x, t) \in \partial \mathcal{B} \times \mathcal{T}
   \end{aligned}

where a zero-flux (Neumann) boundary condition is imposed at the spatial boundaries to ensure that no tumor cells enter or leave the domain during the simulation. :math:`\nabla c` is the spatial gradient of the tumor concentration and :math:`\mathbf{n}` is the outward-pointing unit normal vector at the boundary :math:`\partial \mathcal{B}` of the spatial domain.

Radiation term :math:`R` comes from:

.. math::
   R(\alpha, \beta,  d(x, t), t) =
   \begin{cases}
   1 - S(\alpha, \beta, d(x, t)), & \text{if } t \in \mathcal{T}_{\text{therapy}} \subset \mathcal{T} \\ 
   0, & \text{otherwise}
   \end{cases}

.. math::
  \begin{aligned}
   S(\alpha, \beta, d(x, t)) = e^{-\alpha \cdot \gamma(\alpha, \beta, d(x, t))} \\
   \gamma(\alpha, \beta, d(x, t)) = n \cdot d(x,t) \cdot \left( 1 + \frac{d(x, t)}{\alpha / \beta} \right)
  \end{aligned}


**Definitions:**

- :math:`D`: diffusion coefficient in units :math:`(mm^2/year)`
- :math:`\rho`: proliferation rate in units :math:`(1/year)`
- :math:`c_0`: initial distribution 
- :math:`d(x, t)`: fractionated radiation dose in Gray :math:`(Gy = J/kg)` applied at spatial location :math:`x` and time :math:`t`. This value represents the portion of the total prescribed dose delivered at each treatment instance. It reflects the real-world approach of splitting radiation into smaller daily doses rather than applying a full dose at once. Dose distribution varies spatially depending on the tumor's size, and temporally based on treatment schedule
- :math:`\alpha`: radio-sensitivity parameter in :math:`(Gy^{-1})` that qualifies probability of immediate (single-hit) cell death caused by radiation. In the linear-quadratic model for radiation efficacy, this term governs the linear component of damage
- :math:`\beta`: radio-sensitivity parameter in :math:`(Gy^{-2})` that qualifies probability of quadratic (double-hit) cell death, where two radiation-induced sub-lethal damages combine to cause cell death. In the linear-quadratic model for radiation efficacy, this term governs the quadratic component of damage
- The function :math:`\gamma(\cdot)` represents the biologically effective dose (BED) which scales the efficacy of radiation treatment by number of fractions given n, :math:`\alpha`, and :math:`\beta`
- The function :math:`S(\cdot)` represents the probability a tumor cell survives given that it received radiation dose :math:`d`
- The function :math:`R(\cdot)` represents the probability a tumor cell dies given that it received radiation dose :math:`d`

Nondimensional Model
------------------------

To represent the spatial domain as a one-dimensional linear array instead of a radial slice of a sphere, we can nondimensionalize the model presented above by introducing nondimensional quantities:

- :math:`\bar{x} = x / L`
- :math:`\bar{t} = \rho \cdot t`
- :math:`\bar{c} = (cL^3) / c_0`

where L represents the length of the one-dimensional spatial domain, :math:`c_0` represents the initial number of tumor cells and works as an arbitrary scaling term. This yields the following PDE:

.. math::

  \begin{aligned}
    \bar{c}_t(\bar{x}, \bar{t}) &= D^* \nabla^2 \bar{c} +  \bar{c} - R^*(\alpha, \beta, d(\bar{x}, \bar{t}), \bar{t}) \bar{c}\,,  \quad && (\bar{x}, \bar{t}) \in [0, 1] \times [0, \bar{T}] \\
    \bar{c}(\bar{x}, 0) &= L^3 e^{-100 \bar{x}^2}\,, \quad && \bar{x} \in [0, 1]  \\ 
  \end{aligned}

where quantities :math:`D^*` and :math:`R^*` are given by:

.. math::
   D^* = D/(\rho L^2)

.. math::
   R^*(\alpha, \beta,  d(\bar{x}, \bar{t}), \bar{t}) =
   \begin{cases}
   (1 - S(\alpha, \beta, d(\bar{x}, \bar{t}))) / \rho, & \text{if } \bar{t} \in \bar{\mathcal{T}}_\text{therapy} \subset \bar{\mathcal{T}} \\ 
   0, & \text{otherwise}
   \end{cases}


Model Implementation Details
------------------------

Glioblastoma diagnosis and treatment monitoring are typically performed using two types of MRI scans: gadolinium enhanced T1-weighted and T2-weighted imaging (referred to here as T1 and T2, respectively). While these scans do not directly measure tumor cell density, modeling literature based on heuristic thresholds infer that [5]_:

- The T1 region corresponds to areas of high tumor cell density, typically >80% of the carrying capacity
- The T2 region corresponds to areas of moderate tumor cell density, typically >16% of the carrying capacity

Using these thresholds, our model is capable of generating virtual MRI predictions allowing us to track T1 and T2 tumor radii dynamically over time and spatially define dose regions during radiation therapy administration.

To simulate the full treatment cycle–diagnosis, therapy, and follow-up–we allow the virtual tumor to grow until the T1 region reaches a radius of 10-20mm, triggering the onset of therapy. We then apply radiation according to a predefined schedule, followed by a post-treatment simulation period of 100 days of free growth.


Numerical Implementation
------------------------


The external beam radiation therapy schedule follows the standard protocol adopted by the University of Washington Medical Center, and is divided into two sequential stages:

- Stage 1: 28 days of 1.8 :math:`(Gy)` each, delivered to the T2 region + a 25 mm margin
- Stage 2: 6 days of 1.8 :math:`(Gy)` each, delivered to the T1 region + a 20 mm margin

This results in a total dose of 61.2 :math:`(Gy)` delivered over 34 days of treatment.
Radiation is administered on a 5-days-on, 2-days-off schedule to account for weekend breaks.


Nondimensional Model
------------------------

We derive the numerical implementation scheme for those looking for inner details of the environment. We use a first-order finite-difference scheme to approxiate the dimensionless PDE:

.. math::
  \bar{c}_t(\bar{x}, \bar{t}) &= D^* \nabla^2 \bar{c} +  \bar{c} - R^*(\alpha, \beta, d(\bar{x}, \bar{t}), \bar{t}) \bar{c} \\

Consider the first-order taylor approximation as

.. math::
  \bar{c}(\bar{x}, \bar{t}+1) = \bar{c}(\bar{x}, \bar{t}) + \Delta \bar{t} \cdot \bar{c}_t(\bar{x}, \bar{t})

with finite spatial derivatives approximated by first-order centered differences

.. math::
  \frac{\partial^2 \bar{c}}{\partial \bar{x}^2} = \frac{\bar{c}_{j+1}^n - 2\bar{c}_j^n+\bar{c}_{j-1}^n}{(\Delta \bar{x})^2}

where :math:`\Delta \bar{t}=d\bar{t}=\text{dimensionless time step}`, :math:`\Delta \bar{x}=d\bar{x}=\text{dimensionless spatial step}`, :math:`n=0, ..., Nt`, :math:`j=0, ..., Nx`, where :math:`Nt` and :math:`Nx` are the total number of discretized temporal and spatial steps respectively. Substituting into our original equation yields

.. math::
  \bar{c}_{j}^{n+1} = \bar{c}_{j}^{n} + \Delta \bar{t} (D^* \cdot (\frac{\bar{c}_{j+1}^n - 2\bar{c}_j^n+\bar{c}_{j-1}^n}{(\Delta \bar{x})^2}) + \bar{c}_{j}^n - R^* \bar{c}_{j}^n)

The last thing to consider is the boundary conditions for finding :math:`\bar{c}_{j}^{n+1}` when :math:`j = 0` or :math:`j = Nx`. In these cases, we set :math:`\bar{c}_{-1}^{n} = \bar{c}_{1}^{n}` and :math:`\bar{c}_{Nx+1}^{n} = \bar{c}_{Nx-1}^{n}` respectively to create a symmetric and mirrored concentration field across the boundary to satisfy the no-flux boundary condition.

References
------------------------

.. [1] Rockne, R. et al. *A mathematical model for brain tumor response to radiation therapy*, Journal of Mathematical Biology, 2009.

.. [2] Rockne, R. et al. *Predicting efficacy of radiotherapy in individual glioblastoma patients in vivo: a mathematical modeling approach*, Physics in Medicine & Biology, 2010.

.. [3] Tamimi AF, Juweid M. *Epidemiology and Outcome of Glioblastoma* In: De Vleeschouwer S, editor. **Glioblastoma**. Codon Publications; 2017.

.. [4] John Hopkins Medicine. *Brain Tumor Types*.

.. [5] Swanson, KR. et al. *A mathematical modelling tool for predicting survival of individual patients following resection of glioblastoma: a proof of principle*, British Journal of Cancer, 2007.

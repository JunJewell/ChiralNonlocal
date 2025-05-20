# ChiralNonlocal

Code to simulate a two-species 2D nonlocal reaction–advection–diffusion system with chiral movement. See the paper [[here]] for details of the model and theory. Videos showing example simulations are also included.

---

## Videos

The `Videos` folder contains animations of evolving density heatmaps corresponding to figures in the paper. Each video shows:

- **Left**: The densities of both species superimposed, using the colourmap shown in `colour_map.pdf`.
- **Centre**: The density of the chaser species, $c$, normalised to show the minimum and maximum density at each time point.
- **Right**: The density of the runner species, $\rho$, again normalised for visibility.

Note that video playback speed may vary, but simulation time is always shown at the top centre as `"Time="`.

Videos are named after the corresponding figures in the paper. For example:
- **Fig. 4e,f**: Comparison of oscillations (parallel chase-and-run) vs steady state (chiral chase-and-run).
- **Fig. 5**: Simulation where linear theory predicts early-time oscillations but these later settle to a steady state.
- **Fig. 6**: Species mixing and separation at early and late times.
- **Fig. 7**: Dynamics with no cross-species interaction, meaning that the species are only coupled via volume-filling.
- **Fig. 8**: Population chase-and-run dynamics for different chiral running angles.
- **Fig. 9a,b**: Travelling holes, without and with chirality, respectively.

---

## Code Structure

### Simulation

- `parent_simulation_class.py`: Abstract base class for simulating systems with nonlocality, diffusion, and/or advection on a 2D periodic domain. Daughter classes define specifics such as chirality, sensing mode, and governing equations.
- `chiral_directsensing_packing_NAD_class.py`: Daughter class for chiral, direct sensing, volume-filling systems.
- `chiral_gradientsensing_packing_NAD_class.py`: Daughter class for chiral, gradient sensing, volume-filling systems.

To run a simulation, create an instance of one of the daughter classes with the desired parameters, and call either `simulate` or `simulate_with_progress` (which saves results to CSV).

### Visualisation

- `dispersion_relation.py`: Computes and plots dispersion relations derived from linear stability analysis.
- `plot_class.py`: Provides methods to visualise and save simulation results, including heatmaps and animations.
- `matplotlib_style.py`: Defines consistent plotting style.

### Execution Examples

- `simulate_chiral_directsensing_packing_NAD.py`: Example script for chiral direct sensing systems.
- `simulate_chiral_gradientsensing_packing_NAD.py`: Example for chiral gradient sensing systems.
- `read_data_and_plot.py`: Reads simulation data and generates plots.
- `read_and_simulate_further.py`: Continues a simulation from saved data.
- `concatenate_data.py`: Combines multiple simulation output files for simulations that were continued from saved data.

---


## Numerical Integration
Numerical integration of integro-PDEs. such as
$$\frac{\partial u}{\partial t} = \nabla^2 u -\boldsymbol{\nabla}\cdot\left(u(1-u)\frac{\mu}{\xi^{2}} \int\int\boldsymbol{\hat{s}}\tilde{\Omega}\left(\frac{s}{\xi}\right)u(\boldsymbol{x}+\boldsymbol{s},t) d s_x d s_y\right),$$
and its 2 species and chiral equivalents, is carried out using the method-of-lines, first discretising in space and then integrating the resulting ODEs with backwards differentiation formulae. The latter is implemented through SciPy's `integrate.solve_ivp` function - see [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) for details. For diffusion and advection terms, standard centred stencils are used throughout. Chirality is included simply by rotating the vector of advection using a matrix multiplication. The integral term is calculated using a fast Fourier transform method. Further detail can be found [in this repository](https://github.com/JunJewell/NonlocalReactAdvectDiffuse2D), which uses the same underlying method.

# FiniteVolumeGPU

This Python software package implements several finite volume discretizations on Cartesian grids for the shallow water equations and the Euler equations. 

## Setup
A good place to start exploring this codebase is the notebooks. Complete the following steps to run the notebooks:

1. Install `conda` (e.g. [Miniforge][mf] or [Anaconda][ac])
2. Change directory to the repository root and run the following commands
3. `conda env create -f conda_environment.yml`
4. `conda activate ShallowWaterGPU`
5. `jupyter notebook`

Make sure you are running the correct kernel ("conda:ShallowWaterGPU"). If not, change kernel using the "Kernel"-menu in the notebook.

## Troubleshooting
Have a look at the conda documentation and https://towardsdatascience.com/how-to-set-up-anaconda-and-jupyter-notebook-the-right-way-de3b7623ea4a

## Benchmarking for efficiency and accurancy of numerical schemes
1. Follow [Setup](#setup) above.
1. Run `make plots`. This will:
    1. Generate all the solution files by calling `benchmark_simulate.py`
    1. Call `benchmark_postprocess.py` to calculate the errors, execution time, and peak performance of the solutions.
    1. Run `papermill benchmark_plotter.ipynb` with each initial condition to generate figures in corresponding notebooks `benchmark_plots_*.ipynb` as well as PDFs in the `figures/` directory.

### Benchmarking files
- `benchmark_simulate.py`: Runs a single simulation and records the solution, time taken, and other data.
- `benchmark_postprocess.py`: Reads all the simulation files and calculates performance and accuracy.
- `benchmark_plotter.ipynb`: Jupyter notebook that creates and exports all plots. Called through `papermill` when `make plots` is run to generate `benchmark_plots_bump.ipynb` and `benchmark_plots_dambreak.ipynb`.
- `benchmark_common.py`: Some functions used in multiple files.
- `misc_plotting.py`: Some miscellaneous functions related to the plotting.

### To implement a new simulator
`GPUSimulators/WAF.py` is derived from the base class in `GPUSimulators/Simulators.py`. It relies on a kernel in `GPUSimulators/cuda/SWE2D_WAF.cu`.
To create your own, implement your versions of those two files (keep the naming conventions the same because the `makefile` relies on that.), and then update each of the above benchmarking files so that they are aware of these new classes.
Don't forget the `GetSimulator` class in `benchmark_common.py`.
Finally, mention the simulator name in the `makefile`'s array `simulators` at the top of the file.

### To implement a new test case/initial condition
Implement it as a function in `GPUSimulators/helpers/InitialCondition.py`.
Then add the corresponding dictionary entry to `GetInitialCondition.ics` dictionary in `benchmark_common.py`.
Finallly, update the `all` target in the `makefile` with the desired notebook.

[mf]:https://github.com/conda-forge/miniforge
[ac]:https://www.anaconda.com/

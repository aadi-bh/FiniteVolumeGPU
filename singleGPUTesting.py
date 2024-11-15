# -*- coding: utf-8 -*-

"""
This python module implements simulations for benchmarking

Copyright (C) 2018  SINTEF ICT

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np
import gc
import logging
import os

# CUDA
import pycuda.driver as cuda

# Simulator engine etc
from GPUSimulators import Common, CudaContext
from GPUSimulators import EE2D_KP07_dimsplit
from GPUSimulators.helpers import InitialConditions as IC
from GPUSimulators.Simulator import BoundaryCondition as BC

import argparse
parser = argparse.ArgumentParser(description='Single GPU testing.')
parser.add_argument('-nx', type=int, default=128)
parser.add_argument('-ny', type=int, default=128)


args = parser.parse_args()

####
# Initialize logging
####
log_level_console = 20
log_level_file = 10
log_filename = 'single_gpu.log'
logger = logging.getLogger('GPUSimulators')
logger.setLevel(min(log_level_console, log_level_file))

ch = logging.StreamHandler()
ch.setLevel(log_level_console)
logger.addHandler(ch)
logger.info("Console logger using level %s",
            logging.getLevelName(log_level_console))

fh = logging.FileHandler(log_filename)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s: %(message)s')
fh.setFormatter(formatter)
fh.setLevel(log_level_file)
logger.addHandler(fh)
logger.info("File logger using level %s to %s",
            logging.getLevelName(log_level_file), log_filename)


####
# Initialize CUDA
####
cuda.init(flags=0)
logger.info("Initializing CUDA")
cuda_context = CudaContext.CudaContext(autotuning=False)


####
# Set initial conditions
####
logger.info("Generating initial conditions")
nx = args.nx
ny = args.ny

gamma = 1.4
roughness = 0.125
save_times = np.linspace(0, 0.5, 10)
outfile = "single_gpu_out.nc"
save_var_names = ['rho', 'rho_u', 'rho_v', 'E']

arguments = IC.genKelvinHelmholtz(nx, ny, gamma)
arguments['context'] = cuda_context
arguments['theta'] = 1.2


####
# Run simulation
####
logger.info("Running simulation")
# Helper function to create MPI simulator


def genSim(**kwargs):
    local_sim = EE2D_KP07_dimsplit.EE2D_KP07_dimsplit(**kwargs)
    return local_sim


outfile = Common.runSimulation(
    genSim, arguments, outfile, save_times, save_var_names)

####
# Clean shutdown
####
local_sim = None
cuda_context = None
arguments = None
logging.shutdown()
gc.collect()



####
# Print completion and exit
####
print("Completed!")
exit(0)

# Simulation calling script

# imports
#Import packages we need
import numpy as np
from matplotlib import animation, rc
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import mpld3

import subprocess
import socket
import sys
import time
import os
import gc
import datetime
import logging

import pycuda.driver as cuda

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from GPUSimulators import Common, LxF, FORCE, HLL, HLL2, KP07, KP07_dimsplit, WAF
from GPUSimulators import CudaContext
import atexit
from GPUSimulators.helpers import InitialConditions

import argparse
# collect arguments
    # simulator
    # initial conditions
    # domain size, ref_size
    # force_rerun
    # transpose
    # open-ended if possible for quick hacking
ref_nx = 8192 # *4
ref_ny = ref_nx
width = 50
height = width
bump_size = 10
logger = logging.getLogger(__name__)
cfl_scale = 0.9
g = 9.81
g = 9.81
H_REF = 0.5
H_AMP = 0.1
U_REF = 0.0
U_AMP = 0.1
force_rerun = True
transpose = False

# Reference solution computed by just running simulators on finer meshes
# Must be factors of `ref_nx` for downsampling to work
domain_sizes = [8]#, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
simulators = [LxF.LxF]#, FORCE.FORCE, HLL.HLL, HLL2.HLL2, KP07.KP07, KP07_dimsplit.KP07_dimsplit, WAF.WAF]


def init_logger(name, outfile, print_level=1, file_level=10):
    logger = logging.getLogger(name)
    logger.setLevel(min(print_level, file_level))

    ch = logging.StreamHandler()
    ch.setLevel(print_level)
    logger.addHandler(ch)
    logger.log(print_level, "Console logger using level %s", logger.level)

    logger.log(file_level, "File logging level something")

    fh = logging.FileHandler(outfile)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(file_level)
    logger.addHandler(fh)

    logger.info("Python version %s", sys.version)
    return logger

def create_cuda_context(name, blocking=False, no_cache=False, use_autotuning=True):
    context_flags = None
    if (blocking):
        context_flags = cuda.ctx_flags.SCHED_BLOCKING_SYNC
    logger = logging.getLogger(__name__)
    logger.debug("Creating context")
    context = CudaContext.CudaContext(context_flags=context_flags, 
                                     use_cache=not no_cache, 
                                     autotuning = use_autotuning)
    def exitfunc():
        logger.info("Exitfunc: Resetting CUDA context stack")
        while (cuda.Context.get_current() != None):
            context = cuda.Context.get_current()
            logger.info("Popping <%s>", str(context.handle))
            cuda.Context.pop()
        logger.debug("===")
    atexit.register(exitfunc)
    return context
    
def gen_filename(simulator, nx, ny):
    ic = 'smooth1d'
    return os.path.abspath(os.path.join("data", ic, str(simulator.__name__) + "_" + str(nx) + "_" + str(ny) + ".npz"))


def run_benchmark(datafilename, simulator, simulator_args, nx, reference_nx, ny, reference_ny,
                  h_ref=0.5, h_amp=0.1, u_ref=0.0, u_amp=0.1, v_ref=0.0, v_amp=0.1,
                  dt=None, tf=1.0, max_nt=np.inf, force_rerun=False, transpose=False):
    if (datafilename and os.path.isfile(datafilename) and force_rerun == False):
        print(f"WARNING: Previous simulation found, skipping simulation run for {simulator.__name__} on {nx} cells")
        logger.info("Skipping  simulation because previous simulation found for #TODO.")
        return [0, 0, 0]
    else:
        width = 100
        test_data_args = {
            'nx': nx,
            'ny': ny,
            'width': 100,
            'height': 100,
            'bump_size': 20,
            'ref_nx': reference_nx,
            'ref_ny': reference_ny,
            'h_ref': h_ref, 'h_amp': h_amp,
            'u_ref': u_ref, 'u_amp': u_amp,
            'v_ref': v_ref, 'v_amp': v_amp
        }
        h0, hu0, hv0, dx, dy = InitialConditions.bump(**test_data_args)

        # Initialise simulator
        with Common.Timer(simulator.__name__ + "_" + str(nx)) as timer:
            if (transpose):
                h0 = np.ascontiguousarray(np.transpose(h0))
                hu0, hv0 = np.ascontiguousarray(np.transpose(hv0)), np.ascontiguousarray(np.transpose(hu0))
                dx, dy = dy, dx
                nx, ny = ny, nx

            sim_args = {
                'h0': h0, 'hu0': hu0, 'hv0': hv0,
                'nx': nx, 'ny': ny,
                'dx': dx, 'dy': dy
            }
            sim_args.update(simulator_args)

            sim = simulator(**sim_args)
            # final time, number of steps, and computing time reported by pycuda
            t,  nt, elapsed_time = sim.simulate(tf, max_nt, dt=None)
            sim.check()

            nt = sim.simSteps()
            t = sim.simTime()
            dt = sim.simTime() / nt
            h, hu, hv = sim.download()

            if (transpose):
                h = np.ascontiguousarray(np.transpose(h))
                hu, hv = np.ascontiguousarray(np.transpose(hv)), np.ascontiguousarray(np.transpose(hu))

            if (datafilename):
                dirname = os.path.dirname(datafilename)
                if (dirname and not os.path.isdir(dirname)):
                    os.makedirs(dirname)
                np.savez_compressed(datafilename, h=h, hu=hu, hv=hv)
    gc.collect() # Force garbage collection
    return [t, nt, elapsed_time]

if __name__ == "__main__":
    logger = init_logger(__name__, 'badname.log')
    ctx = create_cuda_context('nonamestoday')
    sim_args = {
    'context': ctx,
    'g': g,
    'cfl_scale': cfl_scale
    }
    # run simulator
        # this is going to be a bit harder than we think, just to get the simulator working
# Make space to store
    sim_elapsed_time = np.zeros((len(simulators), len(domain_sizes)))
    sim_dt = np.zeros_like(sim_elapsed_time)
    sim_nt = np.zeros_like(sim_elapsed_time)


    for i in range(len(simulators)):
        # Run reference with a low CFL-number. TODO IT DOES NOT!
        # This should also serve as warmup for now? TODO
        # warmup!
        datafilename = gen_filename(simulators[i], ref_nx, ref_ny)
        _, _, secs = run_benchmark(datafilename, 
                          simulators[i],
                          sim_args,
                          16, 16, 16, 16,
                          h_ref=H_REF, h_amp=H_AMP,
                          u_ref=U_REF, u_amp=U_AMP,
                          v_ref=0.0, v_amp=0.0,
                          tf = np.inf, max_nt = 1,
                          #dt=0.25*0.7*(width/ref_nx)/(u_ref+u_amp + np.sqrt(g*(h_ref+h_amp))),
                          force_rerun=True,
                          transpose=transpose)
        logger.info(f"{simulators[i].__name__} completed warmup simulation in {secs}s.")

        # Run on all the sizes
        for j, (nx, ny) in enumerate(zip(domain_sizes, domain_sizes)):
            datafilename = gen_filename(simulators[i], nx, ny)
            t, nt, secs = run_benchmark(datafilename, 
                          simulators[i],
                          sim_args,
                          nx, ref_nx, ny, ref_ny,
                          h_ref=H_REF, h_amp=H_AMP,
                          u_ref=U_REF, u_amp=U_AMP,
                          v_ref=0.0, v_amp=0.0,
                          tf = 1.0, max_nt = np.inf,
                          #dt=0.25*0.7*(width/ref_nx)/(u_ref+u_amp + np.sqrt(g*(h_ref+h_amp))),
                          force_rerun=force_rerun,
                          transpose=transpose)
            logger.info(f"[{simulators[i].__name__} {nx}x{ny} done in {secs}s.")

            # store
            sim_elapsed_time[i, j] = secs
            sim_dt[i, j] = t / nt
            sim_nt[i, j] = nt
            

    # _ to prevent output being printed to the screen
    _ = gc.collect()
# write arguments and results
    # what format are the results going to be in?
    # npz is fine for those, but then there needs to be metadata about max_nt, etc.

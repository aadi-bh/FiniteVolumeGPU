# Simulation calling script

# imports
#Import packages we need
import numpy as np
import sys
import os
import gc
import logging
import argparse
import atexit

import pycuda.driver as cuda

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from GPUSimulators import Common, LxF, FORCE, HLL, HLL2, KP07, KP07_dimsplit, WAF
from GPUSimulators import CudaContext
from GPUSimulators.helpers import InitialConditions

def init_logger(name, outfile, print_level=1, file_level=10):
    logger = logging.getLogger(name)
    logger.setLevel(min(print_level, file_level))

    ch = logging.StreamHandler()
    ch.setLevel(print_level)
    logger.addHandler(ch)
    logger.log(print_level, "Console logger using level %s", logger.level)

    logger.log(file_level, "File logging level %s", file_level)

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
    
def gen_filename(args, nx, ny):
    return os.path.abspath(os.path.join("data", str(args.ic.__name__), str(args.simulator.__name__) + "_" + str(nx) + "_" + str(ny) + ".npz"))


def run_benchmark(datafilename, simulator, simulator_args, ic, nx, reference_nx, ny, reference_ny,
                  dt=None, tf=1.0, max_nt=np.inf, force_rerun=False, transpose=False):
    if (datafilename and os.path.isfile(datafilename) and force_rerun == False):
        print(f"WARNING: Previous simulation found, skipping simulation run for {simulator.__name__} on {nx} cells")
        logger.info("Skipping  simulation because previous simulation found for #TODO.")
        return [0, 0, 0]
    else:
        test_data_args = {
            'nx': nx,
            'ny': ny,
            'ref_nx': reference_nx,
            'ref_ny': reference_ny,
            'num_ghost_cells': GetSimulator.num_ghost_cells[simulator.__name__]
        }
        # This will change according to the init
        test_data_args.update(GetInitialCondition.ics[ic.__name__]['test_data_args'])
        h0, hu0, hv0, dx, dy, = ic(**test_data_args)

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
                np.savez_compressed(datafilename, h=h, hu=hu, hv=hv, t=t, nt=nt, elapsed_time=elapsed_time, test_data_args=test_data_args)
    gc.collect() # Force garbage collection
    return [t, nt, elapsed_time]

class GetSimulator(argparse.Action):
    simulators = {'LxF': LxF.LxF,
                  'FORCE': FORCE.FORCE,
                  'HLL': HLL.HLL,
                  'HLL2': HLL2.HLL2,
                  'KP07': KP07.KP07,
                  'KP07_dimsplit': KP07_dimsplit.KP07_dimsplit,
                  'WAF': WAF.WAF,
                  }
    def __call__(self, _, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.simulators[values])
    num_ghost_cells = {'LxF': 1,
                       'FORCE': 1,
                       'HLL': 1,
                       'HLL2': 2,
                       'KP07': 2,
                       'KP07_dimsplit': 2,
                       'WAF': 2}

class GetInitialCondition(argparse.Action):
    ics = {'bump': {'fn': InitialConditions.bump,
#                    'tf': 1.0,
#                    'max_nt': 100, # for now
                    'test_data_args': {
                        'width': 100,
                        'height': 100
                    }
                    },
          'dambreak': {'fn': InitialConditions.dambreak,
                    'test_data_args': {
                        'width': 100,
                        'height': 100,
                        'damloc' : 0.5,
                    }
                  },
                  'constant': {'fn': InitialConditions.constant,
                               'test_data_args': {
                                   'width': 10,
                                   'height': 10,
                                   'constant': 1.0}
                  }
        }
    def __call__(self, _, namespace, values, option_string=None):
        data = self.ics[values]
        setattr(namespace, self.dest, data['fn'])
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a simulator")
    parser.add_argument('ic', choices=GetInitialCondition.ics.keys(), action=GetInitialCondition)
    parser.add_argument("simulator", choices=GetSimulator.simulators.keys(), action=GetSimulator)
    parser.add_argument('--cfl', type=float, default=0.9, required=False)
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--ref-nx', type=int, default=8192)
    parser.add_argument('--ny', type=int, default=2)
    parser.add_argument('--ref-ny', type=int, default=None)
    parser.add_argument('--force-rerun', action='store_true', default=True, help="Rerun simulation even if corresponding datafile exists")
    parser.add_argument('--transpose', action='store_true', default=False, help="run the transposed initial condition")
    parser.add_argument('--logfile', default='gpusimulator', help="Name of log file")
    parser.add_argument('--tf', type=float, default=None)
    parser.add_argument('--nt', type=int, default=None)

    logger = init_logger(__name__, 'gpusimulator.log')
    logger = logging.getLogger(__name__)

#    H_REF = 0.5
#    H_AMP = 0.1
#    U_REF = 0.0
#    U_AMP = 0.1
    args = parser.parse_args()
    if args.tf != None:
        args.nt = np.inf
    elif args.nt != None:
        args.tf = np.inf
    else:
        raise ValueError("Please specify at least one of --tf or --nt")
    logger.info("Arguments: " + str(args))

    ctx = create_cuda_context('nonamestoday')

    sim_args = {
    'context': ctx,
    'g': 9.81,
    'cfl_scale': args.cfl,
    }
    benchmark_args = {
        'simulator': args.simulator,
        'simulator_args': sim_args,
        'ic': args.ic,
        'force_rerun': args.force_rerun,
        'transpose': args.transpose,
        #dt=0.25*0.7*(width/ref_nx)/(u_ref+u_amp + np.sqrt(g*(h_ref+h_amp))),
    }
    # Make space to store
    # sim_elapsed_time = 0.0
    # sim_dt = 0.0
    # sim_nt = 0.0
        # Run reference with a low CFL-number. TODO IT DOES NOT!
        # This should also serve as warmup for now? TODO
        # warmup!
    _, _, secs = run_benchmark(datafilename = None,
                               **benchmark_args,
                               nx=min(16, args.nx), reference_nx=min(16, args.ref_nx),
                               ny=min(16, args.ny), reference_ny=min(16, args.ref_ny),
                               max_nt = 1, tf=np.inf)
    logger.info(f"{args.simulator.__name__} completed warmup simulation in {secs}s.")

        # Run on all the sizes
    datafilename = gen_filename(args, args.nx, args.ny)
    t, nt, secs = run_benchmark(datafilename = datafilename, 
                          **benchmark_args,
                          nx = args.nx, reference_nx = args.ref_nx,
                          ny = args.ny, reference_ny = args.ref_ny,
                          tf = args.tf, max_nt = args.nt)
    logger.info(f"[{args.simulator.__name__} {args.nx}x{args.ny}] done in {secs}s ({nt} steps).")
    gc.collect()
    sys.exit(0)

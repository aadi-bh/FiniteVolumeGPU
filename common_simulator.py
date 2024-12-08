import argparse
import logging
import sys
import os
import pycuda.driver as cuda
import atexit
import numpy as np

from GPUSimulators import Common, LxF, FORCE, HLL, HLL2, KP07, KP07_dimsplit, WAF
from GPUSimulators import CudaContext
from GPUSimulators.helpers import InitialConditions


def gen_filename(args, nx, ny, ic:str=None, simulator=None, prefix=None):
    if prefix == None:
        if args.nt == np.inf and args.tf < np.inf:
            prefix = 'space'
        elif args.nt < np.inf  and args.tf == np.inf:
            prefix = 'time'
        else:
            raise ValueError("gen_filename unable deduce the purpose of simulation.\n One and only one of args.nt and args.tf must be finite.")
    if args != None:
        if args.ic != None and ic == None:
            ic = args.ic.__name__
        if args.simulator != None and simulator == None:
            simulator = args.simulator.__name__

    directory = prefix + '_data'
    return os.path.abspath(os.path.join(directory, ic, simulator + "_" + str(nx) + "_" + str(ny) + ".npz"))

def gen_results_filename(kind:str, simulator:str, ic:str):
    return os.path.join(kind + '_data', 'results', ic, simulator + '.npz')

def init_logger(name, outfile, print_level=20, file_level=10):
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

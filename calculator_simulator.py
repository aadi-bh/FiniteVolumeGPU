import argparse
import numpy as np
import os
import glob
from common_simulator import *

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(10)
logger.addHandler(ch)

parser = argparse.ArgumentParser("Calculates benchmarking results given benchmarking data")
parser.add_argument('directory', choices=['space', 'time'], default=None)
parser.add_argument('ic', choices=GetInitialCondition.ics.keys(), action=GetInitialCondition)
parser.add_argument('simulator', choices=GetSimulator.simulators.keys(), action=GetSimulator)
parser.add_argument('--sizes', nargs='+', help="List of nx_ny")
parser.add_argument('--ref', nargs=1, help="Reference solution for erorr calculation")
args = parser.parse_args()

# Generate filenames
if args.sizes == None or len(args.sizes) == 0:
    directory = os.path.dirname(gen_filename(args, 0, 0, prefix=args.directory))
    unchecked_filenames = glob.glob(os.path.join(directory, args.simulator.__name__ + "_[0-9]*_[0-9]*.npz"))
    assert(len(unchecked_filenames) > 0)
    print(f"Found {len(unchecked_filenames)} files")
else:
    unchecked_filenames = [gen_filename(args, size_str.split('_')[0], size_str.split('_')[1], args.directory) for size_str in args.sizes]

# Check existence
filenames = []
for filename in unchecked_filenames:
    if os.path.isfile(filename):
        filenames.append(filename)
    else:
        logger.info(f"Skipping not-a-file: {filename}.")
if len(filenames) < len(unchecked_filenames):
    logger.info(f"Specified {len(unchecked_filenames)} files but found only {len(filenames)}")

ds_x = np.zeros(len(filenames))
ds_y = np.zeros_like(ds_x)
def gen_space_results(filenames):

def gen_time_results(filenames):
    longsim_elapsed_time = np.zeros_like(ds_x)
    max_nt = np.zeros_like(ds_x)
    for j, filename in enumerate(filenames):
        with np.load(filename) as data:
            ds_y[j], ds_x[j] = data['h'].shape
            max_nt[j] = data['nt']
            longsim_elapsed_time[j] = data['elapsed_time']
    
    assert(np.all(max_nt == max_nt[0]))
    secs_per_timestep = longsim_elapsed_time / max_nt
    megacells_per_sec = ds_x * ds_y * 10**-6 * max_nt / longsim_elapsed_time
    
    results_filename = gen_results_filename(args.directory, args.simulator.__name__, args.ic.__name__)
    if not os.path.isdir(os.path.dirname(results_filename)):
        os.makedirs(os.path.dirname(results_filename))
    np.savez_compressed(results_filename, ds_x=ds_x, ds_y=ds_y, 
                        secs_per_timestep=secs_per_timestep, 
                        megacells_per_sec=megacells_per_sec)
    print(f"Created {results_filename}")
import argparse
import numpy as np
import os
import glob
from common_simulator import *
import scipy

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(10)
logger.addHandler(ch)

parser = argparse.ArgumentParser("Calculates benchmarking results given benchmarking data")
parser.add_argument('directory', choices=['space', 'time', 'both'], default=None)
parser.add_argument('ic', choices=GetInitialCondition.ics.keys(), action=GetInitialCondition)
parser.add_argument('simulator', choices=GetSimulator.simulators.keys(), action=GetSimulator)
parser.add_argument('--ref', type=str, help="Reference solution for erorr calculation")
parser.add_argument('--sizes', nargs='+', help="List of nx_ny")
PEAK_PERFORMANCE_FROM_LAST = 3
args = parser.parse_args()

# Generate filenames
if args.sizes == None or len(args.sizes) == 0:
    directory = os.path.dirname(gen_filename(args, 0, 0, prefix=args.directory))
    unchecked_filenames = glob.glob(os.path.join(directory, args.simulator.__name__ + "_[0-9]*_[0-9]*.npz"))
    assert len(unchecked_filenames) > 0
    print(f"Found {len(unchecked_filenames)} files")
else:
    unchecked_filenames = [gen_filename(args, size_str.split('_')[0], size_str.split('_')[1], prefix = args.directory) for size_str in args.sizes]

# Check existence
filenames = []
for filename in unchecked_filenames:
    if os.path.isfile(filename):
        filenames.append(filename)
    else:
        logger.info(f"Skipping not-a-file: {filename}.")
if len(filenames) < len(unchecked_filenames):
    logger.info(f"Specified {len(unchecked_filenames)} files but found only {len(filenames)}")

def save_results(**kwargs):
    results_filename = gen_results_filename(kind = args.directory, simulator=args.simulator.__name__, ic = args.ic.__name__)
    if not os.path.isdir(os.path.dirname(results_filename)):
        os.makedirs(os.path.dirname(results_filename))
    np.savez_compressed(results_filename, **kwargs)
    print(f"Created {results_filename}")

def gen_space_results(filenames, ref_solution):
    with np.load(ref_solution) as d:
        ref_dx = d['dx']
        ref_dy = d['dy']
        ref_h = d['h']
    if ref_h.ndim == 1:
        ref_nx = ref_h.shape[0]
        ref_ny = 1
    elif ref_h.ndim == 2:
        ref_ny = ref_h.shape[0]
        ref_nx = ref_h.shape[1]
    assert ref_ny * ref_nx > 0

    ds_x = np.zeros(len(filenames))
    ds_y = np.zeros_like(ds_x)
    sim_errors = np.zeros_like(ds_x)
    sim_cons = np.ones_like(ds_x)
    sim_errors = np.zeros_like(ds_x)
    sim_nt = np.zeros_like(ds_x)
    sim_tf = np.zeros_like(ds_x)

    for j, filename in enumerate(filenames):
        with np.load(filename) as d:
            ds_y[j], ds_x[j] = d['h'].shape
            dx = d['dx']
            dy = d['dy']
            h = d['h']
            sim_nt[j] = d['nt']
            sim_tf[j] = d['tf']
        if h.ndim == 1:
            nx = h.shape[0]
            ny = 1
        elif h.ndim == 2:
            ny = h.shape[0]
            nx = h.shape[1]
        assert nx * ny > 0
        
        assert ref_nx >= nx
#        target_ny = min(ref_ny, ny)
        ref_h_downsampled = InitialConditions.downsample(ref_h, x_factor=ref_nx / nx, y_factor= ref_ny / ny) # int(ref_ny / target_ny))
#        h_downsampled = InitialConditions.downsample(h, x_factor=1.0, y_factor= int(ny / target_ny))

        sim_errors[j] = np.linalg.norm((ref_h_downsampled - h).flatten(), ord=1) * dx * dy # * (ny / target_ny)
        sim_cons[j] = (np.sum(ref_h) * ref_dx* ref_dy - np.sum(h) * dx * dy)

    # check that the solutions are indeed comparable
    assert np.all(sim_tf == sim_tf[0])
    save_results(ds_x = ds_x,
                 ds_y = ds_y,
                 sim_errors = sim_errors,
                 sim_cons=sim_cons,
                 sim_nt = sim_nt,
                 sim_tf = sim_tf[0],
                 ref_dx=ref_dx,
                 ref_dy=ref_dy,
                 ref_nx = ref_nx,
                 ref_ny = ref_ny)
    
def gen_time_results(filenames):
    longsim_elapsed_time = np.zeros_like(ds_x)
    max_nt = np.zeros_like(ds_x)
    tf = np.zeros_like(ds_x)
    for j, filename in enumerate(filenames):
        with np.load(filename) as data:
            ds_y[j], ds_x[j] = data['h'].shape
            max_nt[j] = data['nt']
            longsim_elapsed_time[j] = data['elapsed_time']
            tf[j] = data['tf']

    
    assert np.all(max_nt == max_nt[0])
    secs_per_timestep = longsim_elapsed_time / max_nt
    megacells = ds_x * ds_y * 10**-6
    megacells_per_sec = megacells * max_nt / longsim_elapsed_time
    # Peak megacells will use the last two/three values
    x = megacells[-PEAK_PERFORMANCE_FROM_LAST: ]
    y = megacells_per_sec[-PEAK_PERFORMANCE_FROM_LAST: ]
    res = scipy.stats.linregress(x, y)
    peak_megacells_per_sec = np.mean(x) * res.slope + res.intercept
    
    save_results(ds_x = ds_x,
                 ds_y = ds_y,
                 secs_per_timestep = secs_per_timestep,
                 megacells_per_sec = megacells_per_sec,
                 peak_megacells_per_sec = peak_megacells_per_sec,
                 sim_tf = tf,
                 sim_nt = max_nt[0])

# ============================================================

ds_x = np.zeros(len(filenames))
ds_y = np.zeros_like(ds_x)

if (args.directory == 'time' or args.directory == 'both'):
    gen_time_results(filenames)
elif (args.directory == 'space' or args.directory == 'both'):
    gen_space_results(filenames, args.ref)
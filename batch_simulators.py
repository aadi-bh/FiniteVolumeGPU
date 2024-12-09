import subprocess
import numpy as np
import argparse
from common_simulator import *

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', default=False)
parser.add_argument('ic', default=None, choices=GetInitialCondition.ics.keys(), action=GetInitialCondition)
parser.add_argument('--sizes-file', type=str, default='domain_sizes.csv')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--tf', type=float, default=None)
group.add_argument('--nt', type=int, default=None)
args = parser.parse_args()

simulators = ['LxF', 'HLL', 'HLL2', 'KP07', 'KP07_dimsplit', 'WAF', 'FORCE']
resolutions = np.loadtxt(args.sizes_file).astype(int)
ref_nx = resolutions[0][-1]
ref_ny = resolutions[1][-1]
domain_sizes_x = resolutions[0]
domain_sizes_y = resolutions[1]

print(list(zip(domain_sizes_x, domain_sizes_y)))
tf = args.tf
nt = args.nt

rootcommand = 'python simulate.py'
ic = args.ic.__name__

for i, simulator in enumerate(simulators):
    for j, (nx, ny) in enumerate(zip(domain_sizes_x, domain_sizes_y)):
#        num_ghost_cells = GetSimulator.num_ghost_cells[simulator]
#        ny = max(ny, num_ghost_cells)
#        ref_ny = max(ny, resolutions[1][-1])
        simulate_args = [ic, simulator,
                         '--nx', str(nx),
                         '--ref-nx', str(ref_nx),
                         '--ny', str(ny),
                         '--ref-ny', str(ref_ny)]
        if tf != None:
            simulate_args += ['--tf', str(tf)]
        if nt != None:
            simulate_args += ['--nt', str(nt)]
        command = rootcommand.split(' ') + simulate_args
        print(command)
        if not args.dry_run:
            completed_process = subprocess.run(command, stdout=subprocess.DEVNULL)
            try:
                completed_process.check_returncode()
            except Exception as e:
                raise e
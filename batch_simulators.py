import subprocess
import numpy as np
import argparse
from common_simulator import *

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', default=False)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--tf', type=float, default=None)
group.add_argument('--nt', type=int, default=None)
parser.add_argument('ic', default=None, choices=GetInitialCondition.ics.keys(), action=GetInitialCondition)
args = parser.parse_args()

simulators = ['LxF', 'FORCE', 'HLL', 'HLL2', 'KP07', 'KP07_dimsplit', 'WAF']
domain_sizes_x = [2**i for i in range(6, 12)]
domain_sizes_y = domain_sizes_x
tf = args.tf
nt = args.nt

rootcommand = 'python simulate.py'
ic = args.ic.__name__

ref_nx = domain_sizes_x[-1] * 2
ref_ny = domain_sizes_y[-1] * 2

domain_sizes_x = [ref_nx] + domain_sizes_x
domain_sizes_y = [ref_ny] + domain_sizes_y

for i, simulator in enumerate(simulators):
    for j, (nx, ny) in enumerate(zip(domain_sizes_x, domain_sizes_y)):
        simulate_args = [ic, simulator,
                         '--nx', str(nx),
                         '--ny', str(ny),
                         '--ref-nx', str(ref_nx),
                         '--ref-ny', str(ref_ny)]
        if tf != None:
            simulate_args += ['--tf', str(tf)]
        if nt != None:
            simulate_args += ['--nt', str(nt)]
        command = rootcommand.split(' ') + simulate_args
        print(command)
        if not args.dry_run:
            completed_process = subprocess.run(command, stdout=subprocess.DEVNULL)
            completed_process.check_returncode()
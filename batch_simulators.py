import subprocess
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', default=False)
args = parser.parse_args()

simulators = ['LxF', 'FORCE', 'HLL', 'HLL2', 'KP07', 'KP07_dimsplit', 'WAF']
domain_sizes_x = [str(2**i) for i in range(6, 14)]
domain_sizes_y = [str(2**i) for i in range(6, 14)]

rootcommand = 'python simulate.py'
ic = 'bump'

ref_nx = domain_sizes_x[-1] * 2
ref_ny = domain_sizes_y[-1] * 2

for i, simulator in enumerate(simulators):
    for j, (nx, ny) in enumerate(zip(domain_sizes_x, domain_sizes_y)):
        simulate_args = [ic, simulator, '--nx', nx, '--ny', ny, '--ref-nx', ref_nx, '--ref-ny', ref_ny]
        command = rootcommand.split(' ') + simulate_args
        print(command)
        if not args.dry_run:
            completed_process = subprocess.run(command)
            completed_process.check_returncode()
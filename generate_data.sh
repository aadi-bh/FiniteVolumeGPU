#!/bin/bash
# conda activate ShallowWaterGPU_py310

python batch_simulators.py --sizes-file domain_sizes.csv --tf 1.0 bump
python batch_simulators.py --sizes-file domain_sizes.csv --nt 100 bump
python batch_simulators.py --sizes-file domain_sizes.csv --tf 6.0 dambreak
python batch_simulators.py --sizes-file domain_sizes.csv --nt 100 dambreak
python batch_simulators.py --sizes-file domain_sizes.csv --tf 1.0 constant
python batch_simulators.py --sizes-file domain_sizes.csv --nt 100 constant


# Convert swashes solution to the format we are used to
# copied from `gen_reference()`` in `ConvergenceShock1D.ipynb`
import os
import numpy as np

def gen_reference(nx):
    csv_filename = os.path.abspath(os.path.join('reference', f'swashes_1_nx={str(nx)}.csv'))
    reference = np.genfromtxt(csv_filename, comments='#', delimiter='\t', skip_header=0, usecols=(0,1,2))
    x, h, u = reference[:, 0], reference[:, 1], reference[:, 2]
    return x, h, h*u

if __name__ == "__main__":
    NX = 4096
    x, h, hu = gen_reference(NX)
    hv = np.zeros_like(hu)
    t = 6.0
#    nt = None
#    elapsed_time = None
    assert h.ndim == 1
    nx = h.shape[0]
    dx = x[1]-x[0]
    dy = 1.0
    ny = 1.0
    npz_filename = os.path.abspath(os.path.join('reference', f'SWASHES_{str(int(nx))}_{str(int(ny))}.npz'))
    np.savez_compressed(npz_filename, dx=dx, dy=dy, h=h, hu=hu, hv=hv, tf=t)
    #, nt=nt, elapsed_time=elapsed_time, test_data_args=test_data_args, cfl=sim_args['cfl_scale'])
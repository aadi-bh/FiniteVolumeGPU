"""
Functions needed for plots, but do not belong in a notebook.

Name:  misc_plotting.py
By:    Aadi B.
Usage: Only through import. See docstrings
"""
import matplotlib.pyplot as plt
import os
import datetime
import socket
import numpy as np
import seaborn as sns
import subprocess
from operator import itemgetter
import gc
import gzip

#Set large figure sizes
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['animation.html'] = 'html5'
#plt.rcParams['legend.markerscale'] = 1.0
#plt.rcParams['lines.markersize'] = 6
plt.rcParams['lines.markeredgewidth'] = 1.5
#plt.rcParams['savefig.dpi'] = 400


def gen_filename(simulator, nx, ic="smooth1d"):
    """DEPRECATED: Returns path of data file corresponding to arguments."""
    return os.path.abspath(os.path.join("data", ic, str(simulator.__name__) + "_" + str(nx) + ".npz"))

def setBwStyles(ax):
    """Set colours and marker cyclers for a given axis."""
    from cycler import cycler
    ax.set_prop_cycle( cycler('marker', ['.', 'x', 4, '+', '*', '1', 5])
                       + cycler('linestyle', ['-.', '--', ':', '-.', '--', ':', '-.'])
                       # + cycler('markersize', [5]*7)
                       # + cycler('color', ['k']*7)
                       # + cycler('color', itemgetter(0, 1, 2, -1, -2, -3, -4)(plt.cm.tab20c.colors))
                       + cycler('color', sns.color_palette("Paired", 7).as_hex())
                       )    

def save_figure(fig, stem, ic):
    """Set metadata and save figure as PDF, with stem inserted into the path before the extension."""
    if (not os.path.isdir("figures")):
        os.mkdir("figures")
    
    fig_filename = os.path.join("figures", ic + "_" + stem + ".pdf")
    
    metadata = {
        'CreationDate': datetime.datetime.now(), #time.strftime("%Y_%m_%d-%H_%M_%S"),
        'Author': socket.gethostname()
    }
        
    legend = fig.gca().legend_
    if (legend != None):
        fig.savefig(fig_filename, dpi=300,format='pdf',
                transparent=True, pad_inches=0.0, facecolor=None, 
                metadata=metadata, 
                bbox_extra_artists=(legend, ), bbox_inches='tight')
    else:
        fig.savefig(fig_filename, dpi=300,format='pdf',
                transparent=True, pad_inches=0.0, facecolor=None, 
                metadata=metadata, bbox_inches='tight')
    fig.savefig(fig_filename.replace('.pdf','.svg'), dpi=300, format='svg', transparent=True, bbox_inches='tight')

def plot_solution(simulator, nx, label, ic="smooth1d", **kwargs):
    """DEPRECATED: Finds and plots the given solution.
    This function needs to be tweaked so much that it is moved to the notebook."""
    datafilename = gen_filename(simulator, nx, ic)
    
    #Read the solution
    with np.load(datafilename) as data:
        h = data['h']
        
    x = np.linspace(0.5, nx-0.5, nx)* 100/float(nx)
    y = h[0,:]
    
    plt.plot(x, y, label=label, **kwargs)
    
    h = None
    x = None
    gc.collect() # Force run garbage collection to free up memory
    
def plot_comparison(nx, **kwargs):
    """DEPRECATED: Plots solutions on top of each other.
    
    This function needs to be tweaked so much it is moved to the notebook."""
    plot_solution(HLL2.HLL2, reference_nx, 'Reference', marker=' ', linestyle='-')

    for i, simulator in enumerate(simulators):
        plot_solution(simulator, nx, simulator.__name__, **kwargs)

def gen_reference(nx, ic='dambreak'):
    """Returns the (x, h, hu) for the reference solutions for the IC."""
    if ic == 'dambreak':
        csv_filename = os.path.abspath(os.path.join("reference", "swashes_1_nx=" + str(nx) + ".csv"))

        #If we do not have the data, generate it    
        if (not os.path.isfile(csv_filename)):
            print("Generating new reference!")
            swashes_path = r'C:\Users\anbro\Documents\programs\SWASHES-1.03.00_win\bin\swashes_win.exe'

            swashes_args = [\
                            '1', # 1D problems \
                            '3', # Dam breaks \
                            '1', # Domain 1 \
                            '1', # Wet domain no friction
                            str(nx) #Number of cells X
                        ]

            with open(csv_filename, 'w') as csv_file:
                p = subprocess.check_call([swashes_path] + swashes_args, stdout=csv_file)

        reference = np.genfromtxt(csv_filename, comments='#', delimiter='\t', skip_header=0, usecols=(0, 1, 2))
        x, h, u = reference[:, 0], reference[:, 1], reference[:, 2]
        return x, h, h*u
    elif ic == 'bump':
        ny = nx
        csv_filename = os.path.abspath(os.path.join("reference", "clawpack_nx=" + str(nx) + ".csv.gz"))

        if (not os.path.isfile(csv_filename)):
            print("Reference file does not exist: ", csv_filename)
            print("Please run and rename.")
            raise FileNotFoundError

        reference = np.genfromtxt(csv_filename, skip_header=9)
        assert reference.shape == (nx**2, 3)
        reference = reference.reshape((ny, nx, 3))
        h = reference[...,0]
        hu = reference[..., 1]
        hv = reference[..., 2]
        
        data = dict()
        if csv_filename.endswith('.gz'):
            open_function = gzip.open
        else:
            open_function = open
        with open_function(csv_filename, 'r') as f:
            for i in range(8):
                l = f.readline().decode('utf-8')
                s = l.split()
                data[s[1]] = s[0]
        dx = float(data['dx'])
        dy = float(data['dy'])
        xlow = float(data['xlow'])
        ylow = float(data['ylow'])
        assert nx == int(data['mx'])
        assert ny == int(data['my'])
        x = xlow + dx * np.linspace(0.5, nx - 0.5, nx)
        y = np.linspace(ylow + dy / 2, ylow + ny * dy - dy/2, ny)
        xx, yy = np.meshgrid(x,y)
        return xx, yy, h, hu, hv

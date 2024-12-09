#Misc plotting setup
import matplotlib.pyplot as plt
import os
import datetime
import socket
import numpy as np
import seaborn as sns
from operator import itemgetter
import gc

#Set large figure sizes
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['animation.html'] = 'html5'
#plt.rcParams['legend.markerscale'] = 1.0
#plt.rcParams['lines.markersize'] = 6
plt.rcParams['lines.markeredgewidth'] = 1.5
#plt.rcParams['savefig.dpi'] = 400


def gen_filename(simulator, nx, ic="smooth1d"):
    return os.path.abspath(os.path.join("data", ic, str(simulator.__name__) + "_" + str(nx) + ".npz"))

def setBwStyles(ax):
    from cycler import cycler
    ax.set_prop_cycle( cycler('marker', ['.', 'x', 4, '+', '*', '1', 5])
                       + cycler('linestyle', ['-.', '--', ':', '-.', '--', ':', '-.'])
                       # + cycler('markersize', [5]*7)
                       # + cycler('color', ['k']*7)
                       # + cycler('color', itemgetter(0, 1, 2, -1, -2, -3, -4)(plt.cm.tab20c.colors))
                       + cycler('color', sns.color_palette("Paired", 7).as_hex())
                       )    

def save_figure(fig, stem, ic):
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
                metadata=metadata)

def plot_solution(simulator, nx, label, ic="smooth1d", **kwargs):
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
    plot_solution(HLL2.HLL2, reference_nx, 'Reference', marker=' ', linestyle='-')

    for i, simulator in enumerate(simulators):
        plot_solution(simulator, nx, simulator.__name__, **kwargs)

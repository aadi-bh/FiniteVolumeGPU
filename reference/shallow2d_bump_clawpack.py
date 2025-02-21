#!/usr/bin/env python
# encoding: utf-8
r"""
Modified from https://www.clawpack.org/gallery/pyclaw/gallery/radial_dam_break.html

Solve the 2D shallow water equations:

.. math::
    h_t + (hu)_x + (hv)_y = 0 \\
    (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x + (huv)_y = 0 \\
    (hv)_t + (huv)_x + (hv^2 + \frac{1}{2}gh^2)_y = 0.

"""

import numpy as np
import gc
from clawpack import riemann
from clawpack.riemann.shallow_roe_with_efix_2D_constants import depth, x_momentum, y_momentum, num_eqn

def downsample(highres_solution, x_factor, y_factor=None):
    if (y_factor == None):
        y_factor = x_factor

    if (len(highres_solution.shape) == 1):
        highres_solution = highres_solution.reshape((1, highres_solution.size))

    assert highres_solution.shape[1] % x_factor == 0 ,f"{highres_solution.shape[1]} is not a multiple of {x_factor}"
    assert highres_solution.shape[0] % y_factor == 0 ,f"{highres_solution.shape[0]} is not a multiple of {y_factor}"
    
    if (x_factor*y_factor == 1):
        return highres_solution
    
    if (len(highres_solution.shape) == 1):
        highres_solution = highres_solution.reshape((1, highres_solution.size))

    nx = highres_solution.shape[1] / x_factor
    ny = highres_solution.shape[0] / y_factor

    return highres_solution.reshape([int(ny), int(y_factor), int(nx), int(x_factor)]).mean(3).mean(1)

def qinit(state,nx=128, ny=128, width=10.0, height=10.0, 
        ref_nx=16384, ref_ny=16384,
        bump_size=2, 
        x_center=0.5, y_center=0.5,
        h_ref=0.5, h_amp=0.1, u_ref=0.0, u_amp=0.1, v_ref=0.0, v_amp=0.1):

    X, Y = state.p_centers

    ref_dx = width / float(ref_nx)
    ref_dy = height / float(ref_ny)

    x_center = ref_dx*ref_nx*x_center
    y_center = ref_dy*ref_ny*y_center
    
    x = ref_dx*(np.arange(0, ref_nx, dtype=np.float32)+0.5) - x_center
    y = ref_dy*(np.arange(0, ref_ny, dtype=np.float32)+0.5) - y_center
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    r = np.sqrt(xv**2 + yv**2)
    xv = None
    yv = None
    gc.collect()
    
    #Generate highres then downsample
    #h_highres = 0.5 + 0.1*np.exp(-(xv**2/size + yv**2/size))
    h_highres = h_ref + h_amp*0.5*(1.0 + np.cos(np.pi*r/bump_size)) * (r < bump_size)
    h = downsample(h_highres, ref_nx/nx, ref_ny/ny)
    h_highres = None
    gc.collect()
    
    #hu_highres = 0.1*np.exp(-(xv**2/size + yv**2/size))
    u_highres = u_ref + u_amp*0.5*(1.0 + np.cos(np.pi*r/bump_size)) * (r < bump_size)
    hu = downsample(u_highres, ref_nx/nx, ref_ny/ny)*h
    u_highres = None
    gc.collect()
    
    #hu_highres = 0.1*np.exp(-(xv**2/size + yv**2/size))
    v_highres = v_ref + v_amp*0.5*(1.0 + np.cos(np.pi*r/bump_size)) * (r < bump_size)
    hv = downsample(v_highres, ref_nx/nx, ref_ny/ny)*h
    v_highres = None
    gc.collect()

    state.q[depth     ,:,:] = h
    state.q[x_momentum,:,:] = hu
    state.q[y_momentum,:,:] = hv


    
def setup(kernel_language='Fortran', use_petsc=False, outdir='./_output',
          solver_type='sharpclaw', riemann_solver='roe',disable_output=False):
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if riemann_solver.lower() == 'roe':
        rs = riemann.shallow_roe_with_efix_2D
    elif riemann_solver.lower() == 'hlle':
        rs = riemann.shallow_hlle_2D

    if solver_type == 'classic':
        solver = pyclaw.ClawSolver2D(rs)
        solver.limiters = pyclaw.limiters.tvd.MC
        solver.dimensional_split=1
    elif solver_type == 'sharpclaw':
        solver = pyclaw.SharpClawSolver2D(rs)

    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall
    solver.bc_lower[1] = pyclaw.BC.wall
    solver.bc_upper[1] = pyclaw.BC.wall

    # Domain:
    xlower = 0.0
    xupper = 100.0
    #  The comparison plots are 128x128
    mx = 1024
    ylower = xlower
    yupper = xupper
    my = mx
    x = pyclaw.Dimension(xlower,xupper,mx,name='x')
    y = pyclaw.Dimension(ylower,yupper,my,name='y')
    domain = pyclaw.Domain([x,y])

    state = pyclaw.State(domain,num_eqn)

    # Gravitational constant
    state.problem_data['grav'] = 9.8

    qinit(state,nx=mx,ny=my)

    claw = pyclaw.Controller()
    claw.tfinal = 1.0
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    if disable_output:
        claw.output_format = None
    claw.outdir = outdir
    claw.num_output_times = 2
    claw.setplot = setplot
    claw.keep_copy = True

    return claw

#--------------------------
def setplot(plotdata):
#--------------------------
    """ 
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """ 
    from clawpack.visclaw import colormaps

    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Figure for depth
    plotfigure = plotdata.new_plotfigure(name='Water height', figno=0)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [0.0, 100]
    plotaxes.ylimits = [0.0, 100]
    plotaxes.title = 'Water height'
    plotaxes.scaled = True

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = 0
    plotitem.pcolor_cmap = colormaps.red_yellow_blue
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = 0.7
    plotitem.add_colorbar = True
    
    # Scatter plot of depth
    plotfigure = plotdata.new_plotfigure(name='Scatter plot of h', figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [0., 100]
    plotaxes.ylimits = [0., 100]
    plotaxes.title = 'Scatter plot of h'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d_from_2d_data')
    plotitem.plot_var = depth
    def q_vs_radius(current_data):
        from numpy import sqrt
        x = current_data.x
        y = current_data.y
        r = sqrt(x**2 + y**2)
        q = current_data.q[depth,:,:]
        return r,q
    plotitem.map_2d_to_1d = q_vs_radius
    plotitem.plotstyle = 'o'


    # Figure for x-momentum
    plotfigure = plotdata.new_plotfigure(name='Momentum in x direction', figno=2)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [-2.5, 2.5]
    plotaxes.ylimits = [-2.5, 2.5]
    plotaxes.title = 'Momentum in x direction'
    plotaxes.scaled = True

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = x_momentum
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.add_colorbar = True
    plotitem.show = False       # show on plot?
    

    # Figure for y-momentum
    plotfigure = plotdata.new_plotfigure(name='Momentum in y direction', figno=3)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = [-2.5, 2.5]
    plotaxes.ylimits = [-2.5, 2.5]
    plotaxes.title = 'Momentum in y direction'
    plotaxes.scaled = True

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = y_momentum
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.add_colorbar = True
    plotitem.show = False       # show on plot?
    
    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)
#    from clawpack.pyclaw import plot
#    plot.html_plot()

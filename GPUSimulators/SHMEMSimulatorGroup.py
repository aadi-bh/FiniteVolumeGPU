# -*- coding: utf-8 -*-

"""
This python module implements SHMEM simulator group class

Copyright (C) 2020 Norwegian Meteorological Institute

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import logging
from GPUSimulators import Simulator, CudaContext
import numpy as np

import pycuda.driver as cuda


class SHMEMGrid(object):
    """
    Class which represents an SHMEM grid of GPUs. Facilitates easy communication between
    neighboring subdomains in the grid. Contains one CUDA context per subdomain.

    XXX: Adapted to debug on a single GPU. Either remove this possibility or 
    make it less hacky...
    """
    def __init__(self, ngpus=None, ndims=2):
        self.logger =  logging.getLogger(__name__)

        cuda.init(flags=0)
        self.logger.info("Initializing CUDA")
        num_cuda_devices = cuda.Device.count()
        
        if ngpus is None:
            #ngpus = num_cuda_devices
            ngpus = 2

        #assert ngpus <= num_cuda_devices, "Trying to allocate more GPUs than are available in the system."   
        assert ndims == 2, "Unsupported number of dimensions. Must be two at the moment"
        #assert ngpus >= 2, "Must have at least two GPUs available to run multi-GPU simulations."

        self.ngpus = ngpus
        self.ndims = ndims

        self.grid = SHMEMGrid.getGrid(self.ngpus, self.ndims)
        
        self.logger.debug("Created {:}-dimensional SHMEM grid, using {:} GPUs".format(
                self.ndims, self.ngpus))    

        # XXX: Is this a natural place to store the contexts? Consider moving contexts out of this class.
        self.cuda_contexts = []

        for i in range(self.ngpus):
            #self.cuda_contexts.append(CudaContext.CudaContext(device=i, autotuning=False))
            self.cuda_contexts.append(CudaContext.CudaContext(device=0, autotuning=False))

    def getCoordinate(self, index):
        i = (index  % self.grid[0])
        j = (index // self.grid[0])
        return i, j

    def getIndex(self, i, j):
        return j*self.grid[0] + i

    def getEast(self, index):
        i, j = self.getCoordinate(index)
        i = (i+1) % self.grid[0]
        return self.getIndex(i, j)

    def getWest(self, index):
        i, j = self.getCoordinate(index)
        i = (i+self.grid[0]-1) % self.grid[0]
        return self.getIndex(i, j)

    def getNorth(self, index):
        i, j = self.getCoordinate(index)
        j = (j+1) % self.grid[1]
        return self.getIndex(i, j)

    def getSouth(self, index):
        i, j = self.getCoordinate(index)
        j = (j+self.grid[1]-1) % self.grid[1]
        return self.getIndex(i, j)
    
    def getGrid(num_gpus, num_dims):
        assert(isinstance(num_gpus, int))
        assert(isinstance(num_dims, int))
        
        # Adapted from https://stackoverflow.com/questions/28057307/factoring-a-number-into-roughly-equal-factors
        # Original code by https://stackoverflow.com/users/3928385/ishamael
        # Factorizes a number into n roughly equal factors

        #Dictionary to remember already computed permutations
        memo = {}
        def dp(n, left): # returns tuple (cost, [factors])
            """
            Recursively searches through all factorizations
            """

            #Already tried: return existing result
            if (n, left) in memo: 
                return memo[(n, left)]

            #Spent all factors: return number itself
            if left == 1:
                return (n, [n])

            #Find new factor
            i = 2
            best = n
            bestTuple = [n]
            while i * i < n:
                #If factor found
                if n % i == 0:
                    #Factorize remainder
                    rem = dp(n // i, left - 1)

                    #If new permutation better, save it
                    if rem[0] + i < best:
                        best = rem[0] + i
                        bestTuple = [i] + rem[1]
                i += 1

            #Store calculation
            memo[(n, left)] = (best, bestTuple)
            return memo[(n, left)]


        grid = dp(num_gpus, num_dims)[1]

        if (len(grid) < num_dims):
            #Split problematic 4
            if (4 in grid):
                grid.remove(4)
                grid.append(2)
                grid.append(2)

            #Pad with ones to guarantee num_dims
            grid = grid + [1]*(num_dims - len(grid))
        
        #Sort in descending order
        grid = np.sort(grid)
        grid = grid[::-1]
        
        return grid

class SHMEMSimulatorGroup(Simulator.BaseSimulator):
    """
    Class which handles communication and synchronization between simulators in different 
    contexts (presumably on different GPUs)
    """
    def __init__(self, grid, **kwargs):
        self.logger =  logging.getLogger(__name__)
        sims = []

        assert(grid.ngpus > 0)

        for i in range(grid.ngpus):
            kwargs['context'] = grid.cuda_contexts[i]
            sims.append(EE2D_KP07_dimsplit.EE2D_KP07_dimsplit(**kwargs))
            #sims[i] = SHMEMSimulator(i, local_sim, grid) # 1st attempt: no wrapper (per sim)
        
        autotuner = sims[0].context.autotuner
        sims[0].context.autotuner = None
        boundary_conditions = sims[0].getBoundaryConditions()
        super().__init__(sims[0].context, 
            sims[0].nx, sims[0].ny, 
            sims[0].dx, sims[0].dy, 
            boundary_conditions,
            sims[0].cfl_scale,
            sims[0].num_substeps,  
            sims[0].block_size[0], sims[0].block_size[1])
        sims[0].context.autotuner = autotuner
        
        self.nsubdomains = grid.ngpus
        self.sims = sims
        self.grid = grid

        self.east = []
        self.west = []
        self.north = []
        self.south = []

        self.nvars = []

        self.read_e = []
        self.read_w = []
        self.read_n = []
        self.read_s = []
        
        self.write_e = []
        self.write_w = []
        self.write_n = []
        self.write_s = []

        self.e = []
        self.w = []
        self.n = []
        self.s = []
        
        for i, sim in enumerate(self.sims):
            #Get neighbor subdomain ids
            self.east[i] = grid.getEast(self.index)
            self.west[i] = grid.getWest(self.index)
            self.north[i] = grid.getNorth(self.index)
            self.south[i] = grid.getSouth(self.index)
            
            #Get coordinate of this subdomain
            #and handle global boundary conditions
            new_boundary_conditions = Simulator.BoundaryCondition({
                'north': Simulator.BoundaryCondition.Type.Dirichlet,
                'south': Simulator.BoundaryCondition.Type.Dirichlet,
                'east': Simulator.BoundaryCondition.Type.Dirichlet,
                'west': Simulator.BoundaryCondition.Type.Dirichlet
            })
            gi, gj = grid.getCoordinate(i)
            if (gi == 0 and boundary_conditions.west != Simulator.BoundaryCondition.Type.Periodic):
                self.west = None
                new_boundary_conditions.west = boundary_conditions.west;
            if (gj == 0 and boundary_conditions.south != Simulator.BoundaryCondition.Type.Periodic):
                self.south = None
                new_boundary_conditions.south = boundary_conditions.south;
            if (gi == grid.grid[0]-1 and boundary_conditions.east != Simulator.BoundaryCondition.Type.Periodic):
                self.east = None
                new_boundary_conditions.east = boundary_conditions.east;
            if (gj == grid.grid[1]-1 and boundary_conditions.north != Simulator.BoundaryCondition.Type.Periodic):
                self.north = None
                new_boundary_conditions.north = boundary_conditions.north;
            sim.setBoundaryConditions(new_boundary_conditions)
                    
            #Get number of variables
            self.nvars[i] = len(sim.getOutput().gpu_variables)
            
            #Shorthands for computing extents and sizes
            gc_x = int(sim.getOutput()[0].x_halo)
            gc_y = int(sim.getOutput()[0].y_halo)
            nx = int(sim.nx)
            ny = int(sim.ny)
            
            #Set regions for ghost cells to read from
            #These have the format [x0, y0, width, height]
            self.read_e.append(np.array([  nx,    0, gc_x, ny + 2*gc_y]))
            self.read_w.append(np.array([gc_x,    0, gc_x, ny + 2*gc_y]))
            self.read_n.append(np.array([gc_x,   ny,   nx,        gc_y]))
            self.read_s.append(np.array([gc_x, gc_y,   nx,        gc_y]))
            
            #Set regions for ghost cells to write to
            self.write_e.append(self.read_e + np.array([gc_x, 0, 0, 0]))
            self.write_w.append(self.read_w - np.array([gc_x, 0, 0, 0]))
            self.write_n.append(self.read_n + np.array([0, gc_y, 0, 0]))
            self.write_s.append(self.read_s - np.array([0, gc_y, 0, 0]))
            
            #Allocate host data
            #Note that east and west also transfer ghost cells
            #whilst north/south only transfer internal cells
            #Reuses the width/height defined in the read-extets above
            self.e.append(np.empty((self.nvars, self.read_e[3], self.read_e[2]), dtype=np.float32))
            self.w.append(np.empty((self.nvars, self.read_w[3], self.read_w[2]), dtype=np.float32))
            self.n.append(np.empty((self.nvars, self.read_n[3], self.read_n[2]), dtype=np.float32))
            self.s.append(np.empty((self.nvars, self.read_s[3], self.read_s[2]), dtype=np.float32))

        self.logger.debug("Initialized {:d} subdomains".format(len(self.sims)))
    

    def substep(self, dt, step_number):
        for i, sim in enumerate(self.sims):
            self.exchange(i)
            sim.substep(dt, step_number)
    
    def getOutput(self):
        # XXX: Does not return what we would expect.
        return self.sims[0].getOutput() 
        
    def synchronize(self):
        for sim in self.sims:
            sim.synchronize()
        
    def check(self):
        # XXX: Does not return what we would expect.
        return self.sims[0].check()
    
    def computeDt(self):
        global_dt = float("inf")

        for sim in self.sims:
            local_dt = sim.computeDt()
            if local_dt < global_dt:
                global_dt = local_dt
            self.logger.debug("Local dt: {:f}".format(local_dt))

        self.logger.debug("Global dt: {:f}".format(global_dt))
        return global_dt
        
    def getExtent(self, index):
        """
        Function which returns the extent of the subdomain with index 
        index in the grid
        """
        width = self.sims[index].nx*self.sims[index].dx
        height = self.sims[index].ny*self.sims[index].dy
        i, j = self.grid.getCoordinate(index)
        x0 = i * width
        y0 = j * height 
        x1 = x0 + width
        y1 = y0 + height
        return [x0, x1, y0, y1]
        
    def exchange(self, i):
        ####
        # First transfer internal cells north-south
        ####
        self.ns_download(i)
        self.ns_upload(i)

        ####
        # Then transfer east-west including ghost cells that have been filled in by north-south transfer above
        ####
        self.ew_download(i)
        self.ew_upload(i)

    def ns_download(self, i):
        #Download from the GPU
        if self.north[i] is not None:
            for k in range(self.nvars[i]):
                self.sims[i].u0[k].download(self.sims[i].stream, cpu_data=self.n[i][k,:,:], asynch=True, extent=self.read_n[i])
        if self.south[i] is not None:
            for k in range(self.nvars[i]):
                self.sims[i].u0[k].download(self.sims[i].stream, cpu_data=self.s[i][k,:,:], asynch=True, extent=self.read_s[i])
        self.sims[i].stream.synchronize()

    def ns_upload(self, i):
        #Upload to the GPU
        if self.north[i] is not None:
            for k in range(self.nvars[i]):
                self.sims[i].u0[k].upload(self.sims[i].stream, self.s[self.north[i]][k,:,:], extent=self.write_n[i])
        if self.south[i] is not None:
            for k in range(self.nvars[i]):
                self.sims[i].u0[k].upload(self.sims[i].stream, self.n[self.south[i]][k,:,:], extent=self.write_s[i])
        
    def ew_download(self, i):
        #Download from the GPU
        if self.east[i] is not None:
            for k in range(self.nvars[i]):
                self.sims[i].u0[k].download(self.sims[i].stream, cpu_data=self.e[i][k,:,:], asynch=True, extent=self.read_e[i])
        if self.west[i] is not None:
            for k in range(self.nvars[i]):
                self.sims[i].u0[k].download(self.sims[i].stream, cpu_data=self.w[i][k,:,:], asynch=True, extent=self.read_w[i])
        self.sims[i].stream.synchronize()

    def ew_upload(self, i):
        #Upload to the GPU
        if self.east[i] is not None:
            for k in range(self.nvars[i]):
                self.sims[i].u0[k].upload(self.sims[i].stream, self.w[self.east[i]][k,:,:], extent=self.write_e[i])
        if self.west[i] is not None:
            for k in range(self.nvars[i]):
                self.sims[i].u0[k].upload(self.sims[i].stream, self.e[self.west[i]][k,:,:], extent=self.write_w[i])

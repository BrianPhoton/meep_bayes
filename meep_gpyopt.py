# Attempt to combine meep optimization tutorial with GPyOpt. See

# https://nbviewer.jupyter.org/github/NanoComp/meep/blob/master/python/examples/adjoint_optimization/02-Waveguide_Bend.ipynb

# and

# https://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/devel/manual/GPyOpt_reference_manual.ipynb

#======================================================================
#Begin with the introductory section of the above Meep tutorial:
#======================================================================

import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
import nlopt
from matplotlib import pyplot as plt
mp.quiet(quietval=True)
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

# Set up optimization problem
resolution = 20

Sx = 6
Sy = 5
cell_size = mp.Vector3(Sx,Sy)

pml_layers = [mp.PML(1.0)]

fcen = 1/1.55
width = 0.2
fwidth = width * fcen
source_center  = [-1.5,0,0]
source_size    = mp.Vector3(0,2,0)
kpoint = mp.Vector3(1,0,0)
src = mp.GaussianSource(frequency=fcen,fwidth=fwidth)
source = [mp.EigenModeSource(src,
                    eig_band = 1,
                    direction=mp.NO_DIRECTION,
                    eig_kpoint=kpoint,
                    size = source_size,
                    center=source_center)]

design_region_resolution = 10
Nx = design_region_resolution
Ny = design_region_resolution

design_variables = mp.MaterialGrid(mp.Vector3(Nx,Ny),SiO2,Si,grid_type='U_SUM')
design_region = mpa.DesignRegion(design_variables,volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(1, 1, 0)))


geometry = [
    mp.Block(center=mp.Vector3(x=-Sx/4), material=Si, size=mp.Vector3(Sx/2, 0.5, 0)), # horizontal waveguide
    mp.Block(center=mp.Vector3(y=Sy/4), material=Si, size=mp.Vector3(0.5, Sy/2, 0)),  # vertical waveguide
    mp.Block(center=design_region.center, size=design_region.size, material=design_variables), # design region
    mp.Block(center=design_region.center, size=design_region.size, material=design_variables,
             e1=mp.Vector3(x=-1).rotate(mp.Vector3(z=1), np.pi/2), e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), np.pi/2))
]

sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    eps_averaging=False,
                    resolution=resolution)

TE_top = mpa.EigenmodeCoefficient(sim,mp.Volume(center=mp.Vector3(0,1,0),size=mp.Vector3(x=2)),mode=1)
ob_list = [TE_top]

def J(alpha):
    return npa.abs(alpha) ** 2

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=ob_list,
    design_regions=[design_region],
    fcen=fcen,
    df = 0,
    nf = 1,
    decay_fields=[mp.Ez]
)

#Plot initial geometry
x0 = 0.5*np.ones((Nx*Ny,))
opt.update_design([x0])

opt.plot2D(True)
plt.show()

#Define the cost function
#note the use of the gradient, which is unnecessary for current implementation of the optimizer with GPyOpt

evaluation_history = []
sensitivity = [0]
#def f(x, grad):
#    f0, dJ_du = opt([x])
#    if grad.size > 0:
#        grad[:] = np.squeeze(dJ_du)
#    evaluation_history.append(np.real(f0))
#    sensitivity[0] = dJ_du
#    return np.real(f0)

def f2(x):
    print(x)
    print(x.ndim)
    if x.ndim > 1:
        #TODO this is a wierd kludge required to get meep to work with GPyOpt.  Why is it necessary?
        x = x[0]
    print(x)
    print(x.ndim)
    f0, dJ_du = opt([x]) 
    evaluation_history.append(np.real(f0))
    sensitivity[0] = dJ_du
    return -np.real(f0) #-1 because we're maximizing the power


#======================================================================
#Use GPyOpt tutorial to optimize waveguide bend:
#======================================================================

from GPyOpt.methods import BayesianOptimization
import GPy
import GPyOpt

#define the bounds
dim = x0.shape[0]
bounds =[]
for dummy in range(dim):
    bounds.append({'type': 'continuous', 'domain': (0,1)}) 

#Define the optimization problem.  This will sample the cost functino
#several times, so you'll see that output.

myBopt_meep = GPyOpt.methods.BayesianOptimization(f2,
                                              domain=bounds,        # box-constraints of the problem
                                             acquisition_type='EI',
                                              acquisition_weight = 20.0,
                                                  exact_feval = True)

# runs the optimization for the three methods
evaluation_history = []
sensitivity = [0]

max_iter = 40  # maximum time 40 iterations
max_time = 6000  # maximum time 60 seconds

myBopt_meep.run_optimization(max_iter,max_time,verbosity=True)

#plot some results:
plt.figure()
plt.plot(evaluation_history,'o-')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('FOM')
plt.show()

myBopt_meep.plot_convergence()

x_opt = myBopt_meep.x_opt
print(myBopt_meep.fx_opt)

opt.update_design([x_opt])
opt.plot2D(True,plot_monitors_flag=False,output_plane=mp.Volume(center=(0,0,0),size=(2,2,0)))
plt.axis("off");
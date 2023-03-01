import numpy as np
import matplotlib.pyplot as plt

# Define simulation parameter 
x0 = 0                          # Left boundary
xn = 1                          # Right boundary
N  = 129                        # Number of nodes
x = np.linspace(x0,xn,n)        # intervals
dx = x[1] - x[0]                # grid spacing
domain_1d = (x, dx, n)          # 

# Define simulation time parameter
cfl = 0.25                              # CFL number
t  = 1.0                                # simulation time in seconds
dt = dt = cfl*dx/(np.sqrt(g*H_bar))     # time step in seconds
simulation_time = (t,dt)                #


# Define physical parameters
g = 9.81                                # Gravity constant

# Define initial conditions
profile_1d = (np.exp(-(x-0.5)**2/0.01) ,x*0)  # Gaussian profile


# Define boundary conditions


# Define numerical flux function 
def flux(h, u):
    return np.array([h*u, h*u**2 + 0.5*g*h**2])

def integrate(domain_1d, profile_1d,boundary, constants,simulation_time, source=False):
    '''
    Solving 1-dimensional shallow water wave equation using Finite Volume
    and HLL Solver 
    '''
    x,dx,n = domain_1d
    g,cfl  = constants
    t, dt  = simulation_time
    h, u   = profile_1d
    hu     = h*u




    if source == False:
        source == 0*E

    
#%%
import numpy as np
import matplotlib.pyplot as plt

# Define simulation parameter 
x0 = 0                          # Left boundary
xn = 1                          # Right boundary
N  = 129                        # Number of nodes
x = np.linspace(x0,xn,N)        # intervals
dx = x[1] - x[0]                # grid spacing
domain_1d = (x, dx, N)          # 

# Define physical parameters
g = 9.81                                # Gravity constant

# Define simulation time parameter
cfl = 0.25                              # CFL number
finaltime  = 1.0                                # simulation time in seconds
dt = dt = cfl*dx/(np.sqrt(g))     # time step in seconds
simulation_time = (finaltime,dt)                #

# constants
constants = (g,cfl)

# Define initial conditions
def initial_profile_1d(x):
    h_init = np.exp(-(x-0.5)**2/0.01)
    u_init = x*0
    return h_init, u_init

initial_profile = initial_profile_1d(x)


# Define boundary conditions


# Define cell control midpoint and cell width
x_half = (x[1:] + x[:-1])/2

cell = x_half[1:] - x_half[:-1]
cell = np.insert(cell,0,x_half[0]-x[0])
cell = np.insert(cell,-1,x[-1]-x_half[-1])  # cell width for each control volume 


# Define numerical flux function 
def flux(h, u):
    return np.array([h*u, h*u**2 + 0.5*g*h**2])

def RHS(t,h,u,dx):
    rhs = 1/dx * (flux(h[1:],u[1:]) - flux(h[:-1],u[:-1]))
    return rhs

def rk4(domain_1d,t,dt,h,u,g,cfl):
    x,dx,n = domain_1d
    f1_h, f1_u = RHS(t     , h          , u          , dx)
    f2_h, f2_u = RHS(t+dt/2, h+f1_h*dt/2, u+f1_u*dt/2, dx)
    f3_h, f3_u = RHS(t+dt/2, h+f2_h*dt/2, u+f1_u*dt/2, dx)
    f4_h, f4_u = RHS(t+dt  , h+f3_h*dt  , u+f1_u*dt  , dx)
    h += (dt/6)* (f1_h + 2*f2_h + 2*f3_h + f4_h)
    u += (dt/6)* (f1_u + 2*f2_u + 2*f3_u + f4_u)
    return h,u 

def integrate(domain_1d, initial_profile,constants,simulation_time, boundary=False, source=False):
    '''
    Solving 1-dimensional shallow water wave equation using Finite Volume
    and HLL Solver 
    '''
    x,dx,n = domain_1d
    g,cfl  = constants
    finaltime, dt  = simulation_time
    h, u   = initial_profile
    hu     = h*u

    timestep = 0
    h_num = h
    u_num = u
    while timestep < finaltime:
        h_num, u_num = rk4(domain_1d,timestep,dt,h_num,u_num,g,cfl)
        timestep += dt
    return (h_num,u_num)


h_num, u_num = integrate(domain_1d, initial_profile,constants,simulation_time, boundary=False, source=False)

plt.plot(x,h_num)
plt.plot(x,np.exp(-(x-0.5)**2/0.01))
plt.show()

# %%

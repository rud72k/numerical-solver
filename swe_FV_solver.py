#%%
import numpy as np
import matplotlib.pyplot as plt

# Define simulation parameter 
x0 = 0                          # Left boundary
xn = 1                          # Right boundary
N  = 4                          # Number of nodes - 1
x = np.linspace(x0,xn,N)        # intervals
dx = x[1] - x[0]                # grid spacing
print(x)
# Define cell control midpoint and cell width
x_half = (x[1:] + x[:-1])/2             # define x_(1/2) etc
x_control = np.append(x0,x_half)        # add left-most node
x_control = np.append(x_control,xn)     # add right-most node

cellwidth = x_control[1:] - x_control[:-1]      # control cell width                   


domain_1d = (x, dx, N)          # 

# Define physical parameters
g = 9.81                                # Gravity constant

# Define simulation time parameter
cfl = 0.25                              # CFL number
finaltime  = 1.0                        # simulation time in seconds
dt = dt = cfl*dx/(np.sqrt(g))           # time step in seconds
simulation_time = (finaltime,dt)        #


# Define initial conditions
def initial_profile_1d(x):
    h_init = np.exp(-(x_control-0.5)**2/0.01)
    u_init = x_control*0
    return h_init, u_init

h_init, u_init = initial_profile_1d(x_control)
h = (h_init[1:]+h_init[:-1])/2
u = (u_init[1:]+u_init[:-1])/2
initial_profile = (h,u)



# Define boundary conditions




# Define numerical flux function 

def flux(q1, q2):
    return np.array([q2, q2**2/q1 + 9.81*q1**2/2])

def RHS(t,q1,q2,dx):
    rhs = (flux(q1,q2) - flux(q1,q2))/dx
    return rhs

def rk4(t,dt,q1,q2,dx):
    f1_q1, f1_q2 = RHS(t     , q1          , q2           , dx)
    f2_q1, f2_q2 = RHS(t+dt/2, q1+f1_q1*dt/2, q2 +f1_q2*dt/2, dx)
    f3_q1, f3_q2 = RHS(t+dt/2, q1+f2_q1*dt/2, q2 +f1_q2*dt/2, dx)
    f4_q1, f4_q2 = RHS(t+dt  , q1+f3_q1*dt  , q2 +f1_q2*dt  , dx)
    q1 += (dt/6)* (f1_q1 + 2*f2_q1 + 2*f3_q1 + f4_q1)
    q2  += (dt/6)* (f1_q2 + 2*f2_q2 + 2*f3_q2 + f4_q2)
    return q1,q2  

def integrate(dx, initial_profile,simulation_time, boundary=False, source=False):
    '''
    Solving 1-dimensional shallow water wave equation using Finite Volume
    and HLL Solver 
    '''
    finaltime, dt  = simulation_time
    h, u   = initial_profile

    t = dt              # initialise first time step
    q1 = h
    q2 = h*u
    while t < finaltime:
        h_num, u_num = rk4(t,dt,q1,q2,dx)
        t += dt
    h_num = q1
    u_num = q2/q1
    return (h_num,u_num)


h_num, u_num = integrate(domain_1d, initial_profile,simulation_time, boundary=False, source=False)

plt.plot(x,h_num)
plt.plot(x,np.exp(-(x-0.5)**2/0.01))
plt.show()

# %%

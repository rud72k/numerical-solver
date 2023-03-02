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

initial_profile = initial_profile_1d(x_control)

# Define boundary conditions

# Define numerical flux function 
def flux(q1,q2):
    flux_1 = q2
    flux_2 = q2**2/q1 + 1/2 * g*q1**2
    return flux_1, flux_2

def RHS(t,q1,q2,dx):
    flux_1, flux_2 = flux(q1,q2)
    RHS1 = (1/dx)*(flux_1[1:] - flux_1[:-1])
    RHS2 = (1/dx)*(flux_2[1:] - flux_2[:-1])
    return RHS1, RHS2

# Define Runge-Kutta methods
# def rk4(t,dt,q1,q2,dx):
#     f1_q1, f1_q2 = RHS(t     , q1           , q2            , dx)
#     f2_q1, f2_q2 = RHS(t+dt/2, q1+f1_q1*dt/2, q2 +f1_q2*dt/2, dx)
#     f3_q1, f3_q2 = RHS(t+dt/2, q1+f2_q1*dt/2, q2 +f1_q2*dt/2, dx)
#     f4_q1, f4_q2 = RHS(t+dt  , q1+f3_q1*dt  , q2 +f1_q2*dt  , dx)
#     q1 += (dt/6)* (f1_q1 + 2*f2_q1 + 2*f3_q1 + f4_q1)
#     q2  += (dt/6)* (f1_q2 + 2*f2_q2 + 2*f3_q2 + f4_q2)
#     return q1,q2  
# def rk4(t,dt,q1,q2,h,u,dx):
#     f1_q1, f1_q2 = RHS(t     , h            , u             , dx)
#     f2_q1, f2_q2 = RHS(t+dt/2, q1+f1_q1*dt/2, q2 +f1_q2*dt/2, dx)
#     f3_q1, f3_q2 = RHS(t+dt/2, q1+f2_q1*dt/2, q2 +f1_q2*dt/2, dx)
#     f4_q1, f4_q2 = RHS(t+dt  , q1+f3_q1*dt  , q2 +f1_q2*dt  , dx)
#     q1 += (dt/6)* (f1_q1 + 2*f2_q1 + 2*f3_q1 + f4_q1)
#     q2 += (dt/6)* (f1_q2 + 2*f2_q2 + 2*f3_q2 + f4_q2)
#     return q1,q2  

def integrate(dx, initial_profile,simulation_time, boundary=False, source=False):
    '''
    Solving 1-dimensional shallow water wave equation using Finite Volume
    and HLL Solver 
    '''
    finaltime, dt  = simulation_time
    g      = 9.81                                   # the gravity constant 
    t = dt                                          # initialise first time step
    h_num, u_num = initial_profile                  # initialise numerical solution
    flux_1 = h_num*u_num                            # initial flux
    flux_2 = h_num*u_num**2 + 1/2 * g*h_num**2      #
    q1 = (h[1:]+h[:-1])/2                           # q_bar = [h_bar, u_bar]
    q2 = (h[1:]+h[:-1])/2 * (u[1:]+u[:-1])/2        #


    while t < finaltime:                            # main loop
        # Integrating with Runge-Kutta 

        f1_q1, f1_q2 = RHS(t     , h            , u             , dx)
        f2_q1, f2_q2 = RHS(t+dt/2, q1+f1_q1*dt/2, q2 +f1_q2*dt/2, dx)
        f3_q1, f3_q2 = RHS(t+dt/2, q1+f2_q1*dt/2, q2 +f1_q2*dt/2, dx)
        f4_q1, f4_q2 = RHS(t+dt  , q1+f3_q1*dt  , q2 +f1_q2*dt  , dx)
        q1 += (dt/6)* (f1_q1 + 2*f2_q1 + 2*f3_q1 + f4_q1)
        q2 += (dt/6)* (f1_q2 + 2*f2_q2 + 2*f3_q2 + f4_q2)


        q1, q2 = rk4(t,dt,q1,q2,h_num,u_num,dx)
        h_num = q1
        u_num = q2/q1
        t += dt

    return (h_num,u_num)


h_num, u_num = integrate(domain_1d, initial_profile,simulation_time, boundary=False, source=False)

plt.plot(x,h_num)
plt.plot(x,np.exp(-(x-0.5)**2/0.01))
plt.show()

# %%

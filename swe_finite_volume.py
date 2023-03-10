#%% 
import numpy as np
import matplotlib.pyplot as plt
import time 

# Define the domain
x0 = 0                      # leftmost nodes
xN = 1                      # rightmost nodes
N = 256                       # number of nodes
x = np.linspace(x0,xN,N+1)
dx = 1/(N-1)


# Define Physical parameter
g_ = 9.81                    # gravity acceleration 


# Define the simulation parameter
H = 0.1
U = 0
cfl = 0.25
dt = cfl*dx/(U + np.sqrt(g_*H))
finaltime = np.pi/2                           

simulation_time = dt, finaltime                     #init time

# Define the initial condition 
def initial_profile_1d(x,H):
    h_init = np.exp(-(x-0.5)**2/0.01) + H
    u_init = x*0
    return np.block([[h_init], [u_init]])

q_init = initial_profile_1d(x,H)           #init profile

# # Change h and u into q = [h hu]^T
# def change_variable(initial):
#     h, u = initial
#     q = np.array([h,h*u])
#     return q
# q = change_variable(initial_profile)

# Define the boundary 
# will modify later

BT_left = (np.array([[H],[U]]))
BT_right = (np.array([[H],[U]]))
BT = BT_left, BT_right                              #init Boundary Terms

# Define the flux and numerical flux
def flux(q):
    '''q = [h u]^T'''
    h, u = q[0], q[1]
    g = 9.81
    flux_0 = u*h
    flux_1 = u**2*h + 1/2 * g*h**2
    flux = np.block([[flux_0], [flux_1]])
    return flux

def max_eigen(q):
    '''q = [h u]^T'''
    g = 9.81
    lambda_ = np.abs(q[1]) + np.sqrt(g*q[0])
    return lambda_

def harmonic_mean(a,b):
    np.seterr('ignore')
    y = np.where(a+b >1e-7,np.divide(2*a*b, a+b),0*a)
    return y

def numerical_flux(q):
    '''
    The vector q = [h u]^T
    Define Rusanov numerical flux.
    This numerical flux using harmonic mean dissipation coefficient
    '''
    alpha_x0       = 0
    alpha_internal = harmonic_mean(flux(q)[:,1:],flux(q)[:,:-1])
    alpha_xN       = 0 
    F_x0       =  flux(q)[:,0]
    F_internal = (flux(q)[:,1:] + flux(q)[:,:-1])/2  
    F_xN       =  flux(q)[:,-1]
    dissipation_0        = alpha_x0       * flux(q)[:,0]
    dissipation_internal = alpha_internal * flux(q)[:,1:] - flux(q)[:,:-1]
    dissipation_N        = alpha_xN       * flux(q)[:,-1]
    F_x0       = F_x0       + dissipation_0
    F_internal = F_internal + dissipation_internal
    F_xN       = F_xN       + dissipation_N 
    F = np.c_[F_x0, F_internal, F_xN]

    return F

# Define the RHS of the equation 
def RHS(t,q,x):
    numerical_flux_q = numerical_flux(q)
    RHS_left    = numerical_flux_q[:,0]    - numerical_flux_q[:,1]
    RHS_internal= numerical_flux_q[:,1:-2] - numerical_flux_q[:,2:-1]
    RHS_right   = numerical_flux_q[:,-2]   - numerical_flux_q[:,-1]
    RHS = np.c_[RHS_left,RHS_internal,RHS_right]
    
    x_half           = (x[1:] + x[:-1])/2           # node for FV: midpoint of each original cell
    width_internal   = x_half[1:] - x_half[:-1]     # width of the controlcell not touch the boundary
    width_left       = x[0] - x_half[0]             # widht of the left-most controlcell touching left boundary
    width_right      = x_half[-1] - x[-1]           # widht of the right-most controlcell, touching right boundary
    controlcell_width = np.block([width_left,width_internal, width_right])

    RHS = RHS*controlcell_width
    return RHS

# Forced Solution
mms = False                             #init mms

# Define integrating function and iteration
def integrate_pde(x,initial_profile,simulation_time, BT, stack=False):
    start = time.time()
    q              = initial_profile
    dt_, finaltime_ = simulation_time                     
    t               = 0                         # initiate time counter
    BT_left, BT_right  = BT                     # Boundary terms

    if stack == True:
        h_stack =[]
        u_stack =[]
    
    # integrate with Runge-Kutta
    # for i in range(100):
    while t < finaltime_:
        f1 = RHS(t      , q           ,x)
        f2 = RHS(t+dt_/2, q + f1*dt_/2,x)
        f3 = RHS(t+dt_/2, q + f2*dt_/2,x)
        f4 = RHS(t+dt_  , q + f3*dt_  ,x)
        q += (dt_/6)* (f1 + 2*f2 + 2*f3 + f4) 
        t += dt_
        # inject the boundary forcefully
        q[0,0] = BT_left[0]                 # h left
        q[0,-1] = BT_right[0]                 # h right
        # q[1,0] = BT_left[1]                 # u left
        # q[1,-1] = BT_right[1]                 # u right
        print(t)

    if stack == True:
        h_stack.append(q[0])
        u_stack.append(q[1])
        q_stack = h_stack, u_stack
        end = time.time()
        time_elapsed = end-start
        print(time_elapsed)
        return q_stack
    else:
        end = time.time()
        time_elapsed = end-start
        print(time_elapsed)
        return q


# Run the simulation 

q = integrate_pde(x,q_init,simulation_time, BT, stack=True) #init initialize the simulation
print('programm running:',time__, 'seconds')
print('Simulation:',)
print('final time:',finaltime)
print('time step:', dt)
print('number of nodes', N+1)

# %%

plt.plot(x,q[0],label='depth')
plt.ylim([-0.05,1.21])
plt.legend()
plt.show()

# %%

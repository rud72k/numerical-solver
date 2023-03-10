#%%
import numpy as np
import matplotlib.pyplot as plt
import time

# Define simulation parameter 
x0 = 0                          # Left boundary
xn = 1                          # Right boundary
N  = 501                          # Number of nodes - 1
x = np.linspace(x0,xn,N)        # intervals
dx = x[1] - x[0]                # grid spacing
dxx = 0.002

# Define cell control midpoint and cell width
x_half = (x[1:] + x[:-1])/2             # define x_(1/2) etc
x_control = np.append(x0,x_half)        # add left-most node
x_control = np.append(x_control,xn)     # add right-most node

cellwidth = x_control[1:] - x_control[:-1]      # control cell width                   


domain_1d = (x, dx, N)          # 

# Define physical parameters
g = 9.81                                # Gravity constant


# Define simulation time parameter
cfl = 0.15                              # CFL number
finaltime  = 0.1                       # simulation time in seconds
dt = dt = cfl*dx/(np.sqrt(g))           # time step in seconds
simulation_time = (finaltime,dt)        #


# Define initial conditions
H = 0.1
U = 0

def initial_profile_1d(x,H):
    '''
    Control cell for Finite Volume is taking into account.
    '''
    h_init = np.exp(-(x-0.5)**2/0.001) + H
    u_init = x*0
    return h_init, u_init

initial_profile = initial_profile_1d(x,H)

# Forced solution 
mms = 0


# Define boundary conditions

# BT_left = np.ones((2,1))*H
# BT_right = np.ones((2,1))*H


BT_left = (np.array([[H],[H*U]]))
BT_right = (np.array([[H],[H*U]]))

from boundary import *

# H_bar, U_bar, g = 0.1, 0, 9.81



# Define numerical flux function 

def flux(q1,q2):
    flux = np.zeros((2,1))
    flux[0] = q2
    flux[1] = q2**2/q1 + 1/2 * g*q1**2
    # flux = np.block([[flux_1], [flux_2]])
    return flux


def numericalflux(qm,qp):
    g = 9.81
    hm = qm[0]
    um = qm[1]/qm[0]
    lambda_m = np.abs(um) + np.sqrt(g*hm)
    hp = qp[0]
    up = qp[1]/qp[0]
    lambda_p = np.abs(up) + np.sqrt(g*hp)

    alpha = 2*(lambda_m*lambda_p)/(lambda_m + lambda_p)     # Rusanov flux (using harmonic mean)
    # alpha = 0
    F_flux = np.zeros((2,1))
    F_flux[:] = (flux(qm[0],qm[1]) + flux(qp[0],qp[1]))/2 
    Qp = np.zeros((2,1))
    Qm = np.zeros((2,1))
    Qp[0,0] = qp[0]
    Qp[1,0] = qp[1]
    Qm[0,0] = qm[0]
    Qm[1,0] = qm[1]
    F_flux = F_flux -  alpha/2*(Qp - Qm)
    
    return F_flux

    
# Define the right hand side of the differential equations
def RHS(t,q1,q2,dx,N):
    F = np.zeros((2,N))
    RHS = np.zeros((2,N))
    # flux_left = flux(q1[:-1],q2[:-1])/dx
    # flux_right = flux(q1[1:],q2[1:])/dx
    F_0 = flux(q1[0],q2[0])                 # left side
    F_N = flux(q1[-1],q2[-1])               # right side

    qm = np.array([q1[0],q2[0]])
    qp = np.array([q1[1],q2[1]])
    F_half_0 = numericalflux(qm,qp)

    # print(F_half_0)
    # print(F_0)
    Flux_difference_0 = - (F_half_0 - F_0)*2/dxx 
    
    qm = np.array([q1[-2],q2[-2]])
    qp = np.array([q1[-1],q2[-1]])
    F_half_N = numericalflux(qm,qp)

    Flux_difference_N = - (F_half_N - F_N)*2/dxx 

    for k in range(1,N):
        qm = np.array([q1[k-1],q2[k-1]])
        qp = np.array([q1[k],q2[k]])
        temp = numericalflux(qm,qp)
        F[0,k-1] = temp[0,0]
        F[1,k-1] = temp[1,0]
    
    for k in range(1,N-1):
        another_temp = - (F[:,k] - F[:,k-1])/dxx
        RHS[0,k] = another_temp[0]
        RHS[1,k] = another_temp[1]
        # RHS[:,k] = - (F[:,k] - F[:,k-1])/dx
    
    # RHS[:,0] = Flux_difference_0
    # RHS[:,-1] = Flux_difference_N

    RHS[0,0] = Flux_difference_0[0]
    RHS[1,0] = Flux_difference_0[1]
    RHS[0,-1] = Flux_difference_N[0]
    RHS[1,-1] = Flux_difference_N[1]

    return RHS

def integrate(dx, initial_profile,simulation_time, boundary=False, source=False, stacked =True):
    '''
    Solving 1-dimensional shallow water wave equation using Finite Volume
    and HLL Solver.
    '''
    start = time.time()
    finaltime, dt  = simulation_time
    g      = 9.81                                   # the gravity constant 
    t = dt                                          # initialise first time step
    h_num, u_num = initial_profile                  # initialise numerical solution

    # change the variables
    q1 = h_num                         # q_bar = [h_bar, u_bar]
    q2 = h_num*u_num                   #
    q  = np.block([[q1],[q2]])
    
    h_stack = []
    u_stack = []

    while t < finaltime:                            # main loop
    # for i in range(500):
        # Integrating with Runge-Kutta 
        f1 = RHS(t     , q1           , q2            , dx,N)
        f2 = RHS(t+dt/2, q1+f1[0]*dt/2, q2 +f1[1]*dt/2, dx,N)
        f3 = RHS(t+dt/2, q1+f2[0]*dt/2, q2 +f2[1]*dt/2, dx,N)
        f4 = RHS(t+dt  , q1+f3[0]*dt  , q2 +f3[1]*dt  , dx,N)
        q += (dt/6)* (f1 + 2*f2 + 2*f3 + f4)
        t += dt
        # revert the variable and stack the solution
        q[0,0] = BT_left[0]
        # q[1,0] = BT_left[1]
        q[0,-1] = BT_right[0] 
        # q[1,-1] = BT_right[1]

        q1 = q[0]
        q2 = q[1]


        if stacked == True:
            h_stack.append(q1)
            u_num = np.where(h_num > 1e-10, np.divide(q[1],q[0]), 0*q[0])
            u_stack.append(u_num)
        

    if stacked == True:
        end = time.time()
        time_elapsed = end-start
        print('integrate with q',time_elapsed)
        return h_stack, u_stack
    else:
        h_num = q[0]
        u_num = np.where(h_num > 1e-10, np.divide(q[1],q[0]), 0*q[0])
        end = time.time()
        time_elapsed = end-start
        print('integrate with q',time_elapsed)
        return h_num, u_num

# ---------------------------------------------#
# Running the program with the setup
# and make a plot

h_num, u_num = integrate(domain_1d, initial_profile,simulation_time, boundary=False, source=False, stacked = False)
plt.plot(x,h_num,label='height')
plt.legend()
plt.ylim(-0.1,1.0)
# plt.plot(x_control,np.exp(-(x_control-0.5)**2/0.01))
# # plt.plot(x_control,h_num[-1] - np.exp(-(x_control-0.5)**2/0.01))
plt.show()

# ---------------------------------------------#
# Running the program with the setup
# and make an animation 
# from celluloid import Camera
# from IPython.display import HTML
# from tqdm import tqdm as td 

# fig, axs = plt.subplots()
# camera = Camera(fig)
# counter = np.int(finaltime/dt -1)

# h_num, u_num = integrate(domain_1d, initial_profile,simulation_time, boundary=False, source=False, stacked = True)


# for i in td(range(counter)):
#     plt.ylim(-.1,1)
#     axs.set_title('Height')
#     axs.plot(x_control,h_num[i],label="numerical",color="orange")
#     if mms != 0:
#         axs.plot(x_control,h_analytic[i],label="analytical",color="blue")
#     if i == 0:
#         axs.legend()
#     # plt.gcf().canvas.draw()
#     for ax in fig.get_axes():
#         ax.label_outer()
#     plt.gcf().canvas.draw()
#     camera.snap()

# animation = camera.animate(blit=True,interval=50)
# play = HTML(animation.to_html5_video())
# print('showing animation')
# # if saveanim != 0:
# #     animation.save('%s' % saveanim)
# plt.show()
# ---------------------------------------------#

# %%

# Define the right hand side of the differential equations
    
def RHS2(t,h,u,dx,N):
    q1 = h
    q2 = h*u
    RHS_ = RHS(t,q1,q2,dx,N)
    RHS_h = RHS_[0]
    RHS_u = np.where(h>1e-6,(RHS_[1] - RHS_[0]*q2/h)/h, RHS_[1]*0)
    return RHS_h, RHS_u

def integrate2(dx, initial_profile,simulation_time, boundary=False, source=False, stacked =True):
    '''
    Solving 1-dimensional shallow water wave equation using Finite Volume
    and HLL Solver.
    '''
    start = time.time()
    finaltime, dt  = simulation_time
    g      = 9.81                                   # the gravity constant 
    t = dt                                          # initialise first time step
    h_num, u_num = initial_profile                  # initialise numerical solution
    

    h_stack = []
    u_stack = []

    while t < finaltime:                            # main loop
    # for i in range(500):
        # Integrating with Runge-Kutta 
        f1_h, f1_u = RHS2(t     , h_num           , u_num            , dx,N)
        f2_h, f2_u = RHS2(t+dt/2, h_num+f1_h*dt/2, u_num +f1_u*dt/2, dx,N)
        f3_h, f3_u = RHS2(t+dt/2, h_num+f2_h*dt/2, u_num +f2_u*dt/2, dx,N)
        f4_h, f4_u = RHS2(t+dt  , h_num+f3_h*dt  , u_num +f3_u*dt  , dx,N)
        h_num += (dt/6)* (f1_h + 2*f2_h + 2*f3_h + f4_h)
        u_num += (dt/6)* (f1_u + 2*f2_u + 2*f3_u + f4_u)
        t += dt
            # q[0,0] = BT_left[0]
            # # q[1,0] = BT_left[1]
            # q[0,-1] = BT_right[0] 
            # # q[1,-1] = BT_right[1]
        h_num[0] = BT_left[0]
        h_num[-1] = BT_right[0] 
        u_num[0] = BT_left[1]
        u_num[-1] = BT_right[1]

        if stacked == True:
            h_stack.append(h_num)
            u_stack.append(u_num)        

    if stacked == True:
        end = time.time()
        time_elapsed = end-start
        print('integrate with h and u',time_elapsed)
        return h_stack, u_stack
    else:
        end = time.time()
        time_elapsed = end-start
        print('integrate with h and u',time_elapsed)        
        return h_num, u_num


BT_left = (np.array([[H],[U]]))
BT_right = (np.array([[H],[U]]))

h_num2, u_num2 = integrate2(domain_1d, initial_profile,simulation_time, boundary=False, source=False, stacked = False)
plt.plot(x,h_num2,label='height')
plt.ylim(-0.1,1.0)
plt.legend()
# plt.plot(x_control,np.exp(-(x_control-0.5)**2/0.01))
# # plt.plot(x_control,h_num[-1] - np.exp(-(x_control-0.5)**2/0.01))
plt.show()

# %%

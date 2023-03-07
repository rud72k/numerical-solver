#%%
import numpy as np
import matplotlib.pyplot as plt

# Define simulation parameter 
x0 = 0                          # Left boundary
xn = 1                          # Right boundary
N  = 501                          # Number of nodes - 1
x = np.linspace(x0,xn,N)        # intervals
dx = x[1] - x[0]                # grid spacing

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
finaltime  = 15.0                        # simulation time in seconds
dt = dt = cfl*dx/(np.sqrt(g))           # time step in seconds
simulation_time = (finaltime,dt)        #


# Define initial conditions
def initial_profile_1d(x):
    '''
    Control cell for Finite Volume is taking into account.
    '''
    h_init = np.exp(-(x_control-0.5)**2/0.005)
    u_init = x_control*0
    return h_init, u_init

initial_profile = initial_profile_1d(x_control)

# Forced solution 
mms = 0


# Define boundary conditions

BT_left = np.zeros((2,1))
BT_right = np.zeros((2,1))

# Define numerical flux function 

def flux(q1,q2):
    flux_1 = q2
    flux_2 = q2**2/q1 + 1/2 * g*q1**2
    flux = np.block([[flux_1], [flux_2]])
    return flux

# Define the right hand side of the differential equations
def RHS(t,q1,q2,dx):
    flux_left = flux(q1[:-1],q2[:-1])
    flux_left = np.block([[BT_left,flux_left]])
    flux_right = flux(q1[1:],q2[1:])
    flux_right = np.block([[flux_right,BT_right]]) 
    RHS = - flux_left + flux_right
    return RHS

def integrate(dx, initial_profile,simulation_time, boundary=False, source=False, stacked =True):
    '''
    Solving 1-dimensional shallow water wave equation using Finite Volume
    and HLL Solver.
    '''
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
    # for i in range(2000):
        # Integrating with Runge-Kutta 
        f1 = RHS(t     , q1           , q2            , dx)
        f2 = RHS(t+dt/2, q1+f1[0]*dt/2, q2 +f1[1]*dt/2, dx)
        f3 = RHS(t+dt/2, q1+f2[0]*dt/2, q2 +f2[1]*dt/2, dx)
        f4 = RHS(t+dt  , q1+f3[0]*dt  , q2 +f3[1]*dt  , dx)
        q += (dt/6)* (f1 + 2*f2 + 2*f3 + f4)
        t += dt
        # revert the variable and stack the solution
        q1 = q[0]
        q2 = q[1]

        if stacked == True:
            h_stack.append(q1)
            u_num = np.where(h_num > 1e-10, np.divide(q[1],q[0]), 0*q[0])
            u_stack.append(u_num)

    if stacked == True:
        return h_stack, u_stack
    else:
        h_num = q[0]
        u_num = np.where(h_num > 1e-10, np.divide(q[1],q[0]), 0*q[0])
        return h_num, u_num

# ---------------------------------------------#
# Running the program with the setup
# and make a plot

h_num, u_num = integrate(domain_1d, initial_profile,simulation_time, boundary=False, source=False, stacked = False)
plt.plot(x_control,h_num)
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

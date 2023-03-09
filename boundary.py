import numpy as py

def LinearSWEconstants(H_bar,U_bar, g):
    lambda1 = -(1/(2*g*H_bar))*np.sqrt(U_bar**2*(g+H_bar)**2 + 4*g*H_bar*( g*H_bar - U_bar**2)) + U_bar*(g+H_bar)/(2*g*H_bar)
    lambda2 =  (1/(2*g*H_bar))*np.sqrt(U_bar**2*(g+H_bar)**2 + 4*g*H_bar*( g*H_bar - U_bar**2)) + U_bar*(g+H_bar)/(2*g*H_bar)
    c = (np.sqrt((lambda1 - U_bar/g)**2 + 1))
    d = (np.sqrt((lambda2 - U_bar/g)**2 + 1))
    return lambda1,lambda2,c,d

def BoundaryCondition(H_bar,U_bar, g):
    '''Setting up the boundary condition'''
    lambda1,lambda2,c,d = LinearSWEconstants(H_bar,U_bar, g)

    ##---------------------------------------------------------------------------------------##
    ## Outflow boundary condition
    p_0 = (1/c)*(np.sqrt(g/H_bar)*(lambda1 - U_bar/g) + 1)
    q_0 = (1/d)*(np.sqrt(g/H_bar)*(lambda2 - U_bar/g) + 1)

    p_N = (1/c)*(-np.sqrt(g/H_bar)*(lambda1 - U_bar/g) + 1)
    q_N = (1/d)*(-np.sqrt(g/H_bar)*(lambda2 - U_bar/g) + 1)
    
    ##---------------------------------------------------------------------------------------##

    ##---------------------------------------------------------------------------------------##
    ##  H_bar u + U_bar h = 0
    #p_N = 1/d*(H_bar + U_bar*(lambda2 - U_bar/g))
    #q_N = 1/c*(H_bar + U_bar*(lambda1 - U_bar/g))
    ##  H_bar u + U_bar h = 0
    #q_0 = 1/d*(H_bar + U_bar*(lambda2 - U_bar/g))
    #p_0 = 1/c*(H_bar + U_bar*(lambda1 - U_bar/g))
    ##---------------------------------------------------------------------------------------##

    ##---------------------------------------------------------------------------------------##
    #
    # h = 0 at x = 1
    # p_N = 1/c*(lambda1 - U_bar/g)
    # q_N = 1/d*(lambda2 - U_bar/g)
    # gammaN = - q_N/p_N         # h =0     / 
    #
    #
    # u = 0 at x = 1
    # p_N = 1/c
    # q_N = 1/d
    gammaN = - c/d           # u = 0    
    #
    ##---------------------------------------------------------------------------------------##
    
    # Boundary condition


    # q_0 = 1/d*(H_bar + U_bar*(lambda2 - U_bar/g))
    # p_0 = 1/c*(H_bar + U_bar*(lambda1 - U_bar/g)) # H_bar * u + U_bar * h = 0

    # p_N = 1/c*((lambda1 - U_bar/g))
    # q_N = 1/d*((lambda2 - U_bar/g))           # ?


    # q_N = (1/d)*(lambda2 - U_bar/g + 1)
    # p_N = (1/c)*(lambda1 - U_bar/g + 1)     # ? recheck


    # gamma0 = - p_0/q_0
    #gammaN = - p_N/q_N         #  H_bar u + U_bar h = 0

    # gammaN = - q_0/p_0          # H_bar u + U_bar h = 0
    
    gamma0 = -p_0/q_0 
    # gammaN = -q_N/p_N 
    return gamma0, gammaN

def BoundaryTerms(h,u,H_bar,U_bar,g, h_analytic, u_analytic):
    '''Set up the Boundary Condition'''
    lambda1,lambda2,c,d = LinearSWEconstants(H_bar,U_bar, g)
    gamma0, gammaN = BoundaryCondition(H_bar,U_bar, g)
    w_1 = ((h-h_analytic)*(lambda1 - U_bar/g)/c) + (u-u_analytic)/c
    w_2 = ((h-h_analytic)*(lambda2 - U_bar/g)/d) + (u-u_analytic)/d
    BC_0 = (w_2[0]-gamma0*w_1[0])
    BC_N = (w_1[-1]-gammaN*w_2[-1])
    # return 0,0
    return BC_0, BC_N

def penaltyparameter(H_bar,U_bar, g):
    lambda1,lambda2,c,d = LinearSWEconstants(H_bar,U_bar, g)
    tau_01 = H_bar*lambda2*(lambda2 - U_bar/g)/d
    tau_02 = g*(lambda2)/d
    tau_N1 = - H_bar*lambda1*(lambda1 - U_bar/g)/c
    tau_N2 = - lambda1*g/c 
    tau = [tau_01,tau_N1,tau_02,tau_N2]
    #return [0,0,0,0]
    return tau
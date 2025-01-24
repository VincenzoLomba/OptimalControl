#
# SQP method for constrained optimization
# Lorenzo Sforni
# Bologna, 26/10/2023
#


import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})


PLOT = False
##############################
# SQP solver
##############################

def SQP_solver(zk, dl, hk, dh, Bk):
  """
      SQP solver using CVXPY
    
      min_{z \in \R^n} dl^T (z - zk) + 0.5*(z-zk)^T B (z-xk)
      subj.to h + dh^T (z-xk) = 0
  """   

  z = cvx.Variable(zk.shape)

  cost = dl.T@(z - zk) + 0.5*cvx.quad_form(z - zk, Bk)
  constraint = [hk + dh.T@(z-zk) == 0]

  problem = cvx.Problem(cvx.Minimize(cost), constraint)
  problem.solve()

  x_star = z.value  
  lambda_QP = constraint[0].dual_value


  return x_star, lambda_QP
  
  
##############################
# Problem
##############################

# Parameters
r = 1; z0 = 0; y0 = 2 # constraint parameters

def cost_fcn(zz):
    """
    Input
    - Optimization variable
        z \in R^2,  z = [z1 z2]
    Output
    - ll cost at z 
        l(z)    = 0.5 z\T Q z + q\T z
    - dl gradient of ll at z, \nabla l(z)
        dl(z)   = Q z + q
    - d2l hessian of ll at z, \nabla^2 l(z)
        d2l(z)  = Q
    """

    Q = np.array([[2,1],[1,4]])
    # Q = np.array([[0.1,0.1],[0.1,0.4]])
    q = np.array([2,0])

    ll = 0.5*zz.T@Q@zz + q@zz
    dl = Q@zz + q
    d2l = Q


    return ll, dl, d2l

def equality_constraint_fcn(zz):
    """
    Equality constraint
    - h(z) = || z - z_c ||^2 - r^2 = 0
    - dh(z) = 2(z - z_c)
    """

    zzc = np.array([z0, y0])

    hh = (zz[0] - zzc[0])**2 + (zz[1] - zzc[1])**2 - r**2
    dh = 2*np.array([zz[0] - zzc[0], zz[1] - zzc[1]])

    return hh, dh

def merit_fcn(zk, lmb, deltaz):
    """
     Merit function
     - M1(z) = l(z) + s || g(z) ||_1
     - Directional derivative
        D_Dz M1(z) = dl(z)^T Dz - s || g(z) ||_1
        s > || lmd ||_{inf}
    """

    # Compute cost and gradient
    lk, dlk = cost_fcn(zk)[:2]

    # Compute constraint and gradient
    hk = equality_constraint_fcn(zk)[0] 

    s = 100*np.ceil(np.linalg.norm([lmb], np.inf))

    M1 = lk + s*np.linalg.norm([hk], 1)

    DM1 = dlk.T@deltaz - s*np.linalg.norm([hk], 1)

    return M1, DM1.squeeze()


##############################
# Optimization
##############################


max_iters = int(1e2)
n_z = 2           # state variable dimension

# Choose methods

methods = ['SQP', 'Newton']
method = 'SQP'

# ARMIJO PARAMETERS

stepsize_0 = 1
cc = 0.5
beta = 0.7
armijo_maxiters = 10 # number of Armijo iterations

# Initialize state, cost and gradient variables (for each algorithm)
zz = np.zeros((n_z, max_iters))
lmbd = np.zeros((max_iters))
ll_mult = np.zeros((n_z, max_iters))  # multipliers

ll = np.zeros((max_iters-1))
dl = np.zeros((n_z, max_iters-1))
hh = np.zeros((max_iters-1))
dh = np.zeros((n_z, max_iters-1))
BB = np.zeros((n_z, n_z, max_iters-1))

descent_norm = np.zeros(max_iters-1) #[for plots]
lagrangian_norm = np.zeros(max_iters-1) #[for plots]


z_init = [0,2.1]    # initial condition s.t. regularization is required
z_init = [-2,2]
zz[:,0] = z_init 

stepsize = stepsize_0

# Algorithm
for kk in range(max_iters-1):

    # Compute cost and gradient
    ll[kk], dl[:,kk], d2l= cost_fcn(zz[:,kk]) 

    # Compute constraint and gradient
    hh[kk], dh[:,kk] = equality_constraint_fcn(zz[:,kk]) 
    
    BB[:,:,kk] = d2l + 2*lmbd[kk]*np.eye(n_z) # Exact Hessian
    # BB[:,:,kk] = d2l                           # Gauss-Newton Approximation

    # check positive definiteness of BB
    print(np.linalg.eigvals(BB[:,:,kk]))
    if np.any(np.linalg.eigvals(BB[:,:,kk]) <= 0):
        print('Hessian not positive definite')
        break

    if method == methods[0]:

        # SQP solver
        zqp, lmb_qp = SQP_solver(zz[:,kk], dl[:,kk], hh[kk], dh[:,kk], BB[:,:,kk])
        #
        # compute the direction and multiplier
        direction = (zqp - zz[:,kk])
        multiplier = lmb_qp

    elif method == methods[1]:

        # build block matrix
        W = np.block([[BB[:,:,kk],dh[:,kk][:,None]], [dh[:,kk][:,None].T, 0]])
        #
        # compute newton direction solving linear system
        newt_dir = np.linalg.solve(W, np.block([-dl[:,kk], -hh[kk]]))
        #
        # compute the direction and multiplier
        direction = newt_dir[:-1]
        multiplier = newt_dir[-1]


    descent_norm[kk] = np.linalg.norm(direction) #[for plots]
    lagrangian_norm[kk] = np.linalg.norm(dl[:,kk] + dh[:,kk]*multiplier, np.inf) #[for plots]

    ############################
    # Armijo stepsize selection
    ############################

    stepsizes = []  # list of stepsizes
    merits_armijo = []

    stepsize = stepsize_0

    M1k,DM1k = merit_fcn(zz[:,kk], multiplier, direction)


    for ii in range(armijo_maxiters):
        
        zzp_temp = zz[:,kk] + stepsize*direction   # temporary update

        MM_temp = merit_fcn(zzp_temp, multiplier, direction)[0]

        stepsizes.append(stepsize)      # save the stepsize
        merits_armijo.append(MM_temp)    # save the cost associated to the stepsize

        if MM_temp > M1k + cc*stepsize*DM1k:
            
            # update the stepsize
            stepsize = beta*stepsize
        
        else:
            print('Armijo stepsize = {}'.format(stepsize))
            break

    ############################
    # Descent plot
    ############################

    steps = np.linspace(0,stepsize_0,int(1e2))
    MM = np.zeros(len(steps))

    M1k,DM1k = merit_fcn(zz[:,kk], multiplier, direction)

    for ii in range(len(steps)):

        step = steps[ii]

        zzp_temp = zz[:,kk] + step*direction   # temporary update

        MM[ii] = merit_fcn(zzp_temp, multiplier, direction)[0]

    if PLOT:
        ############################
        # Plot
        #
        plt.figure(1)
        plt.clf()
        #
        plt.title('Descent (Merit function)')
        plt.plot(steps, MM, color='g', label='$M_1(z^k + \\gamma d^k)$')
        plt.plot(steps, M1k + steps*DM1k, color='r', \
                label='$M_1(z^k) + \\gamma\mathrm{D}_{d^k} M_1(z^k, \lambda^k)$')
        plt.plot(steps, M1k + cc*steps*DM1k, color='g', linestyle='dashed', \
                label='$M_1(z^k) + c\\gamma \mathrm{D}_{d^k} M_1(z^k, \lambda^k)$')
        #
        plt.scatter(stepsizes, merits_armijo, marker='*') # plot the tested stepsize
        #
        plt.grid()
        plt.xlabel('stepsize')
        plt.legend()
        plt.draw()
        #
        plt.show()
        ############################
    
    ############################
    # Update solution and multiplier
    #
    zz[:,kk+1] = zz[:,kk] + stepsize*direction
    lmbd[kk+1] = (1 - stepsize)*lmbd[kk] + stepsize*multiplier



    print(descent_norm[kk])
    print(f'LagNorm = {np.linalg.norm(dl[:,kk] + dh[:,kk]*lmbd[kk], np.inf)}')
    print(f'ConstrNorm = {np.linalg.norm(hh[kk])}')
    print(lmbd[kk+1])
    if kk%1e2 == 0:
        print('ll_{} = {}'.format(kk,ll[kk]))

    if np.linalg.norm(direction) <= 1e-10:
    
        max_iters = kk+1

        break


# Compute optimal solution via cvx solver [plot]
print('zz_max = {}'.format(zz[:,max_iters-1]))

######################################################
# Plots
######################################################


if 1:

    plt.figure('descent direction')
    plt.plot(np.arange(max_iters-1), descent_norm[:max_iters-1])
    plt.xlabel('$k$')
    plt.ylabel('||$d\ell(z^k)||$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)

    plt.figure('lagrangian norm')
    plt.plot(np.arange(max_iters-1), lagrangian_norm[:max_iters-1])
    plt.xlabel('$k$')
    plt.ylabel('$||\\nabla_z\mathcal{L}(z^k)||$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)

plt.figure()
plt.rcParams.update({'font.size': 12})

domain_z = np.arange(-10,10,0.1)
domain_y = np.arange(-10,10,0.1)
domain_z, domain_y = np.meshgrid(domain_z, domain_y)
cost_on_domain = np.zeros(domain_z.shape)

# draw constraint
t = np.linspace(0, 2*np.pi, 100)
z1_t = z0 + r*np.cos(t)
z2_t = y0 + r*np.sin(t)

# evaluate cost on z_t, y_t
ft = np.zeros(z1_t.shape)
for ii in range(z1_t.shape[0]):
    ft[ii] = cost_fcn(np.array((z1_t[ii],z2_t[ii])))[0]


for ii in range(domain_z.shape[0]):
    for jj in range(domain_z.shape[1]):
        cost_on_domain[ii,jj] = np.amin([cost_fcn(np.array((domain_z[ii,jj],domain_y[ii,jj])))[0],4e2]) # take only the cost + saturate (for visualization)

ax = plt.axes(projection='3d')
ax.plot_surface(domain_z, domain_y, cost_on_domain, cmap='Blues', linewidth = 0, alpha=0.4)
ax.plot3D(zz[0,:max_iters-1], zz[1,:max_iters-1], ll[:max_iters-1], color = 'tab:orange')
ax.scatter3D(zz[0,:max_iters-1], zz[1,:max_iters-1], ll[:max_iters-1], color = 'tab:orange', s=50)
ax.plot3D(z1_t, z2_t, ft, 'r', linewidth=2)
ax.set_xlabel('$z^k_0$')
ax.set_ylabel('$z^k_1$')
ax.set_zlabel('$\ell(z^k)$')

plt.show()
# Gradient Method - Vectorial
# Lorenzo Sforni
# Bologna, 16/10/2023

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # used for 3D plot
import cvxpy as cvx


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

PLOT = True #True to show cost plots at each iteration

######################################################
# Functions
######################################################

def cost_fcn(zz):
    """
    Input
    - Optimization variable
        z \in R^2,  z = [z1 z2]
    Output
    - ll  (nonlinear) cost at z 
        l(z) = exp(z1+3*z2-0.1) + exp(z1-3z2-0.1) + exp(-z1-0.1),
    - dl gradient of ll at z, \nabla l(z)
    """
    ll = np.exp(zz[0] + 3*zz[1]-0.1) + np.exp(zz[0]-3*zz[1]-0.1) + np.exp(-zz[0]-4)
    # Compute gradient components 
    dl1 = np.exp(zz[0] + 3*zz[1] - 0.1) + np.exp(zz[0] - 3*zz[1] - 0.1) - np.exp(-zz[0] - 4)
    dl2 = 3*np.exp(zz[0] + 3*zz[1] - 0.1) - 3*np.exp(zz[0] - 3*zz[1] - 0.1)

    dl = np.array([dl1,dl2])
    return ll, dl

def min_cvx_solver(zz):
    """
    Off-the-shelf solver - check exact solution
    Have a look at cvxpy library: https://www.cvxpy.org/
    """
    zz = cvx.Variable(len(zz))
    cost = cvx.exp(zz[0] + 3*zz[1]-0.1) + cvx.exp(zz[0]-3*zz[1]-0.1) + cvx.exp(-zz[0]-4)
  
    problem = cvx.Problem(cvx.Minimize(  
        cost
    ))
    #problem.solve() # Naive implementation, no tolerances -> check gradient of z_star
    problem.solve(solver = 'ECOS', abstol=1e-8) # Smart implementation, choose solver and tolerance
    return zz.value, problem.value

######################################################
# Main code
######################################################

max_iters = 500
stepsize = 0.01
n_z = 2           # state variable dimension

# cc =0.5


# Initialize state, cost and gradient variables (for each algorithm)
zz = np.zeros((n_z, max_iters))
ll = np.zeros((max_iters-1))
dl = np.zeros((n_z, max_iters-1))
dl_norm = np.zeros(max_iters-1) #[for plots]

# Set initial condition for each state variable
# plot domain z1 \in [-3.0,-0.0]
#             z2 \in [-1.5, 1.5]
zz_init = [-2,-1]
zz[:,0] = zz_init 

# Algorithm
plt.figure()
for kk in range(max_iters-1):

    # Compute cost and gradient
    ll[kk], dl[:,kk] = cost_fcn(zz[:,kk]) 
    dl_norm[kk] = np.linalg.norm(dl[:,kk]) #[for plots]

    # select the direction
    direction = - dl[:,kk]

    if PLOT:
        ############################
        # Descent plot
        ############################

        steps = np.linspace(0,1,int(1e3))
        costs = np.zeros(len(steps))

        for ii in range(len(steps)):

            step = steps[ii]

            zzp_temp = zz[:,kk] + step*direction   # temporary update

            costs[ii] = cost_fcn(zzp_temp)[0]


        

        plt.clf()
        plt.title('Descent')
        plt.plot(steps, costs, color='g', label='$\\ell(z^k + stepsize*d^k$)')
        plt.plot(steps, ll[kk] + dl[:,kk].T@direction*steps, color='r', label='$\\ell(z^k) + stepsize*\\nabla\\ell(z^k)^{\\top}d^k$')
        # plt.plot(steps, ll[kk] + cc*dl[:,kk].T@direction*steps, color='g', linestyle='dashed', label='$\\ell(z^k) + stepsize*c*\\nabla\\ell(z^k)^{\\top}d^k$')
        plt.grid()
        plt.legend()
        plt.show(block=False)

    ############################
    ############################

    zz[:,kk+1] = zz[:,kk] + stepsize * direction

    print('ll_{} = {}'.format(kk,ll[kk]), '\tx_{} = {}'.format(kk+1,zz[:,kk+1]))

    if np.linalg.norm(direction) <= 1e-4:
        
        max_iters = kk+1

        break


    plt.pause(0.05)
# Compute optimal solution via cvx solver [plot]
zz_star, ll_star = min_cvx_solver(zz_init)
print('ll_star = {}'.format(ll_star), '\tz_star = {}'.format(zz_star))
print('dl_star = {}'.format(cost_fcn(zz_star)[1]))


######################################################
# Plots
######################################################

plt.figure('cost decrease')
plt.plot(np.arange(max_iters-1), ll[:max_iters-1])
plt.plot(np.arange(max_iters-1), ll_star*np.ones(max_iters-1), linestyle = 'dashed')
plt.xlabel('$k$')
plt.ylabel('$\ell(z^k)$')
plt.grid()
plt.show(block=False)

if 1:

    plt.figure('descent direction')
    plt.plot(np.arange(max_iters-1), dl_norm[:max_iters-1])
    plt.xlabel('$k$')
    plt.ylabel('||$d\ell(z^k)||$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)


    plt.figure('cost error')
    plt.plot(np.arange(max_iters-1), abs(ll[:max_iters-1]-ll_star))
    plt.xlabel('$k$')
    plt.ylabel('$||\ell(z^{k})-\ell(z^{*})||$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)

plt.figure()
plt.rcParams.update({'font.size': 12})
domain_x = np.arange(-3,3,0.1)
domain_y = np.arange(-3,3,0.1)
domain_x, domain_y = np.meshgrid(domain_x, domain_y)
cost_on_domain = np.zeros(domain_x.shape)

for ii in range(domain_x.shape[0]):
    for jj in range(domain_x.shape[1]):
        cost_on_domain[ii,jj] = np.amin([cost_fcn(np.array((domain_x[ii,jj],domain_y[ii,jj])))[0],4e2]) # take only the cost + saturate (for visualization)

ax = plt.axes(projection='3d')
ax.plot_surface(domain_x, domain_y, cost_on_domain, cmap='Blues', linewidth = 0, alpha=0.8)
ax.plot3D(zz[0,:max_iters-1], zz[1,:max_iters-1], ll[:max_iters-1], color = 'tab:orange')
ax.scatter3D(zz[0,:max_iters-1], zz[1,:max_iters-1], ll[:max_iters-1], color = 'tab:orange', s=50)
ax.set_xlabel('$z^k_0$')
ax.set_ylabel('$z^k_1$')
ax.set_zlabel('$\ell(z^k)$')

plt.show()
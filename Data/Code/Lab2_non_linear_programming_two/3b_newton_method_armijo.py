#
# Gradient Method - Newton's Method
# Lorenzo Sforni
# Bologna, 09/10/2023
#

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

PLOT = True   #True to show cost plots at each iteration

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
    dl2 = 3.*np.exp(zz[0] + 3*zz[1] - 0.1) - 3*np.exp(zz[0] - 3*zz[1] - 0.1)

    d2l11 = np.exp(zz[0] + 3*zz[1] - 0.1) + np.exp(zz[0]-3*zz[1]-0.1) + np.exp(-zz[0]-4)
    d2l22 = 9*np.exp(zz[0] + 3*zz[1] - 0.1) + 9*np.exp(zz[0]-3*zz[1]-0.1)
    d2l12 = 3.*np.exp(zz[0] + 3*zz[1] - 0.1) - 3*np.exp(zz[0] - 3*zz[1] - 0.1)

    dl = np.array([dl1,dl2])
    d2l = np.array([[d2l11, d2l12], [d2l12, d2l22]])
    return ll, dl, d2l

def min_cvx_solver(zz):
    """
    Off-the-shelf solver - check exact solution
    Have a look at cvxpy library: https://www.cvxpy.org/
    """
    zz = cvx.Variable(zz.shape[0])
    cost = cvx.exp(zz[0] + 3*zz[1]-0.1) + cvx.exp(zz[0]-3*zz[1]-0.1) + cvx.exp(-zz[0]-4)
  
    problem = cvx.Problem(cvx.Minimize(  
        cost
    ))
    problem.solve() # Naive implementation, no tolerances -> check gradient of z_star
    #problem.solve(solver = 'ECOS', abstol=1e-8) # Smart implementation, choose solver and tolerance
    return zz.value, problem.value


######################################################
# Main code
######################################################

# max_iters = int(5e2)
max_iters = 20

stepsize_0 = 1
n_z = 2           # state variable dimension

max_steps_armijo = 10

cc = 0.5
beta = 0.7

# Initialize state, cost and gradient variables (for each algorithm)
zz = np.zeros((n_z, max_iters))
ll = np.zeros((max_iters-1))
dl = np.zeros((n_z, max_iters-1))
d2l = np.zeros((n_z,n_z, max_iters-1))
dl_norm = np.zeros(max_iters-1) #[for plots]

# Set initial condition for each state variable
# plot domain x0 \in [-3.0,-0.0]
#             x1 \in [-1.5, 1.5]
zz_init = np.array([-2,1])
zz[:,0] = zz_init 

# Algorithm
plt.figure()
for kk in range(max_iters-1):

    # Compute cost and gradient

    # Gradient Method
    ll[kk], dl[:,kk], d2l[:,:,kk] = cost_fcn(zz[:,kk])
    dl_norm[kk] = np.linalg.norm(dl[:,kk]) #[for plots]

    direction = - np.linalg.inv(d2l[:,:,kk])@dl[:,kk]


    ############################
    # Armijo stepsize selection
    ############################

    stepsizes = []  # list of stepsizes
    costs_armijo = []

    stepsize = stepsize_0

    for ii in range(10):
        
        zzp_temp = zz[:,kk] + stepsize*direction   # temporary update

        ll_temp = cost_fcn(zzp_temp)[0]

        stepsizes.append(stepsize)      # save the stepsize
        costs_armijo.append(ll_temp)    # save the cost associated to the stepsize

        if ll_temp > ll[kk] + cc*stepsize*dl[:,kk].T@direction:
            
            # update the stepsize
            stepsize = beta*stepsize
        
        else:
            print('Armijo stepsize = {}'.format(stepsize))
            break

        if ii == max_steps_armijo : 
            print("Warning: no stepsize was found with Armijo rule!")

    if PLOT:
        ############################
        # Descent plot
        ############################

        steps = np.linspace(0,1,int(1e2))
        costs = np.zeros(len(steps))

        for ii in range(len(steps)):

            step = steps[ii]

            zzp_temp = zz[:,kk] + step*direction   # temporary update

            costs[ii] = cost_fcn(zzp_temp)[0]

        # plt.figure()

        plt.clf()
        plt.title('Descent (Second-order)')
        plt.plot(steps, costs, color='g', label='$\\ell(z^k - \\gamma*d^k$)')
        plt.plot(steps, ll[kk] + dl[:,kk].T@direction*steps, color='r', label='$\\ell(z^k) - \\gamma*\\nabla\\ell(z^k)^{\\top}d^k$')
        plt.plot(steps, ll[kk] + dl[:,kk].T@direction*steps + 1/2*direction.T@d2l[:,:,kk]@direction*steps**2, color='b', label='$\\ell(z^k) - \\gamma*\\nabla\\ell(z^k)^{\\top}d^k - \\gamma^2 d^{k\\top}\\nabla^2\\ell(z^k) d^k$')
        plt.plot(steps, ll[kk] + cc*dl[:,kk].T@direction*steps, color='g', linestyle='dashed', label='$\\ell(z^k) - \\gamma*c*\\nabla\\ell(z^k)^{\\top}d^k$')

        plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

        plt.grid()
        plt.legend()

        plt.show(block=False)

    ############################
    ############################


    zz[:,kk+1] = zz[:,kk] + stepsize * direction

    print('Iter {}\tCost: = {}'.format(kk,ll[kk]))

    if np.linalg.norm(direction) <= 1e-4:
    
        max_iters = kk+1

        break

# Compute optimal solution via cvx solver [plot]
zz_star, ll_star = min_cvx_solver(zz_init)
print('ll_star = {}'.format(ll_star), '\tz_star = {}'.format(zz_star))
print('dl_star = {}'.format(cost_fcn(zz_star)[1]))

######################################################
# Plots
######################################################


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
domain_z = np.arange(-3,3,0.1)
domain_y = np.arange(-3,3,0.1)
domain_z, domain_y = np.meshgrid(domain_z, domain_y)
cost_on_domain = np.zeros(domain_z.shape)

for ii in range(domain_z.shape[0]):
    for jj in range(domain_z.shape[1]):
        cost_on_domain[ii,jj] = np.amin([cost_fcn(np.array((domain_z[ii,jj],domain_y[ii,jj])))[0],4e2]) # take only the cost + saturate (for visualization)

ax = plt.axes(projection='3d')
ax.plot_surface(domain_z, domain_y, cost_on_domain, cmap='Blues', linewidth = 0, alpha=0.8)
ax.plot3D(zz[0,:max_iters-1], zz[1,:max_iters-1], ll[:max_iters-1], color = 'tab:blue')
ax.scatter3D(zz[0,:max_iters-1], zz[1,:max_iters-1], ll[:max_iters-1], color = 'tab:blue')
ax.set_xlabel('$z^k_0$')
ax.set_ylabel('$z^k_1$')
ax.set_zlabel('$\ell(z^k)$')

plt.show()


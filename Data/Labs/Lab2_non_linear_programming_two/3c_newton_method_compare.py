#
# Gradient Method - Newton's Method
# Lorenzo Sforni
# Bologna, 09/10/2023
#

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})


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

stepsize = 0.01
n_z = 2           # state variable dimension

# Initialize state, cost and gradient variables (for each algorithm)
zz_grad = np.zeros((n_z, max_iters))
ll_grad = np.zeros((max_iters-1))
dl_grad = np.zeros((n_z, max_iters-1))
dl_norm_grad = np.zeros(max_iters-1) #[for plots]

zz_newt = np.zeros((n_z, max_iters))
ll_newt = np.zeros((max_iters-1))
dl_newt = np.zeros((n_z, max_iters-1))
d2l_newt = np.zeros((n_z,n_z, max_iters-1))
dl_norm_newt = np.zeros(max_iters-1) #[for plots]

# Set initial condition for each state variable
# plot domain x0 \in [-3.0,-0.0]
#             x1 \in [-1.5, 1.5]
zz_init = np.array([-2,1])
zz_grad[:,0] = zz_init 
zz_newt[:,0] = zz_init 

# Algorithm
for kk in range(max_iters-1):

    # Compute cost and gradient

    # Gradient Method
    ll_grad[kk], dl_grad[:,kk] = cost_fcn(zz_grad[:,kk])[:-1]
    dl_norm_grad[kk] = np.linalg.norm(dl_grad[:,kk]) #[for plots]

    # Newton Method
    ll_newt[kk], dl_newt[:,kk], d2l_newt[:,:,kk] = cost_fcn(zz_newt[:,kk])
    dl_norm_newt[kk] = np.linalg.norm(dl_newt[:,kk]) #[for plots]

    zz_grad[:,kk+1] = zz_grad[:,kk] - stepsize * dl_grad[:,kk]
    zz_newt[:,kk+1] = zz_newt[:,kk] - np.linalg.inv(d2l_newt[:,:,kk])@dl_newt[:,kk]

    print('Iter {}\tCost:\tGradient = {}, Newton = {}'.format(kk,ll_grad[kk], ll_newt[kk]))

# Compute optimal solution via cvx solver [plot]
zz_star, ll_star = min_cvx_solver(zz_init)
print('ll_star = {}'.format(ll_star), '\tz_star = {}'.format(zz_star))
print('dl_star = {}'.format(cost_fcn(zz_star)[1]))

######################################################
# Plots
######################################################

legend = ['Gradient', 'Newton']

plt.figure('descent direction')
plt.plot(np.arange(max_iters-1), dl_norm_grad)
plt.plot(np.arange(max_iters-1), dl_norm_newt)
plt.xlabel('$k$')
plt.ylabel('||$d\ell(z^k)||$')
plt.yscale('log')
plt.grid()
plt.legend(legend)
plt.show(block=False)

plt.figure('cost error')
plt.plot(np.arange(max_iters-1), abs(ll_grad-ll_star))
plt.plot(np.arange(max_iters-1), abs(ll_newt-ll_star))
plt.xlabel('$k$')
plt.ylabel('$||\ell(z^{k})-\ell(z^{*})||$')
plt.legend(legend)
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
ax.plot3D(zz_newt[0,:max_iters-1], zz_newt[1,:max_iters-1], ll_newt, color = 'tab:orange')
ax.scatter3D(zz_newt[0,:max_iters-1], zz_newt[1,:max_iters-1], ll_newt, color = 'tab:orange')
ax.plot3D(zz_grad[0,:max_iters-1], zz_grad[1,:max_iters-1], ll_grad, color = 'tab:blue')
ax.scatter3D(zz_grad[0,:max_iters-1], zz_grad[1,:max_iters-1], ll_grad, color = 'tab:blue')
ax.set_xlabel('$z^k_0$')
ax.set_ylabel('$z^k_1$')
ax.set_zlabel('$\ell(z^k)$')

plt.show()


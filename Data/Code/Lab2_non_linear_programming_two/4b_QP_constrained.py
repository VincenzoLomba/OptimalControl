#
# Gradient Method - QP
# Lorenzo Sforni
# Bologna, 18/10/2022
#

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

######################################################
# Functions
######################################################
def cost_fcn(zz, QQ, qq):
    """
    Input
    - Optimization variable
        z \in R^n,  z = [z1 z2 ... zn]
    Output
    - ll  cost at z 
        l(z) = 0.5 * z.T*Q*z + q.T*z,
    - dl gradient of ll at z, \nabla l(z)
    """
    ll  = 0.5 * zz.T @ QQ @ zz + qq.T @ zz
    dl = 0.5 * (QQ+QQ.T) @ zz + qq
    return ll, dl

def min_cvx_solver(QQ, qq, AA, bb):
    """
    Off-the-shelf solver - check exact solution
    Have a look at cvxpy library: https://www.cvxpy.org/

    Obtain optimal solution for constrained QP

        min_{z} 1/2 z^T Q z + q^T z
        s.t.    Az - b <= 0

    """
    zz = cvx.Variable(qq.shape)

    # Cost function
    cost = 0.5* cvx.quad_form(zz,QQ) + qq.T @ zz

    # Constraint Az <= b
    constraint = [AA@zz <= bb]

    problem = cvx.Problem(cvx.Minimize(cost), constraint)
    problem.solve()
    return zz.value, problem.value

def projection(xx, AA, bb):
    """
        Projection using CVXPY

        min_{z \in \R^n} || x - z||^2
        s.t. Ax - b <= 0
    
    """

    zz = cvx.Variable(xx.shape)

    cost = cvx.norm(zz - xx, 2)
    constraint = [AA@zz <= bb]

    problem = cvx.Problem(cvx.Minimize(cost), constraint)
    problem.solve()

    return zz.value


######################################################
# Main code
######################################################

np.random.seed(10)

max_iters = int(5e2)
stepsize = 1e-1
n_z = 5 # state variable dimension

n_p = 4 # constraints dimension            

# Set problem parameters

# Cost

QQ = np.diag(np.random.rand(n_z))
qq = np.random.rand(n_z)

# Constraint
AA = np.random.rand(n_p,n_z)
bb = 3*np.random.rand(n_p)

# Initialize state, cost and gradient variables (for each algorithm)
zz = np.zeros((n_z, max_iters))
ll = np.zeros((max_iters-1))
dl = np.zeros((n_z, max_iters))

# Set initial condition for algorithms
# Find feasible initial condition
zz_init = np.random.rand(n_z)
while not np.all(AA@zz_init - bb < 0):
    zz_init = np.random.rand(n_z)
    print(AA@zz_init - bb)


zz[:,0] = zz_init 

# Algorithm
for kk in range(max_iters-1):

    # compute cost and gradient
    ll[kk], dl[:,kk] = cost_fcn(zz[:,kk], QQ, qq) 

    # Select the direction
    direction = - dl[:,kk]

    # Update the solution
    zz_temp = zz[:,kk] + stepsize * direction

    # Projection step
    zz[:,kk+1] = projection(zz_temp, AA, bb)

    print('ll_{} = {}'.format(kk,ll[kk]), '\tx_{} = {}'.format(kk+1,zz[:,kk+1]))

# Compute optimal solution via cvx solver [plot]
zz_star, ll_star = min_cvx_solver(QQ, qq, AA, bb)
print('ll_star = {}'.format(ll_star), '\tx_star = {}'.format(zz_star))

######################################################
# Plots
######################################################

plt.figure()
plt.plot(np.arange(max_iters-1), ll)
plt.plot(np.arange(max_iters-1), ll_star*np.ones(max_iters-1), linestyle = 'dashed')
plt.xlabel('$k$')
plt.ylabel('$\ell(z^k)$')
plt.legend(['$\ell(z^{k})$','$\ell(z^{*})$'])
plt.grid()


plt.figure()
plt.plot(np.arange(max_iters-1), abs(ll-ll_star))
plt.title('cost error')
plt.xlabel('$k$')
plt.ylabel('$||\ell(z^{k})-\ell(z^{*})||$')
plt.yscale('log')
plt.grid()


plt.show()
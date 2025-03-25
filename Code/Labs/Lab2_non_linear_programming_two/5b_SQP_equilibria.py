#
# SQP method for equilibrium finding 
# Marco Falotico
# Bologna, 28/10/2024
#


import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx


# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

# plt.rcParams["figure.figsize"] = (10,8)
# plt.rcParams.update({'font.size': 22})

##############################
# Optimization
##############################

max_iters = int(300)
n_z = 3          # decision variable dimension

# Define parameters
m = 1.0  # Mass 
k = 10   # Stiffness coefficient
x0 = 4   #spring rest position

xref = np.array([10, 0])
u_eq = k*(xref[0] - x0)
stepsize = 0.1

##############################
# SQP solver
##############################


def SQP_solver(zk, dl, hk, dh, Bk):
  """
      SQP solver using CVXPY
    
      min_{x \in \R^n} dl^T (z - zk) + 0.5*(z-zk)^T B (z-zk)
      subj.to h + dh^T (z-zk) = 0
  """   

  z = cvx.Variable(zk.shape)

  cost = dl.T@(z - zk) + 0.5*cvx.quad_form(z - zk, Bk)
  constraint = [hk + dh.T@(z-zk) == 0]

  problem = cvx.Problem(cvx.Minimize(cost), constraint)
  problem.solve()

  z_star = z.value
  lambda_QP = constraint[0].dual_value


  return z_star, lambda_QP

def cost_fcn(xx,uu,Q,r):
    """
    Input
        Equilibrium state xx, equilibrium input uu
    Output 
        Cost evaluated at xx,uu
            ll = 0.5 zz.T@Q@zz + 0.5 uu.t@R@uu
        Gradient of the cost
            dl = [Q@zz 
                  R@uu]
        Hessian of the cost 
            ddl = [Q    0
                    0   R ]

    """

    ll = 0.5* (xx - xref).T@Q@(xx - xref) + 0.5*r*(uu**2)

    dl = np.zeros((3,))
    ddl = np.zeros((3,3))

    dl[:2,] = Q@(xx-xref)
    dl[2,] = r*uu
    ddl = np.block([[Q, np.zeros((2,1))],[np.zeros((1,2)),r]])


    return ll, dl, ddl

def equality_constraint_fcn(xx,uu):

    hh = uu - k*(xx[0] - x0)
    dh = np.array([-k,0,1]).reshape(3,)


    return hh, dh

# Algorithm
def run_sqp(Q,r,max_iters):

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

    z_init = np.array([8,0,0])
    zz[:,0] = z_init


    for kk in range(max_iters-1):
        
        xx = zz[:2,kk]
        uu = zz[2,kk]

        # Compute cost and gradient
        ll[kk], dl[:,kk], d2l= cost_fcn(xx,uu,Q,r) 

        # Compute constraint and gradient
        hh[kk], dh[:,kk] = equality_constraint_fcn(xx,uu) 
        
        #BB[:,:,kk] = d2l + 2*lmbd[kk]*np.eye(n_z) # Exact Hessian
        BB[:,:,kk] = d2l                           # Gauss-Newton Approximation

        # check positive definiteness of BB
        print(np.linalg.eigvals(BB[:,:,kk]))
        if np.any(np.linalg.eigvals(BB[:,:,kk]) <= 0):
            print('Hessian not positive definite')
            break


        # SQP solver
        zqp, lmb_qp = SQP_solver(zz[:,kk], dl[:,kk], hh[kk], dh[:,kk], BB[:,:,kk])
        
        # compute the direction and multiplier
        direction = (zqp - zz[:,kk])
        multiplier = lmb_qp


        descent_norm[kk] = np.linalg.norm(direction) #[for plots]
        lagrangian_norm[kk] = np.linalg.norm(dl[:,kk] + dh[:,kk]*multiplier, np.inf) #[for plots]

        
        
        #########################################
        # Update solution and multiplier
        ##########################################
        zz[:,kk+1] = zz[:,kk] + stepsize*direction
        lmbd[kk+1] = (1 - stepsize)*lmbd[kk] + stepsize*multiplier

        # print(np.linalg.norm(direction))
        print(f'LagNorm = {np.linalg.norm(dl[:,kk] + dh[:,kk]*lmbd[kk], np.inf)}')
        print(f'ConstrNorm = {np.linalg.norm(hh[kk])}')

        print(f"Position: {zz[0,kk]}")
        # print(lmbd[kk+1])
        if kk%1e2 == 0:
            print('ll_{} = {}'.format(kk,ll[kk]))


    return zz,kk



Q_values = [np.array([[1., 0], [0, 1.]]), np.array([[10., 0], [0, 1.]])]
r_values = [0.1, 0.01, 0.001]

fig, axes = plt.subplots(len(Q_values), len(r_values), figsize=(15, 10), sharex=True, sharey=True)
fig.suptitle("Effect of Q and r on Mass Position Equilibrium and Optimal Input", fontsize=12)

# Define fixed y-axis limits for the secondary axis 
optimal_input_ylim = (0, 100)  

for i, Q in enumerate(Q_values):
    for j, r in enumerate(r_values):
        results, kk = run_sqp(Q, r, max_iters)
        ax = axes[i, j]

        # Plot Mass Position on primary y-axis
        line1, = ax.plot(results[0, :kk], label="Mass Position", color="royalblue", linewidth=2)
        line2, = ax.plot(np.repeat(xref[0], kk), label="Reference Position", color="green", linestyle="--", linewidth=2)
        
        # Add secondary y-axis for Optimal Input with fixed scale
        ax2 = ax.twinx()
        line3, = ax2.plot(results[2, :kk], label="Optimal Input", color="red", linewidth=2)
        line4, = ax2.plot(np.repeat(u_eq,kk), label="Input at Reference Equilibrium", color="orange", linestyle="--", linewidth=2)
        ax2.set_ylim(optimal_input_ylim)  # Apply fixed scale

        # Titles and grids
        ax.set_title(f"Q = {Q[0,0]:.1f}, {Q[1,1]:.1f}; r = {r:.3f}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Combine legends from both y-axes into one
        lines = [line1, line2, line3, line4]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="lower right")

# Set labels for the shared x-axis
plt.xlabel("Iterations")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#NEWTON METHOD FOR OPTIMAL CONTROL:

from numpy import *
import scipy as sp
import matplotlib.pyplot as plt

# import pendulum dynamics
from NL_dynamics_FRA import *

# import cost functions
from cost import *

from useful_functions import *
from solver_ltv_LQR import *

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

#def NM_robust_OC(xx_ref, uu_ref, xx0, TT, max_iter=10, tol=1e-6): 

#######################################
# Algorithm parameters
#######################################

# max_iters = int(1e1)
max_iters = 5

# ARMIJO PARAMETERS
Armijo = True
stepsize_0 = 0.7
cc = 0.5
beta = 0.7
armijo_maxiters = 20 # number of Armijo iterations

term_cond = 1e-19

visu_descent_plot = False
visu_animation = False

#######################################
# Trajectory parameters
#######################################

tf = 10 # final time in seconds

#discretization step from parameters_3:
TT = int(1/dt) # discrete-time samples

######################################
# Reference curve
######################################

step_reference = True

xx_ref, uu_ref = traj_gen(step_reference=step_reference,tf=tf,dt=dt,ns=ns,ni=ni)

######################################
# Arrays to store data
######################################

xx0 = xx_ref[:,0] # initial state
uu0 = uu_ref[:,0] # initial input

# Forward simulation of the dynamics:
uu_init = uu_ref
xx_init = zeros((ns, TT))
xx_init[:,0] = xx0
for tt in range(TT-1): 
  xx_init[:,tt+1]=discretizedDynamicFRA(xx_init[:,tt], uu_init[:,tt])[0].squeeze()

xx_traj = zeros((ns, TT, max_iters)) # state sequence
uu_traj = zeros((ni, TT, max_iters)) # input sequence

#initializing state sequence
xx_traj[:,:,0] = xx_init[:,:] 
for kk in range(max_iters):
  xx_traj[:,0,kk] = xx0

#initializing input sequence:
uu_traj[:,:,0] = uu_init[:,:]


dJ = zeros((ni,TT, max_iters))   # DJ - gradient of J wrt u
JJ = zeros(max_iters)            # collect cost
descent = zeros(max_iters)       # descent module
descent_arm = zeros(max_iters)   # collect armijo descent direction
lmbd = zeros((ns, TT, max_iters))    # co-states lambdas vector

# Defining matrices dimensions
AAt = zeros((ns, ns, TT))
BBt = zeros((ns, ni, TT))
qqt = zeros((ns, TT))
qqT = zeros((ns))
QQt = zeros((ns, ns, TT))
QQT = zeros((ns,ns))
rrt = zeros((ni, TT))
RRt = zeros((ni, ni, TT))
SSt = zeros((ni, ns, TT))

######################################
# Main
######################################

print('-*-*-*-*-*-')

# Create a figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
kk = 0

for kk in range(max_iters-1):

  JJ[kk] = 0 # Cost initialization

  #######################################################
  #STEP 1.1: COMPUTE AA, BB, QQ, RR, SS, qq, rr & costs: 
  #######################################################

  # Cost computation: 
  for tt in range(TT-1): 
    temp_cost = stagecost(xx_traj[:,tt,kk], uu_traj[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
    JJ[kk] += temp_cost
  
  temp_cost = termcost(xx_traj[:,-1,kk], xx_ref[:,-1])[0]
  JJ[kk] += temp_cost

  # defining terminal values for lambdas, QQt, qqt:
  lmbd_temp = termcost(xx_traj[:,TT-1,kk], xx_ref[:,TT-1])[1]
  lmbd[:,TT-1,kk] = lmbd_temp.copy().squeeze()

  QQT = termcost(xx_traj[:,TT-1,kk], xx_ref[:,TT-1])[2]
  qqT = lmbd_temp.squeeze()
                                                                                      
  # Here starts the computation of all quantities:
  for tt in reversed(range(TT-1)): # integration backward in time
    xx_traj[:,tt+1,kk] = discretizedDynamicFRA(xx_traj[:,tt,kk], uu_traj[:,tt,kk])[0].squeeze()
    dfdx, dfdu, d2fdxdx, d2fdxdu, _, d2fdudu = discretizedDynamicFRA(xx_traj[:,tt,kk], uu_traj[:,tt,kk])[1:]
    AAt[:,:,tt] = dfdx.T
    BBt[:,:,tt] = dfdu

    aa, bb, QQt[:,:,tt], SSt[:,:,tt], RRt[:,:,tt] = stagecost(xx_traj[:,tt,kk], uu_traj[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[1:]

    aa = aa.squeeze()
    bb = bb.squeeze()
    qqt[:,tt] = aa
    rrt[:,tt] = bb
  
    #######################################
    #STEP 1.2: COMPUTE COSTATE EQUATIONS:
    #######################################
    AA = dfdx.T
    BB = dfdu
    for tt in reversed(range(TT-1)):  
      a, b, Q, S, R = stagecost(xx_traj[:,tt,kk], uu_traj[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[1:]
      lmbd_temp = AA.T@lmbd[:,tt+1,kk][:,None] + a # costate equation
      lmbd[:,tt,kk] = lmbd_temp.squeeze()
      # We define QQt,RRt,SSt as their normal version <==> they're positive definite
      #if is_pos_def(QQt[:,:,tt]): QQt[:,:,tt] += d2fdxdx@lmbd[:,tt+1,kk]
      #if is_pos_def(SSt[:,:,tt]): SSt[:,:,tt] += d2fdxdu@lmbd[:,tt+1,kk]
      #if is_pos_def(RRt[:,:,tt]): RRt[:,:,tt] += d2fdudu@lmbd[:,tt+1,kk]

      dJ_temp = BB.T@lmbd[:,tt+1,kk][:,None] + b # gradient of J wrt u
      dJ[:,tt,kk] = dJ_temp.squeeze()

  
  xx_0 = zeros((ns))
  KK, sigma, PP, xx_delta, uu_delta = ltv_LQR(AAt, BBt, QQt, RRt, SSt, QQT, TT, xx_0, qqt, rrt, qqT)

  for tt in reversed(range(TT-1)):  # integration backward in time
    descent[kk] += -dJ[:,tt,kk].T@uu_delta[:,tt]
    descent_arm[kk] += dJ[:,tt,kk].T@uu_delta[:,tt]

  ######################################
  #STEP 1.3: STEPSIZE SELECTION ARMIJO:
  ######################################

  stepsizes = []  # list of stepsizes
  costs_armijo = []

  stepsize = stepsize_0

  ns = xx_ref.shape[0]
  ni = uu_ref.shape[0]


  for ii in range(armijo_maxiters):

    # temp solution update

    xx_temp = np.zeros((ns,TT))
    uu_temp = np.zeros((ni,TT))

    xx_temp[:,0] = xx0


    for tt in range(TT-1):
          uu_temp[:,tt] = uu_traj[:,tt,kk] + stepsize*uu_delta[:,tt]
          #uu_temp[:,tt] = uu_traj[:,tt,kk] + stepsize*sigma[:,tt] + KK[:,:,tt]@xx_delta[:,tt]
          xx_temp[:,tt+1] = dyn.discretizedDynamicFRA(xx_temp[:,tt], uu_temp[:,tt])[0].squeeze()

    # temp cost calculation
    JJ_temp = 0

    for tt in range(TT-1):
          temp_cost = stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
          JJ_temp += temp_cost

    temp_cost = termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
    JJ_temp += temp_cost

    stepsizes.append(stepsize)      # save the stepsize
    #costs_armijo.append(np.min([JJ_temp, 100*JJ]))    # save the cost associated to the stepsize

    if JJ_temp > JJ[kk] + cc*stepsize*descent_arm[kk]:
          # update the stepsize
          stepsize = beta*stepsize

    else:
          print('Armijo stepsize = {:.3e}'.format(stepsize))
          break
    
    if ii == armijo_maxiters -1:
          print("WARNING: no stepsize was found with armijo rule!")
  

  ############################################################
  #STEP 1.4: UPDATE
  ############################################################

  xx_temp = zeros((ns, TT))
  uu_temp = zeros((ni, TT))
  xx_temp[:,0] = xx0

  for tt in range(TT-1): 
    uu_temp[:,tt] = uu_traj[:,tt,kk] + stepsize*uu_delta[:,tt]
    #uu_temp[:,tt] = uu_traj[:,tt,kk] + stepsize*sigma[:,tt] + KK[:,:,tt]@xx_delta[:,tt]
    xx_temp[:,tt+1] = discretizedDynamicFRA(xx_temp[:,tt], uu_temp[:,tt])[0].squeeze()
  
  xx_traj[:,:,kk+1] = xx_temp.copy()
  uu_traj[:,:,kk+1] = uu_temp.copy()

  ############################
  # Termination condition
  ############################

  print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk,descent[kk], JJ[kk]))

  if descent[kk] <= term_cond:

    max_iters = kk

    break

xx_star = xx_traj[:,:,max_iters-1]
uu_star = uu_traj[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2] # for plotting purposes

##################################
# Plot final results
##################################

# Plot cost and descent
plt.figure()
plt.plot(range(max_iters), JJ[:max_iters], label='Cost')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.yscale('log')
plt.grid(True)
plt.title('Cost evolution')
plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(range(max_iters), descent[:max_iters], label='Descent Norm')
plt.xlabel('Iterations')
plt.ylabel('||∇J(u^k)||')
plt.yscale('log')
plt.grid(True)
plt.title('Gradient Norm evolution')
plt.legend()
plt.show(block=False)

# Plot final trajectories
tt_hor = linspace(0, tf, TT)

fig, axs = plt.subplots(ns+ni, 1, sharex=True)

# Stati
for i in range(ns):
    axs[i].plot(tt_hor, xx_star[i,:], linewidth=2, label='x_'+str(i+1))
    axs[i].plot(tt_hor, xx_ref[i,:], 'g--', linewidth=2, label='x_ref_'+str(i+1))
    axs[i].grid(True)
    axs[i].set_ylabel('x_'+str(i+1))
    axs[i].legend()

# Input
for j in range(ni):
    axs[ns+j].plot(tt_hor, uu_star[j,:], 'r', linewidth=2, label='u_'+str(j+1))
    axs[ns+j].plot(tt_hor, uu_ref[j,:], 'r--', linewidth=2, label='u_ref_'+str(j+1))
    axs[ns+j].grid(True)
    axs[ns+j].set_ylabel('u_'+str(j+1))
    axs[ns+j].legend()

axs[-1].set_xlabel('Time (s)')

plt.suptitle('Optimal trajectories vs Reference')
plt.tight_layout()
plt.show()

    


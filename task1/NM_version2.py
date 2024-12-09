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

def NM_robust_OC(xx_ref, uu_ref, xx0, TT, max_iters=10, tol=1e-6): 

  #######################################
  # Algorithm parameters
  #######################################

  # ARMIJO PARAMETERS
  Armijo = True
  stepsize_0 = 0.7
  cc = 0.5
  beta = 0.7
  armijo_maxiters = 5 # number of Armijo iterations

  term_cond = tol

  visu_descent_plot = False
  visu_animation = False

  #######################################
  # Trajectory parameters
  #######################################

  tf = 10 # final time in seconds

  #discretization step from parameters_3:
  TT = int(1/dt) # discrete-time samples

  ######################################
  # Arrays to store data
  ######################################

  xx0 = xx_ref[:,0] # initial state
  uu0 = uu_ref[:,0] # initial input
  xx_traj = zeros((ns, TT, max_iters)) # state sequence
  uu_traj = zeros((ni, TT, max_iters)) # input sequence
  dJ = zeros((ni,TT, max_iters))       # DJ - gradient of J wrt u
  JJ = zeros(max_iters)                # collect cost
  descent = zeros(max_iters)           # descent module
  descent_arm = zeros(max_iters)       # collect armijo descent direction
  lmbd = zeros((ns, TT, max_iters))    # co-states lambdas vector

  # Forward simulation of the dynamics (initialize trajectory via shooting):
  uu_init = uu_ref
  xx_init = zeros((ns, TT))
  xx_init[:,0] = xx0
  for tt in range(TT-1): 
    xx_init[:,tt+1]=discretizedDynamicFRA(xx_init[:,tt], uu_init[:,tt])[0].squeeze()

  #initializing state sequence
  xx_traj[:,:,0] = xx_init[:,:] 
  for kk in range(max_iters):
    xx_traj[:,0,kk] = xx0

  #initializing input sequence:
  uu_traj[:,:,0] = uu_init

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

    #####################################################################
    #STEP 1.1: COMPUTE AA, BB, QQ, RR, SS, qq, rr, costs && COSTATE EQ.: 
    #####################################################################

    # Cost computation: 
    JJ[kk] = tot_cost(xx_traj[:,:,kk], uu_traj[:,:,kk], xx_ref, uu_ref, TT)

    AAt, BBt, qqt, rrt, qqT, dJ[:,:,kk] = solver_costate_eq(xx_traj[:,:,kk], uu_traj[:,:,kk], xx_ref, uu_ref, TT)[:6]
    uu_delta = ltv_LQR(AAt, BBt, QQt_cost, RRt_cost, SSt_cost, QQT_cost, TT, zeros_like(xx0), qqt, rrt, qqT)[4]

    descent[kk] = linalg.norm(uu_delta) # We define the descent direction
    if(linalg.norm(uu_delta)<1e-3): break # Check condition on termination

    for tt in range(TT-1):  # We define the Armijo descent direction
      descent_arm[kk] += dJ[:,tt,kk].T@uu_delta[:,tt]

    ######################################
    #STEP 1.2: STEPSIZE SELECTION ARMIJO:
    ######################################

    stepsize = stepsize_armijo(stepsize_0, armijo_maxiters, cc, beta, uu_delta, xx_ref, uu_ref, xx0, uu_traj[:,:,kk], JJ[kk], TT, descent_arm[kk], plot=False)
    
    ###################
    #STEP 1.4: UPDATE:
    ###################
    xx_temp = zeros((ns, TT))
    uu_temp = zeros((ni, TT))
    xx_temp[:,0] = xx0

    for tt in range(TT-1): 
      uu_temp[:,tt] = uu_traj[:,tt,kk] + stepsize*uu_delta[:,tt]
      xx_temp[:,tt+1] = discretizedDynamicFRA(xx_temp[:,tt], uu_temp[:,tt])[0].squeeze()
    
    
    xx_traj[:,:,kk+1] = xx_temp.copy()
    uu_traj[:,:,kk+1] = uu_temp.copy()

    print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk,descent[kk], JJ[kk]))

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

  # States
  for i in range(ns):
      axs[i].plot(tt_hor, xx_star[i,:], linewidth=2, label='x_'+str(i+1))
      axs[i].plot(tt_hor, xx_ref[i,:], 'g--', linewidth=2, label='x_ref_'+str(i+1))
      axs[i].grid(True)
      axs[i].set_ylabel('x_'+str(i+1))
      #axs[i].legend()

  # Input
  for j in range(ni):
      axs[ns+j].plot(tt_hor, uu_star[j,:], 'r', linewidth=2, label='u_'+str(j+1))
      axs[ns+j].plot(tt_hor, uu_ref[j,:], 'r--', linewidth=2, label='u_ref_'+str(j+1))
      axs[ns+j].grid(True)
      axs[ns+j].set_ylabel('u_'+str(j+1))
      #axs[ns+j].legend()

  axs[-1].set_xlabel('Time (s)')

  plt.suptitle('Optimal trajectories vs Reference')
  plt.tight_layout()
  plt.show()

  return xx_traj[:,:,kk], uu_traj[:,:,kk]

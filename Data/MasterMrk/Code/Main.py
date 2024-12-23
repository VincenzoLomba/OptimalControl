# Group 15
# Eugenio Piccini, 
# Mirko Legnini,
# Francesco Davide Bossio
import Dynamics as dyn
import Cost as cst
import utils
import plots
import numpy as np
import scipy.optimize as op
#############################à
# Task parameters
#############################à
task_set=[0, 0, 1, 0, 1] # Set to 1 for tasks to be performed
show_desired_trajectories=True
visu_armijo=False
save_trajectories=False
use_multiple_initial_conditions=False

#######################################
# Trajectory parameters
#######################################
tf = 20 # final time in seconds
max_iters = 20
dt = dyn.dt   # get discretization step from dynamics
xDim = dyn.xDim
uDim = dyn.uDim
TT = int(tf/dt) # discrete-time samples

# Select desired state behavior V and beta
fixed_states_1 = np.array([8, 0]) # straight trajectory
fixed_states_2 = np.array([12, np.pi/6]) # circular trajectory

# Initial condition for equilibrium evaluation delta, Fx and psi dot
initEq = np.array([np.pi/6, 100, np.pi/12])

#################################à
# Task 1
#################################à
if task_set[0]: 
  ####################################à
  # Find equilibria
  ####################################à
  uuDes = np.zeros((uDim,TT))
  # Calc equilibrium pairs delta, Fx and psi dot
  eq1,dict,ier1,mesg1 = op.fsolve(dyn.dynamics_equilibrium, x0=initEq, args=(fixed_states_1), full_output=True)
  eq2,dict,ier2,mesg2 = op.fsolve(dyn.dynamics_equilibrium, x0=initEq,  args=(fixed_states_2), full_output=True)

  # Check eq values
  print (ier1, mesg1)
  print (ier2, mesg2)

  print ("Equilibrium 1 = ", eq1)
  print("Equilibrium 2 = ", eq2)
  check = dyn.dynamics_equilibrium(eq2, fixed_states_2)		
  print ("Derivative check for equilibrium 2 = ", check )

  # Init first equilibrium
  xxEq1=np.zeros(6)
  xxEq1[0]=10
  xxEq1[1]=15
  xxEq1[2]=np.pi/6
  xxEq1[3:5]=fixed_states_1
  xxEq1[5]=eq1[2]
  uuEq1=eq1[0:2]

  ## Test dynamics 1
  xx1 = np.zeros((xDim, int(TT/2)))
  uEquilibrium1=np.zeros((uDim, int(TT/2)))
  for i in range (int(TT/2)):
      uEquilibrium1[:,i]=uuEq1
  xx1 = utils.run_system(xxEq1, uEquilibrium1, int(TT/2))

  # Init second equilibrium
  xxEq2=np.zeros(6)
  xxEq2[0:3]=xx1[0:3,-1]
  xxEq2[3:5]=fixed_states_2
  xxEq2[5]=eq2[2]
  uuEq2=eq2[0:2]

  ## Test dynamics 2
  xx2 = np.zeros((int(xDim), int(TT/2)))
  uEquilibrium2=np.zeros((uDim, int(TT/2)))
  for i in range (int(TT/2)):
      uEquilibrium2[:,i]=uuEq2
  xx2 = utils.run_system(xxEq2, uEquilibrium2, int(TT/2))

  ## Create desired trajectory
  xxDes = np.concatenate((xx1, xx2), axis=-1)
  uuDes=np.concatenate((uEquilibrium1, uEquilibrium2), axis=-1)

  if show_desired_trajectories:  ## Plot desired trajectory
    plots.plotters(xxDes, None, TT, "Task 1 desired trajectory")

  xx_opt, uu_opt=utils.newtons_method(TT, max_iters, xxEq1, xxDes, uuDes, visu_armijo, "Task 1")

#################################à
# Task 2 
#################################à

if task_set[1]: ## if task is required
  #Initialize trajectory points
  tt = np.linspace(0, TT, TT)
  sigmoid=np.zeros((2,TT)) 
  uuDes_smooth=np.zeros((uDim, TT))

  #Calculate desired trajectory points as a sigmoid function
  sigmoid[0,:] = fixed_states_1[0] + (fixed_states_2[0]-fixed_states_1[0])/(1 + np.exp((-tt+(TT/2))/(4*tf)))
  sigmoid[1,:] = fixed_states_1[1] + (fixed_states_2[1]-fixed_states_1[1])/(1 + np.exp((-tt+(TT/2))/(4*tf))) 

  ## Initialize equilibrium values and find quasi static cornering equilibrium points
  eq=np.zeros((3,TT))
  for t in range(TT):
    eq[:,t], dict, ier, msg= op.fsolve(dyn.dynamics_equilibrium, x0=initEq, args=(sigmoid[:,t]), full_output=True)
    if(ier!=1): print(msg)
  
  ##Define initial conditions and integrate positions from the quasi static velocities trajectory
  xxDes_smooth=np.zeros((xDim, TT))
  xxDes_smooth[0,0]=10
  xxDes_smooth[1,0]=15
  xxDes_smooth[2,0]=np.pi/6
  xxDes_smooth[3:5, :]=sigmoid
  xxDes_smooth[5,:]=eq[2,:]
  for t in range(TT-1):
     xxDes_smooth[0:3, t+1]=dyn.integrate_position(xxDes_smooth[:,t])
     uuDes_smooth[:,t]=eq[0:2,t]

  # ## Plot desired trajectory
  if show_desired_trajectories:
    plots.plotters(xxDes_smooth, None, TT, "Task 2 desired trajectory")

  # Set initial conditions and solve via newton's method
  x0=xxDes_smooth[:,0]
  xx_opt_smooth, uu_opt_smooth=utils.newtons_method(TT, max_iters, x0, xxDes_smooth, uuDes_smooth, visu_armijo, "Task 2")
  if save_trajectories:
    np.save("./saved_arrays/x_opt_{}".format(max_iters), xx_opt_smooth)
    np.save("./saved_arrays/u_opt_{}".format(max_iters), uu_opt_smooth)
    
#################################à
# Task 3 
#################################à

if (task_set[2]): #if task is required
  #Initialize arrays for local linearization
  A_opt=np.zeros((xDim, xDim, TT))
  B_opt=np.zeros((xDim, uDim, TT))  

  # load optimal trajectory
  if(task_set[1]): #if computed on this run
    xx_opt=xx_opt_smooth
    uu_opt=uu_opt_smooth

  else: # else load from file
    xx_opt=np.load("./saved_arrays/x_opt_20.npy")
    uu_opt=np.load("./saved_arrays/u_opt_20.npy")
    TT = np.shape(xx_opt)[1]
  # Set costs
  QQt_reg=cst.QQt
  RRt_reg=cst.RRt

  # Compute local linearization
  for t in range(TT):
    ft, fu=dyn.dynamics(xx_opt[:,t], uu_opt[:,t])[1:]
    A_opt[:,:,t]=ft.T
    B_opt[:,:,t]=fu.T

  # Compute LQR gains
  KK=utils.ltv_LQR(A_opt, B_opt, QQt_reg, RRt_reg, QQt_reg, TT)[0]
  # Create initial conditions with noise
  noise_percentage=0.20
  noise=utils.make_some_noise(xx_opt, noise_percentage)
  xx_real, uu_feedback=utils.run_LQR_with_noise(xx_opt, uu_opt, KK, noise, TT, use_multiple_initial_conditions)
  # Save trajectories if required 
  if save_trajectories:
    np.save("./saved_arrays/x_real_task3_{}".format(max_iters), xx_real)
    np.save("./saved_arrays/u_real_task3_{}".format(max_iters), uu_feedback)

#################################à
# Task 4 
#################################à

if task_set[3]: # if task is required
  T_mpc = 10 # set time window


  ## Set costs
  QQt_reg=cst.QQt
  RRt_reg=cst.RRt

  QQT_reg = cst.QQT

  ## Load optimal trajectory 
  if(task_set[1]): #if computed during this run
    xx_opt_mpc=xx_opt_smooth
    uu_opt_mpc=uu_opt_smooth

  elif task_set[2]: #else if already loaded from file
    xx_opt_mpc = xx_opt
    uu_opt_mpc = uu_opt
    
  else: #else load from file
    xx_opt_mpc=np.load("./saved_arrays/x_opt_20.npy")
    uu_opt_mpc=np.load("./saved_arrays/u_opt_20.npy")
    TT = np.shape(xx_opt_mpc)[1]

  # Initialize arrays for local linearization  
  AA = np.zeros((xDim,xDim,TT))
  BB = np.zeros((xDim,uDim,TT))
    
  # Local linearization for the dynamics
  for t in range(TT):
    ft, fu=dyn.dynamics(xx_opt_mpc[:,t], uu_opt_mpc[:,t])[1:]

    AA[:,:,t]=ft.T
    BB[:,:,t]=fu.T
    
  # Initialize array for mpc input and state trajectory
  uu_real_mpc = np.zeros((uDim, TT))
  xx_real_mpc = np.zeros((xDim, TT))
  
  #Create initial condition with noise
  noise_percentage=0.10
  noise=utils.make_some_noise(xx_opt_mpc, noise_percentage)
  xx_real_mpc, uu_real_mpc=utils.run_MPC_with_noise(xx_opt_mpc, uu_opt_mpc, AA, BB, QQt_reg, RRt_reg, QQT_reg, T_mpc, noise, TT, use_multiple_initial_conditions)

#################################à
# Task 5
#################################à
if task_set[4]:
  xx_opt=np.load("./saved_arrays/x_opt_20.npy")
  uu_opt=np.load("./saved_arrays/u_opt_20.npy")

  xx_real=np.load("./saved_arrays/x_real_task3_20.npy")
  uu_real=np.load("./saved_arrays/u_real_task3_20.npy")
  
  plots.make_animation(xx_real,xx_opt, uu_real, TT)
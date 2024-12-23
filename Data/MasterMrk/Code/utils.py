# Group 15
# Eugenio Piccini, 
# Mirko Legnini,
# Francesco Davide Bossio
import Dynamics as dyn
import Cost as cst
import matplotlib.animation as animation
from matplotlib import transforms
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx
import matplotlib.image as mgimg
import plots

xDim=dyn.xDim
uDim=dyn.uDim

def run_system (x0, uu, TT): ## run system dynamics for TT time instants starting from x0
    xx=np.zeros((len(x0), TT))
    xx[:,0]=x0
    for t in range(TT-1):
        xx[:,t + 1] = dyn.dynamics(xx[:,t], uu[:, t])[0]
    return xx

def calc_total_cost(xx, uu, xDes, uDes, TT): ## get total cost for an iteration
    JJ=0
    for t in range(TT-1):
        temp_cost = cst.stagecost(xx[:,t], uu[:,t], xDes[:,t], uDes[:,t])[0]
        JJ+= temp_cost  
    temp_cost = cst.termcost(xx[:,-1], xDes[:,-1])[0]
    JJ+= temp_cost
    return JJ

def armijo(visu_armijo, xDim, uDim, uu,deltau,descent_arm, uu_ref, xx_ref, x0, TT, JJ): #get armijo step size
  stepsizes = []  # list of stepsizes
  costs_armijo = []
 
  stepsize = 1
  ni=uDim
  ns=xDim
  cc=0.5
  for ii in range(20):

    # temp solution update

    xx_temp = np.zeros((ns,TT))
    uu_temp = np.zeros((ni,TT))

    xx_temp[:,0] = x0

    for tt in range(TT-1):
      uu_temp[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
      xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

    # temp cost calculation
    JJ_temp = 0

    for tt in range(TT-1):
      temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
      JJ_temp += temp_cost

    temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
    JJ_temp += temp_cost

    stepsizes.append(stepsize)      # save the stepsize
    costs_armijo.append(np.min([JJ_temp, 10*JJ]))    # save the cost associated to the stepsize

    if JJ_temp > JJ  + cc*stepsize*descent_arm:
        # update the stepsize
        stepsize = 0.7*stepsize
    
    else:
        print('Armijo stepsize = {:.3e}'.format(stepsize))
        break

  # plt.plot(xx_temp[0,:])
  # plt.show()
  ############################
  # Armijo plot
  ############################

  if visu_armijo:

    steps = np.linspace(0,1,int(2e1))
    costs = np.zeros(len(steps))

    for ii in range(len(steps)):

      step = steps[ii]

      # temp solution update

      xx_temp = np.zeros((ns,TT))
      uu_temp = np.zeros((ni,TT))

      xx_temp[:,0] = x0

      for tt in range(TT-1):
        uu_temp[:,tt] = uu[:,tt] + step*deltau[:,tt]
        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

      # temp cost calculation
      JJ_temp = 0

      for tt in range(TT-1):
        temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
        JJ_temp += temp_cost

      temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
      JJ_temp += temp_cost

      costs[ii] = np.min([JJ_temp, 10*JJ])


    plt.figure(1)
    plt.clf()

    plt.plot(steps, costs, color='k', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
    plt.plot(steps, JJ + descent_arm*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    # plt.plot(steps, JJ[kk] - descent[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    plt.plot(steps, JJ + cc*descent_arm*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

    plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

    plt.grid()
    plt.xlabel('stepsize')
    plt.legend()
    plt.draw()

    plt.show()


  return stepsize
  
  TT, np.zeros_like(x0), qq, rr, qT

def ltv_aLQR(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qqin = None, rrin = None, qqfin = None): # affine

  """
	LQR for LTV system with (time-varying) affine cost
	
  Args
    - AAin (nn x nn (x TT)) matrix
    - BBin (nn x mm (x TT)) matrix
    - QQin (nn x nn (x TT)), RR (mm x mm (x TT)), SS (mm x nn (x TT)) stage cost
    - QQfin (nn x nn) terminal cost
    - qq (nn x (x TT)) affine terms
    - rr (mm x (x TT)) affine terms
    - qqf (nn x (x TT)) affine terms - final cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """
	
  try:
    # check if matrix is (.. x .. x TT) - 3 dimensional array 
    ns, lA = AAin.shape[1:]
  except:
    # if not 3 dimensional array, make it (.. x .. x 1)
    AAin = AAin[:,:,None]
    ns, lA = AAin.shape[1:]

  try:  
    ni, lB = BBin.shape[1:]
  except:
    BBin = BBin[:,:,None]
    ni, lB = BBin.shape[1:]

  try:
      nQ, lQ = QQin.shape[1:]
  except:
      QQin = QQin[:,:,None]
      nQ, lQ = QQin.shape[1:]

  try:
      nR, lR = RRin.shape[1:]
  except:
      RRin = RRin[:,:,None]
      nR, lR = RRin.shape[1:]

  try:
      nSi, nSs, lS = SSin.shape
  except:
      SSin = SSin[:,:,None]
      nSi, nSs, lS = SSin.shape

  # Check dimensions consistency -- safety
  if nQ != ns:
    print("Matrix Q does not match number of states")
    exit()
  if nR != ni:
    print("Matrix R does not match number of inputs")
    exit()
  if nSs != ns:
    print("Matrix S does not match number of states")
    exit()
  if nSi != ni:
    print("Matrix S does not match number of inputs")
    exit()


  if lA < TT:
    AAin = AAin.repeat(TT, axis=2)
  if lB < TT:
    BBin = BBin.repeat(TT, axis=2)
  if lQ < TT:
    QQin = QQin.repeat(TT, axis=2)
  if lR < TT:
    RRin = RRin.repeat(TT, axis=2)
  if lS < TT:
    SSin = SSin.repeat(TT, axis=2)

  # Check for affine terms

  augmented = False

  if qqin is not None or rrin is not None or qqfin is not None:
    augmented = True
    print("Augmented term!")

  KK = np.zeros((ni, ns, TT))
  sigma = np.zeros((ni, TT))
  PP = np.zeros((ns, ns, TT))
  pp = np.zeros((ns, TT))

  QQ = QQin
  RR = RRin
  SS = SSin
  QQf = QQfin
  
  qq = qqin
  rr = rrin

  qqf = qqfin

  AA = AAin
  BB = BBin

  xx = np.zeros((ns, TT))
  uu = np.zeros((ni, TT))

  xx[:,0] = x0
  
  PP[:,:,-1] = QQf
  pp[:,-1] = qqf
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    qqt = qq[:,tt][:,None]
    RRt = RR[:,:,tt]
    rrt = rr[:,tt][:,None]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]
    PPtp = PP[:,:,tt+1]
    pptp = pp[:, tt+1][:,None]

    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
    mmt = rrt + BBt.T @ pptp
    
    PPt = AAt.T @ PPtp @ AAt - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ (BBt.T@PPtp@AAt + SSt) + QQt
    ppt = AAt.T @ pptp - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ mmt + qqt

    PP[:,:,tt] = PPt
    pp[:,tt] = ppt.squeeze()


  # Evaluate KK
  
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    qqt = qq[:,tt][:,None]
    RRt = RR[:,:,tt]
    rrt = rr[:,tt][:,None]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]

    PPtp = PP[:,:,tt+1]
    pptp = pp[:,tt+1][:,None]

    # Check positive definiteness

    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
    mmt = rrt + BBt.T @ pptp

    # for other purposes we could add a regularization step here...

    KK[:,:,tt] = -MMt_inv@(BBt.T@PPtp@AAt + SSt)
    sigma_t = -MMt_inv@mmt

    sigma[:,tt] = sigma_t.squeeze()


  

  for tt in range(TT - 1):
    # Trajectory

    uu[:, tt] = KK[:,:,tt]@xx[:, tt] + sigma[:,tt]
    xx_p = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt]

    xx[:,tt+1] = xx_p

    xxout = xx
    uuout = uu

  return KK, sigma, PP, xxout, uuout

def ltv_LQR(AA, BB, QQ, RR, QQf, TT): #linear

  """
	LQR for LTV system with (time-varying) cost	
	
  Args
    - AA (nn x nn (x TT)) matrix
    - BB (nn x mm (x TT)) matrix
    - QQ (nn x nn (x TT)), RR (mm x mm (x TT)) stage cost
    - QQf (nn x nn) terminal cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """
	
  try:
    # check if matrix is (.. x .. x TT) - 3 dimensional array 
    ns, lA = AA.shape[1:]
  except:
    # if not 3 dimensional array, make it (.. x .. x 1)
    AA = AA[:,:,None]
    ns, lA = AA.shape[1:]

  try:  
    ni, lB = BB.shape[1:]
  except:
    BB = BB[:,:,None]
    ni, lB = BB.shape[1:]

  try:
      nQ, lQ = QQ.shape[1:]
  except:
      QQ = QQ[:,:,None]
      nQ, lQ = QQ.shape[1:]

  try:
      nR, lR = RR.shape[1:]
  except:
      RR = RR[:,:,None]
      nR, lR = RR.shape[1:]

  # Check dimensions consistency -- safety
  if nQ != ns:
    print("Matrix Q does not match number of states")
    exit()
  if nR != ni:
    print("Matrix R does not match number of inputs")
    exit()


  if lA < TT:
      AA = AA.repeat(TT, axis=2)
  if lB < TT:
      BB = BB.repeat(TT, axis=2)
  if lQ < TT:
      QQ = QQ.repeat(TT, axis=2)
  if lR < TT:
      RR = RR.repeat(TT, axis=2)
  
  PP = np.zeros((ns,ns,TT))
  KK = np.zeros((ni,ns,TT))
  
  PP[:,:,-1] = QQf
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    PPtp = PP[:,:,tt+1]
    
    PP[:,:,tt] = QQt + AAt.T@PPtp@AAt - \
        + (AAt.T@PPtp@BBt)@np.linalg.inv((RRt + BBt.T@PPtp@BBt))@(BBt.T@PPtp@AAt)
  
  # Evaluate KK
  
  
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    PPtp = PP[:,:,tt+1]
    
    KK[:,:,tt] = -np.linalg.inv(RRt + BBt.T@PPtp@BBt)@(BBt.T@PPtp@AAt)

  return KK, PP
    
def solve_costate_equation(xx, xxDes, uu, uuDes, TT): #solve the costate equations given a full evolution and a desired curve
  ## Returns linearized dynamics, cost residuals, the gradient of J wrt u and the lagrange multipliers lambda
  lmbd=np.zeros_like(xx)
  AA=np.zeros((xDim,xDim, TT))
  BB=np.zeros((xDim,uDim, TT))
  dJ=np.zeros_like(uu)
  qq=np.zeros_like(xxDes)
  rr=np.zeros_like(uuDes)

  qT=cst.termcost(xx[:,TT-1], xxDes[:,TT-1])[1].squeeze()
  lmbd[:,TT-1]=qT 

  for t in reversed(range(TT-1)):  # integration backward in time 
    at, bt = cst.stagecost(xx[:,t], uu[:,t], xxDes[:,t], uuDes[:,t])[1:]
    
    fx, fu = dyn.dynamics(xx[:,t], uu[:,t])[1:]
    At = fx.T
    Bt = fu.T
 
    lmbd_temp = At.T@lmbd[:,t+1][:,None] + at       # costate equation
    dJ_temp = Bt.T@lmbd[:,t+1][:,None] + bt         # gradient of J wrt u
 
    lmbd[:,t] = lmbd_temp.squeeze()
    dJ[:,t] = dJ_temp.squeeze()

    qq[:,t]=at.squeeze()
    rr[:,t]=bt.squeeze()
    AA[:,:,t]=At
    BB[:,:,t]=Bt

  return AA, BB, qq, rr, qT, dJ, lmbd

def linear_mpc(AA, BB, QQ, RR, QQf, xxt, xx_des, uu_des, T_pred = 5): ##solves linear mpc problem
  """
      Linear MPC solver - Constrained LQR

      Given a measured state xxt measured at t
      gives back the optimal input to be applied at t

      Args
        - AA, BB: linear dynamics
        - QQ,RR,QQf: cost matrices
        - xxt: initial condition (at time t)
        - T: time (prediction) horizon

      Returns
        - u_t: input to be applied at t
        - xx, uu predicted trajectory

  """

  xxt = xxt.squeeze()

  xDim, uDim = BB[:,:,0].shape

  xx_mpc = cvx.Variable((xDim, T_pred))
  uu_mpc = cvx.Variable((uDim, T_pred))

  cost = 0
  constr = []

  for tt in range(T_pred-1):
      cost += cvx.quad_form((xx_mpc[:,tt]-xx_des[:,tt]), QQ) + cvx.quad_form((uu_mpc[:,tt]-uu_des[:,tt]), RR)
      constr += [xx_mpc[:,tt+1] == AA[:,:,tt]@xx_mpc[:,tt] + BB[:,:,tt]@uu_mpc[:,tt] # dynamics constraint # other constraints
                ]
  # sums problem objectives and concatenates constraints.
  cost += cvx.quad_form((xx_mpc[:,T_pred-1]-xx_des[:,T_pred-1]), QQf)
  constr += [xx_mpc[:,0] == xxt]

  problem = cvx.Problem(cvx.Minimize(cost), constr)
  problem.solve()

  if problem.status == "infeasible":
  # Otherwise, problem.value is inf or -inf, respectively.
      print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

  return uu_mpc[:,0].value, xx_mpc.value, uu_mpc.value

def newtons_method(TT, max_iters, x0, xxDes, uuDes, visu_armijo, taskname="Task"): ##solves an optimization problem using newton's method

  xx = np.zeros((xDim, TT, max_iters))
  uu = np.zeros((uDim,TT, max_iters))
  JJ = np.zeros(max_iters)
  KK=np.zeros((uDim, xDim, TT, max_iters))
  AA=np.zeros((xDim,xDim, TT))
  BB=np.zeros((xDim,uDim, TT))
  dJ=np.zeros((uDim, TT, max_iters))
  descent=np.zeros((max_iters))
  descent_arm=np.zeros((max_iters))
  SSt=np.zeros((uDim, xDim))
  qq=np.zeros_like(xxDes)
  rr=np.zeros_like(uuDes)
  deltau=np.zeros_like(uu)

  ###########################
  ## Task 1 Newton's method
  ###########################
  uu_init = uuDes

  ## Initialize first feasible trajectory via shooting
  uu[:,:,0] = uu_init
  xx[:,:,0]=run_system(x0, uu_init, TT)

  for k in range(max_iters-1):   
    JJ[k]=calc_total_cost(xx[:,:,k], uu[:,:,k], xxDes, uuDes, TT)

    #################################
    # Descent direction calculation
    #################################

    AA, BB, qq, rr, qT, dJ[:,:,k]=solve_costate_equation(xx[:,:,k], xxDes, uu[:,:,k], uuDes, TT)[:6]
    deltau=ltv_aLQR(AA, BB, cst.QQt, cst.RRt, SSt, cst.QQT, TT, np.zeros_like(x0), qq, rr, qT)[4]


    descent[k]=np.linalg.norm(deltau)
    if(np.linalg.norm(deltau)<1e-3): break

    descent_arm[k]=0
    for t in range(TT):
      descent_arm[k]+=deltau[:,t].T @ dJ[:,t, k]
    ## Armijo
    #if k%2==0: visu_armijo=True
    #else: visu_armijo=False
    step_size= armijo(visu_armijo, xDim, uDim, uu[:,:,k], deltau, descent_arm[k], uuDes, xxDes, x0, TT, JJ[k])
    #step_size= utils.armijo(visu_armijo, xDim, uDim, uu[:,:,k], deltau[:,:,k], descent_arm[k], uuDes, xxDes, x0, TT)
    xx_temp = np.zeros((xDim,TT))
    uu_temp = np.zeros((uDim,TT))

    xx_temp[:,0] = x0

    for tt in range(TT-1):
      uu_temp[:,tt] = uu[:,tt,k] + step_size*deltau[:,tt]
      xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

    xx[:,:,k+1] = xx_temp
    uu[:,:,k+1] = uu_temp

    print('\nIter = {}\n Descent = {:.3e}\t Cost = {:.3e}\n'.format(k,descent[k], JJ[k]))

  plots.globeplotters(xx[:,:,0], xx[:,:,1], xx[:,:,int(k/2)], xx[:,:,k], xxDes, TT, taskname+" state different iterations")
  plots.globeuplotters(uu[:,:,0], uu[:,:,1], uu[:,:,int(k/2)], uu[:,:,k], uuDes, TT, taskname+ "input for different iterations")

  plots.plotters(xx[:,:,k], xxDes, TT, taskname+" optimal state trajectory")
  plots.uplotters(uu[:,:,k],uuDes,TT,taskname+" optimal input trajectory")
  plt.semilogy(range(k), JJ[:k], label='cost over time')
  plt.grid()
  plt.ylabel("Cost (logarithmic)")
  plt.xlabel("Iterations")
  plt.title("Cost over iteraytions")
  plt.show()  
  plt.semilogy(range(k), descent[:k], label='Norm of gradient')
  plt.ylabel("Gradient norm (logarithmic)")
  plt.xlabel("Iterations")
  plt.grid()
  plt.title("Norm of gradient")
  plt.show() 

  return xx[:,:,k], uu[:,:,k]

def make_some_noise(xx, noise_percentage): ## makes noise proportional to the maximum values for each state 
  noise_weights=np.max(np.abs(xx), axis=1)*noise_percentage
  noise= (np.multiply(noise_weights.T, (np.random.rand(xDim)-1/2)))
  return noise 

def run_MPC_with_noise(xx_opt_mpc, uu_opt_mpc, AA, BB, QQt_reg, RRt_reg, QQT_reg, T_mpc, noise, TT, use_multiple_initial_conditions):
  states= ["x", "y", "$\\psi$", "V", "$\\beta$", "$\\dot{\\psi}$"]
  if use_multiple_initial_conditions:
    for i in range(xDim): 
      print("Running MPC with error on {}".format(states[i]))
      #Initialize arrays for real input and state trajectory
      xx_real_mpc=np.zeros((xDim,TT))
      uu_real_mpc=np.zeros((uDim, TT))
      xx_real_mpc[:,0]=xx_opt_mpc[:,0]
      xx_real_mpc[i,0]+=noise[i] #noise only on state initial state i

      #Run mpc controlled system
      for t in range(TT-T_mpc-1):
        uu_real_mpc[:,t] = linear_mpc(AA[:,:,t:t+T_mpc], BB[:,:,t:t+T_mpc], QQt_reg, RRt_reg, QQT_reg, xx_real_mpc[:,t], xx_opt_mpc[:,t:t+T_mpc], uu_opt_mpc[:,t:t+T_mpc], T_mpc)[0]
        xx_real_mpc[:,t+1] = dyn.dynamics(xx_real_mpc[:,t],uu_real_mpc[:,t])[0]   
        if t%10==0:
          print('MPC iter: ', t)
      # Plot results
      plots.plotters(xx_real_mpc[:,:-T_mpc], xx_opt_mpc[:,:-T_mpc], TT-T_mpc, 'MPC state trajectory, noise on {}'.format(states[i]))
      plots.uplotters(uu_real_mpc[:,:-T_mpc-1], uu_opt_mpc[:,:-T_mpc-1], TT-T_mpc-1, 'MPC input trajectory, noise on {}'.format(states[i]))

  xx_real_mpc=np.zeros((xDim,TT))
  uu_real_mpc=np.zeros((uDim, TT))    
  xx_real_mpc[:,0]=xx_opt_mpc[:,0] + noise
  #Run mpc controlled system
  for t in range(TT-T_mpc-1):
    uu_real_mpc[:,t] = linear_mpc(AA[:,:,t:t+T_mpc], BB[:,:,t:t+T_mpc], QQt_reg, RRt_reg, QQT_reg, xx_real_mpc[:,t], xx_opt_mpc[:,t:t+T_mpc], uu_opt_mpc[:,t:t+T_mpc], T_mpc)[0]
    xx_real_mpc[:,t+1] = dyn.dynamics(xx_real_mpc[:,t],uu_real_mpc[:,t])[0]   
    if t%10==0:
      print('MPC iter: ', t)
  # Plot results
  plots.plotters(xx_real_mpc[:,:-T_mpc], xx_opt_mpc[:,:-T_mpc], TT-T_mpc, 'MPC state trajectory')
  plots.uplotters(uu_real_mpc[:,:-T_mpc-1], uu_opt_mpc[:,:-T_mpc-1], TT-T_mpc-1, 'MPC input trajectory')
  return xx_real_mpc, uu_real_mpc
  
def run_LQR_with_noise(xx_opt, uu_opt, KK, noise, TT, use_multiple_initial_conditions):
  states= ["x", "y", "$\\psi$", "V", "$\\beta$", "$\\dot{\\psi}$"]
  if use_multiple_initial_conditions:
    for i in range(xDim): 
      #Initialize arrays for real input and state trajectory
      xx_real=np.zeros((xDim,TT))
      uu_feedback=np.zeros((uDim, TT))
      xx_real[:,0]=xx_opt[:,0]
      xx_real[i,0]+=noise[i] #noise only on state initial state i
      # Run system with real dynamics and feedback input
      for t in range(TT-1):
        uu_feedback[:,t]=uu_opt[:,t] + KK[:,:,t] @ (xx_real[:,t] - xx_opt[:,t])
        xx_real[:,t+1]=dyn.dynamics(xx_real[:,t], uu_feedback[:,t])[0]
      # Plot the results  
      plots.plotters(xx_real, xx_opt, TT, "LQR state trajectory, error on {}".format(states[i]))
      plots.uplotters(uu_feedback[:,:TT-1], uu_opt[:,:TT-1], TT-1, "LQR input trajectory, error on {}".format(states[i]))
  xx_real=np.zeros((xDim,TT))
  uu_feedback=np.zeros((uDim, TT))
  xx_real[:,0]=xx_opt[:,0] + noise
  # Run system with real dynamics and feedback input
  for t in range(TT-1):
    uu_feedback[:,t]=uu_opt[:,t] + KK[:,:,t] @ (xx_real[:,t] - xx_opt[:,t])
    xx_real[:,t+1]=dyn.dynamics(xx_real[:,t], uu_feedback[:,t])[0]
  # Plot the results  
  plots.plotters(xx_real, xx_opt, TT, "LQR state trajectory")
  plots.uplotters(uu_feedback[:,:TT-1], uu_opt[:,:TT-1], TT-1, "LQR input trajectory")
  return xx_real, uu_feedback
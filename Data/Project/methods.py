
from numpy import *
from builtins import all
from control import dare
from dynamics import runDynamicFunction
from costs import totalCostFunction, stageCostTrkTrj, termCostTrkTrj
from matplotlib import pyplot as plt

def runNewtonMethodTrkTrj(xx_des, uu_des, xx0, TT, maxIterations, discretizedDynamicFuntion, tolerance, QQ, RR, givenQQT=None):
    """
    Newton's Like Method in closed loop version for an Optimal Control Trajectory Tracking Problem
    Arguments:
    - xx_des: column vector state desired curve (dimension: ns*TT) (supposed to be TT >> ns) (ns is the number of states of the system)
      (usage: fot t=0 we have the initial state xx0,
              then we have the state curve for t from 1 up to T-2,
              then value for t=T-1 is considered as the state terminal value)
    - uu_des: column vector input desired curve (dimension: ni*TT) (supposed to be TT >> ni) (ni is the number of inputs of the system)
      (usage: for t from 0 to T-2, we have the input curve;
              for t=T-1 we have a non actuated input that should anyway be not totally neglected in the optimization algorithm)
    - xx0: column vector fixed initial state (from a python variable p.o.f., this is of shape (ns,))
    - TT: number of time steps (each one of duration dt, enough for evolve from t=0 to t=T, where [0, T] is the considered horizon)
    - maxIterations: maximum number of iterations for the method to converge
    - discretizedDynamicFuntion: functions of (xx_t, uu_t) that implements the discretized dynamics of the system that is being considered,
                                 requiring as arguments respectively the state and input values at time t,
                                 returning the state value at time t+1 AND all the jacobians and hessians of the dynamics wrt state and input in the following order:
                                 xxp, dfdx, dfdu, d2fdxdx, d2fdxdu, d2fdudx, d2fdudu
    - tolerance: minimum value that the norm of the descent direction has to reach to consider the optimization as converged and completed
    Returns:
    - xx: column vector state (feasible) trajectory obtained through the optimization (coupled with uu; returning a state-input trajectory)
    - uu: column vector input (feasible) trajectory obtained through the optimization (coupled with xx; returning a state-input trajectory)
    """

    # Compute ni and ns and be sure that the desired curves are in the right shape
    ni = min(uu_des.shape)
    ns = min(xx_des.shape)
    if (uu_des.ndim == 1): ni = 1; uu_des = uu_des.reshape(ni, uu_des.shape[0])
    if (xx_des.ndim == 1): ns = 1; xx_des = xx_des.reshape(ns, xx_des.shape[0])

    # Definition the stage cost function at the generic istant of time t (time invariant cost matrixes QQ and RR are considered here)
    def stageCostFunction(xx_t, uu_t, xx_des_t, uu_des_t, t):
        return stageCostTrkTrj(xx_t, uu_t, xx_des_t, uu_des_t, QQ, RR)

    # Initialization of the collections
    xxCollection = zeros((xx_des.shape[0], TT, maxIterations))
    uuCollection = zeros((uu_des.shape[0], TT, maxIterations))
    stepSizeCollection = zeros((maxIterations))

    # Initialization of the N.M. with a feasible trajectory via shooting (running forward and in an open loop fashion the dynamic of the system)
    uuCollection[:,:,0] = uu_des
    xxCollection[:,:,0] = runDynamicFunction(discretizedDynamicFuntion, uu_des, xx0, TT)

    # Execution of the N.M. single step iteration (where k is the iteration index)
    k = 0
    while k < maxIterations:

        print("\n[E] N.M. now approaching iteration ", format(k+1))

        print("Solving the costate equation (and computing all involved quantities)...")
        _, AA, BB, qq, rr, qqT, QQtilde, SStilde, RRtilde, QQT, grdJdu, ll = solveCostateEquationTrkTrj(
            xxCollection[:,:,k], uuCollection[:,:,k], xx_des, uu_des, discretizedDynamicFuntion, stageCostFunction, TT, givenQQT
        )

        print("Actual cost: ", ll)

        # Definition the terminal cost function at time T (for this particular N.M. iteration)
        def terminalCostFunction(xT, xT_des): return termCostTrkTrj(xT, xT_des, QQT)
        
        print("Solving the affine LQP that gives the descent direction (regularized version of the N.M. is considered)...")
        KK, sigma, _, _, deltau = solveAffineLQP(AA, BB, QQtilde, RRtilde, SStilde, QQT, TT, zeros_like(xx0), qq, rr, qqT)
        descentDirectionNorm = linalg.norm(grdJdu)
        print("Descent direction norm (||-gradJ(u)||): {:.6f}".format(descentDirectionNorm))
        print("Input varation norm given by the N.M. (||deltau||): {:.6f}".format(linalg.norm(deltau)))

        if descentDirectionNorm < tolerance:
            print("The N.M. successfully converged in ", k+1, " iterations!")
            break

        print("Computing the stepsize exploiting the Armijo's rule...")
        stepSizeCollection[k] = armijoStepSize(
            uuCollection[:,:,k], xx_des, uu_des, xx0, ll, deltau, grdJdu, TT, discretizedDynamicFuntion, stageCostFunction, terminalCostFunction,
            stepSizeCollection[k] if (k < maxIterations - 1 and stepSizeCollection[k+1] < 0) else None
        )
        if stepSizeCollection[k] < 0:
            if k > 0:
                print("Rolling-back to the previous iteration and choosing a smaller stepsize (smaller w.r.t {})".format(stepSizeCollection[k-1]))
                k -= 1
                continue
            else:
                print("WARNING: already in the first iteration of the N.M., it was not possible to determine a step size using the Armijo method!")
                break

        # State-input trajectory update (in closed-loop version)
        xxCollection[:,0,k+1] = xx0;
        for tt in range(TT-1):
            uuCollection[:,tt,k+1] = uuCollection[:,tt,k] + stepSizeCollection[k]*deltau[:,tt]
            #uuCollection[:,tt,k+1] = uuCollection[:,tt,k] + KK[:,:,tt]@(xxCollection[:,tt,k+1] - xxCollection[:,tt,k]) + stepsize*sigma[:,tt]
            xxCollection[:,tt+1,k+1] = discretizedDynamicFuntion(xxCollection[:,tt,k+1], uuCollection[:,tt,k+1])[0]
        uuCollection[:,TT-1,k+1] = uuCollection[:,TT-1,k] + stepSizeCollection[k]*deltau[:,TT-1]

        k += 1

    if k >= maxIterations-1:
        print("WARNING: the N.M. was not able to converge (not converging in ", maxIterations, " iterations)!")
    return xxCollection[:,:,k], uuCollection[:,:,k]

def solveCostateEquationTrkTrj(xx, uu, xx_des, uu_des, discretizedDynamicFuntion, stageCostFunction, TT, QQT=None):
    """
    Implementation of the backwards-in-time solution of the costate equation of an Optimal Control Trajectory Tracking Problem
    Notice that in this method are computed (and returned in the following order):
    - the costate trajectory that is the solution of the costate equation (lmbda)
    - the Jacobians of the dynamic wrt x and wrt u at xx,uu at each time step (respectively AA and BB)
    - the Jacobians of the stage cost wrt x and wrt u at xx,uu at each time step (respectively qq and rr)
    - the Jacobian of the terminal cost wrt x at the terminal state value (alias qqT)
    - the transposed Hessians of the stage cost wrt x and wrt u at xx,uu at each time step (QQtilde as d2lldxdx, SStilde as d2lldxdu, RRtilde as d2lldudu)
    - the transposed Hessian of the terminal cost wrt x at the terminal state value (alias QQT)
    - the gradient of the cost function (expressed as only a function of the input) at xx,uu at each time step (alias grdJdu) to use for the Armijo's rule stepsize selection
    - the acutal cost associated to the given trajectory xx,uu having xx_des, uu_des as desired curves
    """
    lmbda = zeros_like(xx)
    AA = zeros((xx.shape[0], xx.shape[0], TT))
    BB = zeros((xx.shape[0], uu.shape[0], TT))
    qq = zeros_like(xx)
    rr = zeros_like(uu)
    QQtilde = zeros((xx.shape[0], xx.shape[0], TT))
    SStilde = zeros((uu.shape[0], xx.shape[0], TT))
    RRtilde = zeros((uu.shape[0], uu.shape[0], TT))
    grdJdu = zeros_like(uu)
    ll = 0
    for tt in reversed(range(TT-1)):
        llTemp, qqTemp, rrTemp, QQtildeTransposed, SStildeTransposed, RRtildeTransposed = stageCostFunction(
            xx[:,tt], uu[:,tt], xx_des[:,tt], uu_des[:,tt], None
        )
        ll += llTemp
        qq[:,tt] = squeeze(qqTemp)
        rr[:,tt] = squeeze(rrTemp)
        QQtilde[:,:,tt] = QQtildeTransposed.T
        SStilde[:,:,tt] = SStildeTransposed.T
        RRtilde[:,:,tt] = RRtildeTransposed.T
        AA[:,:,tt], BB[:,:,tt] = discretizedDynamicFuntion(xx[:,tt], uu[:,tt])[1:3]
        if tt == TT-2:
            # First iteration, definition of the terminal cost matrix as solution of the ARE for the last available instant of time (if required)
            if QQT is None or (all((x == 0 or x == None) for x in QQT.flatten())):
                QQT = solveARE(AA[:,:,tt], BB[:,:,tt], QQtilde[:,:,tt], RRtilde[:,:,tt], SStilde[:,:,tt])
            llTemp, qqT, QQT = termCostTrkTrj(xx[:,-1], xx_des[:,-1], QQT)
            ll += llTemp
            lmbda[:,TT-1] = squeeze(qqT)
        lmbda[:,tt] = qq[:,tt] + AA[:,:,tt].T@lmbda[:,tt+1]
        grdJdu[:,tt] = rr[:,tt] + BB[:,:,tt].T@lmbda[:,tt+1]
    return lmbda, AA, BB, qq, rr, qqT, QQtilde, SStilde, RRtilde, QQT, grdJdu, ll


def solveAffineLQP(AA, BB, QQ, RR, SS, QQT, TT, xx0, qq, rr, qqT):
    """ Affine Linear Quadratic Optimization Problem Solver """

    ns = AA.shape[0]
    ni = BB.shape[1]
    KK = zeros((ni, ns, TT))
    sigma = zeros((ni, TT))
    PP = zeros((ns, ns, TT))
    pp = zeros((ns, TT))
    xx = zeros((ns, TT))
    uu = zeros((ni, TT))
    xx[:,0] = xx0
    PP[:,:,-1] = QQT
    pp[:,-1] = squeeze(qqT)

    # Solve the DRE for each time step
    for tt in reversed(range(TT-1)):
        QQt = QQ[:,:,tt]
        qqt = qq[:,tt][:,None] # Here [:,None] is used to convert a row vector to a column vector
        RRt = RR[:,:,tt]
        rrt = rr[:,tt][:,None] # Here [:,None] is used to convert a row vector to a column vector
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        SSt = SS[:,:,tt]
        PPtp = PP[:,:,tt+1]
        pptp = pp[:, tt+1][:,None] # Here [:,None] is used to convert a row vector to a column vector
        MMt_inv = linalg.inv(RRt + BBt.T @ PPtp @ BBt)
        mmt = rrt + BBt.T @ pptp
        PPt = AAt.T @ PPtp @ AAt - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ (BBt.T@PPtp@AAt + SSt) + QQt
        ppt = AAt.T @ pptp - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ mmt + qqt
        PP[:,:,tt] = PPt
        pp[:,tt] = ppt.squeeze()
    
    # Evaluate KK and sigma
    for tt in range(TT-1):
        QQt = QQ[:,:,tt]
        qqt = qq[:,tt][:,None] # Here [:,None] is used to convert a row vector to a column vector
        RRt = RR[:,:,tt]
        rrt = rr[:,tt][:,None] # Here [:,None] is used to convert a row vector to a column vector
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        SSt = SS[:,:,tt]
        PPtp = PP[:,:,tt+1]
        pptp = pp[:,tt+1][:,None] # Here [:,None] is used to convert a row vector to a column vector
        MMt_inv = linalg.inv(RRt + BBt.T @ PPtp @ BBt)
        mmt = rrt + BBt.T @ pptp
        KK[:,:,tt] = -MMt_inv@(BBt.T@PPtp@AAt + SSt)
        sigma_t = -MMt_inv@mmt
        sigma[:,tt] = sigma_t.squeeze()
    
    # Evaluate the optimal trajectory
    for tt in range(TT - 1):
        uu[:,tt] = KK[:,:,tt]@xx[:, tt] + sigma[:,tt]
        xxp = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt]
        xx[:,tt+1] = xxp
        xxout = xx
        uuout = uu
    return KK, sigma, PP, xxout, uuout


def solveLinearLQP(AA, BB, QQ, RR, QQT, TT, xx0):
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
	
    ns = AA.shape[0]
    ni = BB.shape[1]
    PP = zeros((ns,ns,TT))
    KK = zeros((ni,ns,TT))
    xx = zeros((ns, TT))
    uu = zeros((ni, TT))
    xx[:,0] = xx0
    PP[:,:,-1] = QQT
    
    # Solve Riccati equation
    for tt in reversed(range(TT-1)):
        QQt = QQ[:,:,tt]
        RRt = RR[:,:,tt]
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        PPtp = PP[:,:,tt+1]
        
        PP[:,:,tt] = QQt + AAt.T@linalg.pinv(eye(ns) + PPtp@BBt@linalg.pinv(RRt) @ BBt.T)@PPtp@AAt
    
    # Evaluate KK
    for tt in range(TT-1):
        QQt = QQ[:,:,tt]
        RRt = RR[:,:,tt]
        AAt = AA[:,:,tt]
        BBt = BB[:,:,tt]
        PPtp = PP[:,:,tt+1]
        
        KK[:,:,tt] = - linalg.pinv(RRt + BBt.T@PPtp@BBt)@BBt.T@PPtp@AAt
    
    # Evaluate the optimal trajectory
    for tt in range(TT - 1):
        uu[:,tt] = KK[:,:,tt]@xx[:, tt]
        xxp = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt]
        xx[:,tt+1] = xxp
        xxout = xx
        uuout = uu

    return KK, PP, xxout, uuout


def armijoStepSize(uu, xx_des, uu_des, xx0, ll, deltau, grdJdu, TT, discretizedDynamicFuntion, stageCostFunction, terminalCostFunction, stepsizeInitialGuess=None):
    """ Armijo's Rule for Step Size Selection """

    armijoMaximumIterations = 14 # sufficient to attempt a stepsize of 0.01
    armijoBeta = 0.7
    armijoC = 0.5
    ns = xx_des.shape[0]
    ni = uu_des.shape[0]
    armijoStepsizes = []
    armijoCosts = []

    armijoLinePendence = dot(squeeze(deltau), squeeze(grdJdu))
    print("Armijo line pendence {:.5f}: ({:.1f}° between deltau and grdJu)".format(
        armijoLinePendence, rad2deg(arccos(armijoLinePendence/linalg.norm(deltau)/linalg.norm(grdJdu)))
    ))
    stepsize = float(1 if (stepsizeInitialGuess is None) else stepsizeInitialGuess*armijoBeta*armijoBeta)
    print("Using as initial guess for the Armijo stepsize: {:.10}".format(stepsize))
    for ii in range(armijoMaximumIterations):

        tempxx = zeros((ns,TT))
        tempuu = zeros((ni,TT))
        tempxx[:,0] = xx0
        for tt in range(TT-1):
            tempuu[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
            tempxx[:,tt+1] = discretizedDynamicFuntion(tempxx[:,tt], tempuu[:,tt])[0]
        tempJJ = totalCostFunction(tempxx, tempuu, xx_des, uu_des, TT, stageCostFunction, terminalCostFunction)
        print("New cost achieved by moving with stepsize {:.10}: ".format(stepsize), tempJJ)

        armijoStepsizes.append(stepsize)
        armijoCosts.append(tempJJ)

        if tempJJ >= ll + armijoC*stepsize*armijoLinePendence:
            stepsize = armijoBeta*stepsize
        else:
            print("Detected Armijo stepsize = {:.10} (in {} iterations)".format(stepsize, ii+1))
            break
        if ii == armijoMaximumIterations-1:
            print("WARNING: no stepsize was found applying the Armijo's Rule (not converging in {} iterations)(last stepsize attempted: {:.10})!".format(armijoMaximumIterations, stepsize))
            stepsize = -1

    armijoStepsizes.append(0)
    armijoCosts.append(ll)
    armijoStepsizes = array(armijoStepsizes)
    plt.figure(1); plt.clf()
    plt.plot(armijoStepsizes, armijoCosts, color='k', label='$J(\\mathbf{u}^k+stepsize*d^k)$')
    plt.plot(armijoStepsizes, ll + armijoLinePendence*armijoStepsizes, color='r', label='$J(\\mathbf{u}^k)+stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    plt.plot(armijoStepsizes, ll + armijoC*armijoLinePendence*armijoStepsizes, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k)+stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    plt.scatter(armijoStepsizes, armijoCosts, marker='*')
    plt.grid()
    plt.xlabel('stepsize')
    plt.legend()
    plt.draw()
    plt.show()

    return stepsize


def ComputeLocalLin(xx_star, uu_star, QQt, RRt, QQT, TT, Dynamics, solver_LQP): 
    ns = xx_star.shape[0]
    ni = uu_star.shape[0]
    AA_star = zeros((ns, ns, TT))
    BB_star = zeros((ns, ni, TT))

    # Compute local linearization
    for tt in range(TT): 
        dfdx, dfdu = Dynamics(xx_star[:,tt], uu_star[:,tt])[1:]
        AA_star[:,:,tt] = dfdx
        BB_star[:,:,tt] = dfdu
    
    # Compute LQR gains
    KK = solver_LQP(AA_star, BB_star, QQt, RRt, QQT, TT)[0]
    return KK, AA_star, BB_star

def solveARE(A, B, Q, R, S):
    # https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator#Infinite-horizon,_discrete-time
    augmented = (S is not None) and (not (all((x == 0 or x == None) for x in S.flatten())))
    if augmented:
        aA = A - B@linalg.pinv(R)@S.T
        aQ = Q - S@linalg.pinv(R)@S.T
        return dare(aA, B, aQ, R)[0]
    else: return dare(A, B, Q, R)[0]

def GenerateNoise(xx, noise_std_percentage): 
    # Compute standard deviation for each state row-wise
    state_sd = std(xx, axis=1) # shape: (ns,)

    # Scale the standard deviation by the percentage
    noise_sd = state_sd*noise_std_percentage

    # Generate Gaussian nois with zero mean and scaled (by percentage) std deviation for each state
    noise = random.randn(*xx.shape)*noise_sd[:,newaxis]
    return noise


def SolveLQPwithNoise(xx_star, uu_star, KK, noise, TT, dynamics): 
    ns = xx_star.shape[0]
    ni = uu_star.shape[0]
    xx_track = zeros((ns, TT))
    uu_track = zeros((ni, TT))
    xx_track[:,0] = xx_star[:,0] # Initializing the tracking trajectory as the optimal one
    
    for i in range(ns):
        xx_track[i,0] += noise[i] # Adding the noise on the initial state to consider perturbed initial condition

        for tt in range(TT): 
            uu_track[:,tt] = uu_star[:, tt] + KK[:,:,tt]@(xx_track[:,tt] - xx_star[:,tt]) # Computing the controller using LQR gain in a closed loop fashion
            xx_track[:,tt+1] = dynamics(xx_track[:,tt], uu_track[:,tt])

    return xx_track, uu_track
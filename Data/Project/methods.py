
from numpy import *
from dynamics import runDynamicFunction
from costs import totalCostFunction

def runNewtonMethod(xx_des, uu_des, xx0, TT, maxIterations, discretizedDynamicFuntion, stageCostFunction, terminalCostFunction, tolerance):
    """
    Newton's Like Method in closed loop version for Optimal Control
    Arguments:
    - xx_des: column vector state desired curve (dimension: ns*TT) (TT>ns)
    - uu_des: column vector input desired curve (dimension: ni*TT) (TT>ni)
    - xx0: column vector fixed initial state (from a python variable p.o.f., this is of shape (ns,))
    - TT: number of time steps (each one of duration dt, enough for evolve from t=0 to t=T, where [0, T] is the considered horizon)
    - maxIterations: maximum number of iterations for the method to converge
    - discretizedDynamicFuntion: functions of (xx_t, uu_t) that implements the discretized dynamics of the system that is being considered,
                                    requiring as arguments respectively the state and input values at time t,
                                    returning all the jacobians and hessians of the dynamics wrt them AND the state value at time t+1 in the following order:
                                    xxp, dfdx, dfdu, d2fdxdx, d2fdxdu, d2fdudx, d2fdudu 
    - stageCostFunction: function of (xx_t, uu_t, xx_des_t, uu_des_t) that computes the stage cost requiring as arguments respectively:
                         trajectory state and input values at time t, desired state and input values at time t
    - terminalCostFunction: function of (xT, xT_des) that computes the terminal cost requiring as arguments respectively:
                            trajectory terminal state value and desired terminal state value
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

    xxCollection = zeros((xx_des.shape[0], TT, maxIterations))
    uuCollection = zeros((uu_des.shape[0], TT, maxIterations))
    descentDirectionNormCollection = zeros((maxIterations))

    # Initialization of the N.M. with a feasible trajectory via shooting (running forward and in an open loop fashon the dynamic of the system)
    uuCollection[:,:,0] = uu_des
    xxCollection[:,:,0] = runDynamicFunction(discretizedDynamicFuntion, uu_des, xx0, TT)

    # Execution of the N.M. single step iteration (function of k, the iteration index)
    for k in range(maxIterations-1):

        # Compute the actual cost (at iteration k) of the trajectory xx,uu having xx_des, uu_des as desired curves
        ll = totalCostFunction(xxCollection[:,:,k], uuCollection[:,:,k], xx_des, uu_des, TT, stageCostFunction, terminalCostFunction)

        # Solve the costate equation
        lmbda, AA, BB, qq, rr, qqT, QQtilde, SStilde, RRtilde, QQT, grdJdu = solveCostateEquation(
            xxCollection[:,:,k], uuCollection[:,:,k], xx_des, uu_des, discretizedDynamicFuntion, stageCostFunction, terminalCostFunction, TT
        )

        # Solve the affine LQP that gives the descent direction (notice that the regularized version of the N.M. is considered)
        KK, sigma, _, _, deltau = solveAffineLQP(AA, BB, QQtilde, RRtilde, SStilde, QQT, TT, zeros_like(xx0), qq, rr, qqT)
        descentDirectionNormCollection[k] = linalg.norm(deltau)
        if descentDirectionNormCollection[k] < tolerance:
            print("N.M. converged in {k} iterations!")
            break

        # Comuputing the stepsize with the Armijo's rule
        stepsize = armijoStepSize(
            uuCollection[:,:,k], xx_des, uu_des, xx0, ll, deltau, grdJdu, TT, discretizedDynamicFuntion, stageCostFunction, terminalCostFunction
        )

        # State-input trajectory update (in closed-loop version)
        xxCollection[:,0,k+1] = xx0;
        for tt in range(TT-1):
            uuCollection[:,tt,k+1] = uuCollection[:,tt,k] + KK[:,:,tt]@(xxCollection[:,tt,k+1] - xxCollection[:,tt,k]) + stepsize*sigma[:,tt]
            xxCollection[:,tt+1,k+1] = discretizedDynamicFuntion(xxCollection[:,tt,k+1], uuCollection[:,tt,k+1])[0]

    if k == maxIterations-1:
        print("WARNING: the N.M. was not able to converge (not converging in {maxIterations} iterations!")
    return xxCollection[:,:,k], uuCollection[:,:,k]

def solveCostateEquation(xx, uu, xx_des, uu_des, discretizedDynamicFuntion, stageCostFunction, terminalCostFunction, TT):
    """
    Implementation of the backwards-in-time solution of the costate equation of an Optimal Control Problem
    Notice that in this method are also computed:
    - the Jacobians of the dynamic wrt x and wrt u at xx,uu at each time step (respectively AA and BB)
    - the Jacobians of the stage cost wrt x and wrt u at xx,uu at each time step (respectively qq and rr)
    - the Jacobian of the terminal cost wrt x at the terminal state value (alias qqT)
    - the transposed Hessians of the stage cost wrt x and wrt u at xx,uu at each time step (QQtilde as d2lldxdx, SStilde as d2lldxdu, RRtilde as d2lldudu)
    - the transposed Hessian of the terminal cost wrt x at the terminal state value (alias QQT)
    - the gradient of the cost function (expressed as only a function of the input) at xx,uu at each time step (alias grdJdu) to use for the Armijo's rule stepsize selection
    """
    lmbda = zeros_like(xx)
    AA = zeros((xx.shape[0], xx.shape[0], TT))
    BB = zeros((xx.shape[0], uu.shape[0], TT))
    qq = zeros_like(xx)
    rr = zeros_like(uu)
    QQtilde = zeros((xx.shape[0], xx.shape[0], TT))
    SStilde = zeros((xx.shape[0], uu.shape[0], TT))
    RRtilde = zeros((uu.shape[0], uu.shape[0], TT))
    grdJdu = zeros_like(uu)
    qqT, QQT = terminalCostFunction(xx[:,TT-1], xx_des[:,TT-1])[1:3]
    lmbda[:,TT-1] = squeeze(qqT)
    for tt in reversed(range(TT-1)):
        qq[:,tt] = squeeze(stageCostFunction(xx[:,tt], uu[:,tt], xx_des[:,tt], uu_des[:,tt])[1])
        rr[:,tt] = squeeze(stageCostFunction(xx[:,tt], uu[:,tt], xx_des[:,tt], uu_des[:,tt])[2])
        QQtilde[:,:,tt], SStilde[:,:,tt], RRtilde[:,:,tt] = stageCostFunction(xx[:,tt], uu[:,tt], xx_des[:,tt], uu_des[:,tt])[3:]
        QQtilde[:,:,tt] = QQtilde[:,:,tt].T
        #SStilde[:,:,tt] = SStilde[:,:,tt].T
        RRtilde[:,:,tt] = RRtilde[:,:,tt].T
        AA[:,:,tt], BB[:,:,tt] = discretizedDynamicFuntion(xx[:,tt], uu[:,tt])[1:3]
        lmbda[:,tt] = qq[:,tt] + AA[:,:,tt].T@lmbda[:,tt]
        grdJdu[:,tt] = rr[:,tt] + BB[:,:,tt].T@lmbda[:,tt]
    return lmbda, AA, BB, qq, rr, qqT, QQtilde, SStilde, RRtilde, QQT, grdJdu

def solveAffineLQP(AA, BB, QQ, RR, SS, QQT, TT, xx0, qq, rr, qqT):
    """ Affine Linear Quadratic Optimization Problem Solver """

    ns = AA.shape[0]
    ni = BB.shape[0]
    KK = zeros((ni, ns, TT))
    sigma = zeros((ni, TT))
    PP = zeros((ns, ns, TT))
    pp = zeros((ns, TT))
    xx = zeros((ns, TT))
    uu = zeros((ni, TT))
    xx[:,0] = xx0
    PP[:,:,-1] = QQT
    pp[:,-1] = qqT

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
        xxp = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt] #here
        xx[:,tt+1] = xxp
        xxout = xx
        uuout = uu
    return KK, sigma, PP, xxout, uuout

def armijoStepSize(uu, xx_des, uu_des, xx0, ll, deltau, grdJdu, TT, discretizedDynamicFuntion, stageCostFunction, terminalCostFunction):
    """ Armijo's Rule for Step Size Selection """

    stepsizeInitialGuess = 1
    armijoMaximumIterations = 50
    armijoBeta = 0.7
    armijoC = 0.5
    ns = xx_des.shape[0]
    ni = uu_des.shape[0]
    armijoStepsizes = []
    armijoStepsizesCosts = []
    descendentArmijoDirection = grdJdu.T@deltau

    stepsize = stepsizeInitialGuess
    for ii in range(armijoMaximumIterations):

        xx_temp = zeros((ns,TT))
        uu_temp = zeros((ni,TT))
        xx_temp[:,0] = xx0
        for tt in range(TT-1):
            uu_temp[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
            xx_temp[:,tt+1] = discretizedDynamicFuntion(xx_temp[:,tt], uu_temp[:,tt])[0]
        JJ_temp = 0
        for tt in range(TT-1):
            temp_cost = stageCostFunction(xx_temp[:,tt], uu_temp[:,tt], xx_des[:,tt], uu_des[:,tt])[0]
            JJ_temp += temp_cost
        temp_cost = terminalCostFunction(xx_temp[:,-1], xx_temp[:,-1])[0]
        JJ_temp += temp_cost

        armijoStepsizes.append(stepsize)

        if JJ_temp >= ll + armijoC*stepsize*descendentArmijoDirection:
            stepsize = armijoBeta*stepsize
        else:
            print("Detected Armijo stepsize = {:.3e}".format(stepsize) + " (in {ii} iterations)")
            break
        if ii == armijoMaximumIterations-1:
            print("WARNING: no stepsize was found applying the Armijo's Rule (not converging in {armijoMaximumIterations} iterations!")

    return stepsize

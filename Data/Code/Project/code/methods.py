
# Solver for an Optimal Control Trajectory Tracking Problem (and all the involved functions)

from numpy import repeat, newaxis, zeros, zeros_like, squeeze, linalg, eye, dot, seterr, argmin, allclose
from builtins import all
from control import dare
from dynamics import runDynamicFunction
from costs import totalCostFunction, stageCostTrkTrj, termCostTrkTrj
from datetime import datetime
from miscellaneous import getTimeDifferenceAsString, correctStateInputCurvesShapes

import matplotlib.pyplot as pyplot
from numpy import linspace as linespace

def runNewtonMethodTrkTrj(xx_des, uu_des, xx0, maxIterations, discretizedDynamicFuntion, tolerance, QQ, RR, QQT=None, fixedStepsize=None):
    """
    Newton's Like Method in closed loop version for an Optimal Control Trajectory Tracking Problem.
    
    Arguments:
    - xx_des: column vector state desired curve (dimension: ns*TT) (ns is the number of states of the system)
      (usage: fot t=0 we have the initial state xx0,
              then we have the state curve for t from 1 up to TT-2,
              then value for t=TT-1 is considered as the state terminal value)
    - uu_des: column vector input desired curve (dimension: ni*TT) (ni is the number of inputs of the system)
      (usage: for t from 0 to TT-2, we have the input curve;
              for t=TT-1 we MUST have the input value that MUST be of equilibria for the system dynamic if considered within the terminal state value)
    - xx0: column vector fixed initial state (dimensions: ns*1)
    - maxIterations: maximum number of allowed iterations for the method to converge
    - discretizedDynamicFuntion: function of (xx_t, uu_t) that implements the discretized dynamics of the system that is being considered,
                                 requiring as arguments respectively the state and input values at time t,
                                 returning the state value at time t+1 AND all the jacobians of the dynamic w.r.t. state and input in the following order:
                                 xxp, dfdx, dfdu
    - tolerance: minimum value that the norm of the descent direction has to reach to consider the optimization as converged and completed
    - QQ: nsxns stage state cost matrix (if time invariant) or nsxnsxTT stage state cost tensor (if time variant)
    - RR: nixni stage input cost matrix (if time invariant) or nixnixTT stage input cost tensor (if time variant)
    - QQT: nsxns terminal state cost matrix (if None, the solution of the ARE at t=TT-1 is used for the terminal cost)
    - fixedStepsize: fixed stepsize to use for the optimization (if None, the Armijo's rule is exploited to compute the stepsize)

    Returns:
    - xx: column vector state (feasible) trajectory obtained through the optimization (coupled with uu; returning a state-input trajectory)
    - uu: column vector input (feasible) trajectory obtained through the optimization (coupled with xx; returning a state-input trajectory)
    """
    
    # Compute ni and ns and be sure that the desired curves are in the right shape
    xx_des, uu_des, ns, ni = correctStateInputCurvesShapes(xx_des, uu_des)

    # Retrive the TT value (the number of time steps, each one of duration dt, enough for evolve from t=0 to t=T, where [0, T] is the considered horizon)
    TT = xx_des.shape[1]; TT2 = uu_des.shape[1]
    if TT != TT2: raise ValueError(
        "The desired state curve and the desired input curve do not match in their second dimension, alias TT value ("+str(TT)+" VS "+str(TT2)+")"
    )

    # Definition the stage cost function at the generic istant of time t.
    # QQ and RR are supposed to be time variant, so in general they are 3D tensors.
    # If they are time invariant, they are supposed to be provided as 2D matrices, so here they're repeated along the third dimension.
    if (QQ.ndim < 3): QQ = repeat(QQ[:, :, newaxis], TT, axis=2)
    if (RR.ndim < 3): RR = repeat(RR[:, :, newaxis], TT, axis=2)
    def stageCostFunction(xx_t, uu_t, xx_des_t, uu_des_t, t):
        return stageCostTrkTrj(ns, ni, xx_t, uu_t, xx_des_t, uu_des_t, QQ[:,:,t], RR[:,:,t])
    
    # Definition the terminal state cost matrix at the terminal time T (if no QQT is provided, the ARE solution is used)
    if QQT is None or (all((x == 0 or x == None) for x in QQT.flatten())):
        AT, BT = discretizedDynamicFuntion(xx_des[:,-1], uu_des[:,-1])[1:3]
        QT, ST, RT = stageCostFunction(xx_des[:,-1], uu_des[:,-1], xx_des[:,-1], uu_des[:,-1], TT-1)[3:6]
        QQT = solveARE(AT, BT, QT, RT, ST.T)
    def terminalCostFunction(xT, xT_des): return termCostTrkTrj(ns, xT, xT_des, QQT)

    # Preparing the data structure for the N.M. iterations
    data = TrjTrkOCPData(ns, ni, xx_des, uu_des, TT, maxIterations)

    # Initialization of the N.M. with a feasible trajectory by running forward and in an open loop fashion the dynamic of the system.
    # Using in this run a costant input curve (obtained by repeating the first one value of the desired input curve)
    data.uuCollection[:,:,0] = repeat(uu_des[:,0], TT)
    data.xxCollection[:,:,0] = runDynamicFunction(discretizedDynamicFuntion, data.uuCollection[:,:,0], xx0)

    # Execution of the N.M. single step iteration (where k is the iteration index)
    k = 0
    while k < maxIterations:

        print("\n[>] N.M. now approaching iteration ", format(k+1))

        print("Solving the costate equation (and computing all involved quantities)...")
        _, AA, BB, _, _, _, qq, rr, qqT, QQtilde, SStilde, RRtilde, data.grdJJCollection[:,:,k], ll = solveCostateEquation(
            data.xxCollection[:,:,k], data.uuCollection[:,:,k], xx_des, uu_des,
            discretizedDynamicFuntion, stageCostFunction, terminalCostFunction, TT
        )

        print("Actual cost: ", ll)

        print("Solving the affine LQP that gives the descent direction (regularized version of the N.M. is considered)")
        KK, sigma, _, _, deltau = solveAffineLQP(AA, BB, QQtilde, RRtilde, SStilde, QQT, TT, zeros_like(xx0), qq, rr, qqT)
        descentDirectionNorm = linalg.norm(data.grdJJCollection[:,:,k])
        print("Descent direction norm (||-gradJ(u)||): {:.12f}".format(descentDirectionNorm))
        print("N.M. direction norm (||deltau||): {:.12f}".format(linalg.norm(deltau)))

        if descentDirectionNorm < tolerance:
            print("The N.M. successfully converged in", k+1, "iterations (required time: {})!".format(data.getElapsedTime()))
            break
        
        direction = deltau
        if fixedStepsize is None:
            print("Computing the stepsize exploiting the Armijo's rule (relying on the Newton Direction)...")
            stepsize, armijoStepsizes, armijoCosts = armijoStepSize(
                data.uuCollection[:,:,k], data.xxCollection[:,:,k], xx_des, uu_des, xx0,
                ll, direction, data.grdJJCollection[:,:,k], KK, sigma, TT,
                discretizedDynamicFuntion, stageCostFunction, terminalCostFunction
            )
            data.armijoStepsizesCollection.append(armijoStepsizes)
            data.armijoCostsCollection.append(armijoCosts)
            if stepsize > 0:
                print("After exploiting the Armijo's rule, using as stepsize: {:.10}".format(stepsize))
            else:
                print("ERROR: Armijo's rule failed, alias, in the moving direction the cost is always increasing, even for small stepsizes!")
                print("The method failed in its minimum search, now ending (elapsed time: {})!".format(data.getElapsedTime()))
                break
        else:
            print("Using as stepsize the given fixed value: {:.10}".format(fixedStepsize))
            stepsize = fixedStepsize

        print("Updating the state-input trajectory (in closed-loop version)")
        data.uuCollection[:,:,k+1], data.xxCollection[:,:,k+1] = updateInputStateTrajectory(
            ns, ni, xx0, data.uuCollection[:,:,k], data.xxCollection[:,:,k], stepsize, direction, KK, sigma, TT, discretizedDynamicFuntion
        )

        k += 1 # Incrementing the iteration index

        if k >= maxIterations-1:
            print("WARNING: the N.M. was not able to converge (not converging in ", maxIterations, " iterations) (elapsed time: {})!".format(data.getElapsedTime()))
            break

    data.K = k
    return data

def solveCostateEquation(xx, uu, xx_des, uu_des, discretizedDynamicFuntion, stageCostFunction, termCostFunction, TT):
    """
    Implementation of the backwards-in-time solution of the costate equation of an Optimal Control Problem.
    Notice that in this method are computed (and returned in the following order):
    - the costate trajectory, alias the solution of the costate equation (lmbda)
    - the jacobians of the dynamic w.r.t. x and w.r.t. u at xx,uu at each time step (respectively AA and BB)
    - the [transposed] hessians of the dynamic w.r.t. x and w.r.t. u at xx,uu at each time step (QQdyn as d2fdxdx, SSdyn as d2fdxdu, RRdyn as d2fdudu)
    - the transposed jacobians of the stage cost w.r.t. x and w.r.t. u at xx,uu at each time step (respectively qq and rr)
    - the transposed jacobian of the terminal cost w.r.t. x at the terminal state value (alias qqT)
    - the [transposed] hessians of the stage cost w.r.t. x and w.r.t. u at xx,uu at each time step (QQtilde as d2lldxdx, SStilde as d2lldxdu, RRtilde as d2lldudu)
    - the gradient of the cost function (expressed as only a function of the input) at xx,uu at each time step (alias grdJdu)
    - the acutal cost associated to the given trajectory xx,uu having xx_des, uu_des as desired curves (alias ll)
    """
    lmbda = zeros_like(xx)
    ns = xx.shape[0]
    ni = uu.shape[0]
    AA = zeros((ns, ns, TT))
    BB = zeros((ns, ni, TT))
    qq = zeros_like(xx)
    rr = zeros_like(uu)
    QQtilde = zeros((ns, ns, TT))
    SStilde = zeros((ni, ns, TT))
    RRtilde = zeros((ni, ni, TT))
    QQdyn = zeros((ns, ns, ns, TT))
    SSdyn = zeros((ns, ni, ns, TT))
    RRdyn = zeros((ns, ni, ni, TT))
    grdJdu = zeros_like(uu)
    ll, qqT, _ = termCostFunction(xx[:,-1], xx_des[:,-1])
    lmbda[:,TT-1] = squeeze(qqT)
    for tt in reversed(range(TT-1)):
        llTemp, qqTemp, rrTemp, QQtildeTransposed, SStildeTransposed, RRtildeTransposed = stageCostFunction(
            xx[:,tt], uu[:,tt], xx_des[:,tt], uu_des[:,tt], tt
        )
        ll += llTemp
        qq[:,tt] = squeeze(qqTemp)
        rr[:,tt] = squeeze(rrTemp)
        QQtilde[:,:,tt] = QQtildeTransposed # Already transposed (in a sense) cause that Hessian is symmetric
        SStilde[:,:,tt] = SStildeTransposed.T
        RRtilde[:,:,tt] = RRtildeTransposed # Already transposed (in a sense) cause that Hessian is symmetric
        # AA[:,:,tt], BB[:,:,tt], QQdyn[:,:,:,tt], _, SSdyn[:,:,:,tt], RRdyn[:,:,:,tt] = discretizedDynamicFuntion(xx[:,tt], uu[:,tt])[1:]
        AA[:,:,tt], BB[:,:,tt] = discretizedDynamicFuntion(xx[:,tt], uu[:,tt])[1:]
        lmbda[:,tt] = qq[:,tt] + AA[:,:,tt].T@lmbda[:,tt+1]
        grdJdu[:,tt] = rr[:,tt] + BB[:,:,tt].T@lmbda[:,tt+1]
    return lmbda, AA, BB, QQdyn, SSdyn, RRdyn, qq, rr, qqT, QQtilde, SStilde, RRtilde, grdJdu, ll

def solveAffineLQP(AA, BB, QQ, RR, SS, QQT, TT, xx0, qq, rr, qqT):
    """ Affine Linear Quadratic Optimal Control Problem Solver """
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

def solveLQP(AA, BB, QQ, RR, QQT, TT, xx0):
    """ Linear Quadratic Optimal Control Problem Solver """
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

def armijoStepSize(uu, xx, xx_des, uu_des, xx0, ll, direction, grdJdu, KK, sigma, TT, discretizedDynamicFuntion, stageCostFunction, terminalCostFunction, stepsizeInitialGuess = None):
    """ Armijo's Rule for Step Size Selection """

    armijoMaximumIterations = 14
    armijoBeta = 0.7
    armijoC = 0.5
    ns = xx_des.shape[0]
    ni = uu_des.shape[0]
    armijoStepsizes = []
    armijoCosts = []

    armijoLinePendence = dot(squeeze(direction), squeeze(grdJdu))
    print(" | Armijo line pendence: {:.16f} (alias dot-product between gradJ(u) and the moving direction)".format(armijoLinePendence))
    stepsizeInitialGuess = float(1 if (stepsizeInitialGuess is None) else stepsizeInitialGuess)
    stepsize = stepsizeInitialGuess
    print(" | Using as initial guess for the Armijo stepsize: {:.10}".format(stepsize))

    for ii in range(armijoMaximumIterations):

        originalNPErrSettings = seterr(all='ignore')
        try:
            tempuu, tempxx = updateInputStateTrajectory(ns, ni, xx0, uu, xx, stepsize, direction, KK, sigma, TT, discretizedDynamicFuntion)
            tempJJ = totalCostFunction(tempxx, tempuu, xx_des, uu_des, TT, stageCostFunction, terminalCostFunction)
        except linalg.LinAlgError:
            tempJJ = -1
            pass
        finally: seterr(**originalNPErrSettings)

        print(" | New cost achieved by moving with stepsize {:.10}:".format(stepsize), " not evaluable (retrived a LinAlgError)" if tempJJ < 0 else tempJJ)
        if tempJJ >= 0:
            armijoStepsizes.append(stepsize)
            armijoCosts.append(tempJJ)

        if tempJJ < 0 or tempJJ >= ll + armijoC*stepsize*armijoLinePendence:
            stepsize = armijoBeta*stepsize
        else:
            print(" | Detected Armijo stepsize = {:.10} (in {} iterations)".format(stepsize, ii+1))
            break
        if ii >= armijoMaximumIterations-1:
            print(" | WARNING: no stepsize was found applying the Armijo's Rule (not converging in {} iterations)(last stepsize attempted: {:.10})!".format(
                armijoMaximumIterations, stepsize/armijoBeta
            ))
            minCostIndex = argmin(armijoCosts)
            if armijoCosts[minCostIndex] < ll:
                stepsize = armijoStepsizes[minCostIndex]
                print(" | One (or more) of the tested stepsizes led to a cost lower than the initial one, so the best one of them is selected:", stepsize)
            else:
                print(" | WARNING: All the attempted stepsizes leads to a cost higher than the initial one!")
                stepsize = -1

    armijoStepsizes.append(0)
    armijoCosts.append(ll)
    return stepsize, armijoStepsizes, armijoCosts

def solveARE(A, B, Q, R, S):
    # https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator#Infinite-horizon,_discrete-time
    augmented = (S is not None) and (not (all((x == 0 or x == None) for x in S.flatten())))
    if augmented:
        aA = A - B@linalg.pinv(R)@S.T
        aQ = Q - S@linalg.pinv(R)@S.T
        return dare(aA, B, aQ, R)[0]
    else: return dare(A, B, Q, R)[0]

def isAPositiveDefiniteMatrix(m):
    if not allclose(m, m.T):
        return False
    try:
        linalg.cholesky(m)
        return True
    except linalg.LinAlgError:
        return False

def updateInputStateTrajectory(ns, ni, xx0, uu_old, xx_old, stepsize, deltau, KK, sigma, TT, discretizedDynamicFuntion):
    uu_new = zeros((ni, TT))
    xx_new = zeros((ns, TT))
    xx_new[:,0] = xx0
    if xx_old is None or KK is None or sigma is None:
        # Open-loop version
        for tt in range(TT-1):
            uu_new[:,tt] = uu_old[:,tt] + stepsize*deltau[:,tt]
            xx_new[:,tt+1] = discretizedDynamicFuntion(xx_new[:,tt], uu_new[:,tt])[0]
        return uu_new, xx_new
    else:
        # Closed-loop version
        for tt in range(TT-1):
            uu_new[:,tt] = uu_old[:,tt] + KK[:,:,tt]@(xx_new[:,tt] - xx_old[:,tt]) + stepsize*sigma[:,tt]
            xx_new[:,tt+1] = discretizedDynamicFuntion(xx_new[:,tt], uu_new[:,tt])[0]
        return uu_new, xx_new

class TrjTrkOCPData:
    def __init__(self, ns, ni, xxDesired, uuDesired, TT, maxIterations):
        self.startingTime = datetime.now()
        self.endingTime = None
        self.ns = ns
        self.ni = ni
        self.xxDesired = xxDesired
        self.uuDesired = uuDesired
        self.xxCollection = zeros((ns, TT, maxIterations))
        self.uuCollection = zeros((ni, TT, maxIterations))
        self.grdJJCollection = zeros_like(self.uuCollection)
        self.armijoStepsizesCollection =  [[] for _ in range(maxIterations)]
        self.armijoCostsCollection =  [[] for _ in range(maxIterations)]
        self.K = None
    def setEndingTime(self):
        self.endingTime = datetime.now()
    def getElapsedTime(self):
        if self.endingTime == None: self.setEndingTime()
        return getTimeDifferenceAsString(self.endingTime, self.startingTime)
    def getOptimalTrajectory(self):
        return self.xxCollection[:,:,self.K], self.uuCollection[:,:,self.K]
    def getOptimalCost(self):
        return self.grdJJCollection[:,:,self.K]
    def getOptimalTrajectoryErrorsAtFinalTime(self):
        x,u = self.getOptimalTrajectory()
        xd = self.xxDesired
        ud = self.uuDesired
        return x[:,-1]-xd[:,-1], u[:,-2]-ud[:,-2]
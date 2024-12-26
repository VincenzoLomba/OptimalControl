
from numpy import *
from builtins import all
from control import dare
from dynamics import runDynamicFunction
from costs import totalCostFunction, stageCostTrkTrj, termCostTrkTrj
from matplotlib import pyplot as plt

def runNewtonMethodTrkTrj(xx_des, uu_des, xx0, TT, maxIterations, discretizedDynamicFuntion, tolerance, QQ, RR, givenQQT=None, givenFixedStepsize=None):
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
    - QQ: nsxns stage state cost matrix (if time invariant) or nsxnsxTT stage state cost tensor (if time variant)
    - RR: nixni stage input cost matrix (if time invariant) or nixnixTT stage input cost tensor (if time variant)
    - givenQQT: nsxns terminal state cost matrix (if None, the solution of the ARE is used for the terminal cost)
    - givenFixedStepsize: fixed stepsize to use for the optimization (if None, the Armijo's rule is exploited to compute the stepsize)

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
    if (QQ.ndim < 3): QQ = repeat(QQ[:, :, newaxis], TT, axis=2)
    if (RR.ndim < 3): RR = repeat(QQ[:, :, newaxis], TT, axis=2)
    def stageCostFunction(xx_t, uu_t, xx_des_t, uu_des_t, t):
        return stageCostTrkTrj(xx_t, uu_t, xx_des_t, uu_des_t, QQ[:,:,t], RR[:,:,t])

    # Initialization of the collections
    xxCollection = zeros((xx_des.shape[0], TT, maxIterations))
    uuCollection = zeros((uu_des.shape[0], TT, maxIterations))

    # Initialization of the N.M. with a feasible trajectory via shooting (running forward and in an open loop fashion the dynamic of the system)
    uuCollection[:,:,0] = uu_des
    xxCollection[:,:,0] = runDynamicFunction(discretizedDynamicFuntion, uu_des, xx0, TT)

    # Execution of the N.M. single step iteration (where k is the iteration index)
    k = 0
    while k < maxIterations:

        print("\n[E] N.M. now approaching iteration ", format(k+1))

        print("Solving the costate equation (and computing all involved quantities)...")
        lmbda, AA, BB, QQext, SSext, RRext, qq, rr, qqT, QQtilde, SStilde, RRtilde, QQT, grdJdu, ll = solveCostateEquationTrkTrj(
            xxCollection[:,:,k], uuCollection[:,:,k], xx_des, uu_des, discretizedDynamicFuntion, stageCostFunction, TT, givenQQT
        )

        print("Actual cost: ", ll)

        # Definition the terminal cost function at time T (for this particular N.M. iteration)
        def terminalCostFunction(xT, xT_des): return termCostTrkTrj(xT, xT_des, QQT)

        print("Computing the matrices for the affine LQP that gives the descent direction (and checking if regularization is needed)...")
        regularizationNeeded = False
        LQQQ = QQtilde
        LQSS = SStilde
        LQRR = RRtilde
        for tt in range(TT-1):
            for ii in range(ns):
                LQQQ[:,:,tt] += QQext[ii,:,:,tt]*lmbda[ii,tt+1]
            if not isAPositiveDefiniteMatrix(LQQQ[:,:,tt]):
                regularizationNeeded = True
                LQQQ = QQtilde
                break
            else:
                for ii in range(ns):
                    LQSS[:,:,tt] += SSext[ii,:,:,tt]*lmbda[ii,tt+1]
                if not isAPositiveDefiniteMatrix(LQSS[:,:,tt]):
                    regularizationNeeded = True
                    LQQQ = QQtilde
                    LQSS = SStilde
                    break
                else:
                    for ii in range(ni):
                        LQRR[:,:,tt] += RRext[ii,:,:,tt]*lmbda[ii,tt+1]
                    if not isAPositiveDefiniteMatrix(LQRR[:,:,tt]):
                        regularizationNeeded = True
                        LQQQ = QQtilde
                        LQSS = SStilde
                        LQRR = RRtilde
                        break

        print("Solving the affine LQP that gives the descent direction {}...".format(
            "(regularized version of the N.M. is considered, cause needed)" if regularizationNeeded else ""
        ))
        KK, sigma, _, _, deltau = solveAffineLQP(AA, BB, LQQQ, LQRR, LQSS, QQT, TT, zeros_like(xx0), qq, rr, qqT)
        descentDirectionNorm = linalg.norm(grdJdu)
        print("Descent direction norm (||-gradJ(u)||): {:.12f}".format(descentDirectionNorm))
        print("Input varation norm given by the N.M. (||deltau||): {:.12f}".format(linalg.norm(deltau)))

        if descentDirectionNorm < tolerance:
            print("The N.M. successfully converged in ", k+1, " iterations!")
            break
        
        if givenFixedStepsize is None:
            print("Computing the stepsize exploiting the Armijo's rule...")
            stepsize = armijoStepSize(
                uuCollection[:,:,k], xxCollection[:,:,k],xx_des, uu_des, xx0, ll, deltau, grdJdu, KK, sigma, TT, discretizedDynamicFuntion,
                stageCostFunction, terminalCostFunction
            )
            print("After exploiting the Armijo's rule, using as stepsize: {:.10}".format(stepsize))
        else:
            print("Using as stepsize the given fixed value: {:.10}".format(givenFixedStepsize))
            stepsize = givenFixedStepsize

        print("Updating the state-input trajectory (in closed-loop version)")
        uuCollection[:,:,k+1], xxCollection[:,:,k+1] = updateInputStateTrajectory(
            xx0, uuCollection[:,:,k], xxCollection[:,:,k], stepsize, deltau, KK, sigma, TT, ni, ns, discretizedDynamicFuntion
        )

        k += 1 # Increment the iteration index

    if k >= maxIterations-1:
        print("WARNING: the N.M. was not able to converge (not converging in ", maxIterations, " iterations)!")
    return xxCollection[:,:,k], uuCollection[:,:,k]

def solveCostateEquationTrkTrj(xx, uu, xx_des, uu_des, discretizedDynamicFuntion, stageCostFunction, TT, QQT=None):
    """
    Implementation of the backwards-in-time solution of the costate equation of an Optimal Control Trajectory Tracking Problem
    Notice that in this method are computed (and returned in the following order):
    - the costate trajectory that is the solution of the costate equation (lmbda)
    - the Jacobians of the dynamic wrt x and wrt u at xx,uu at each time step (respectively AA and BB)
    - the transposed Hessians of the dynamic wrt x and wrt u at xx,uu at each time step (QQext as d2fdxdx, SSext as d2fdxdu, RRext as d2fdudu)
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
    QQext = zeros((xx.shape[0], xx.shape[0], xx.shape[0], TT))
    SSext = zeros((xx.shape[0], uu.shape[0], xx.shape[0], TT))
    RRext = zeros((xx.shape[0], uu.shape[0], uu.shape[0], TT))
    grdJdu = zeros_like(uu)
    ll = 0
    for tt in reversed(range(TT-1)):
        llTemp, qqTemp, rrTemp, QQtildeTransposed, SStildeTransposed, RRtildeTransposed = stageCostFunction(
            xx[:,tt], uu[:,tt], xx_des[:,tt], uu_des[:,tt], tt
        )
        ll += llTemp
        qq[:,tt] = squeeze(qqTemp)
        rr[:,tt] = squeeze(rrTemp)
        QQtilde[:,:,tt] = QQtildeTransposed # Already transposed (in a sense) cause the Hessian is symmetric; QQtilde[:,:,tt] = QQtildeTransposed.T
        SStilde[:,:,tt] = SStildeTransposed.T
        RRtilde[:,:,tt] = RRtildeTransposed # Already transposed (in a sense) cause the Hessian is symmetric; RRtilde[:,:,tt] = RRtildeTransposed.T
        AA[:,:,tt], BB[:,:,tt], QQext[:,:,:,tt], _, SSext[:,:,:,tt], RRext[:,:,:,tt] = discretizedDynamicFuntion(xx[:,tt], uu[:,tt])[1:]
        if tt == TT-2:
            # First iteration, definition of the terminal cost matrix as solution of the ARE for the last available instant of time (if required)
            if QQT is None or (all((x == 0 or x == None) for x in QQT.flatten())):
                QQT = solveARE(AA[:,:,tt], BB[:,:,tt], QQtilde[:,:,tt], RRtilde[:,:,tt], SStilde[:,:,tt])
            llTemp, qqT, QQT = termCostTrkTrj(xx[:,-1], xx_des[:,-1], QQT)
            ll += llTemp
            lmbda[:,TT-1] = squeeze(qqT)
        lmbda[:,tt] = qq[:,tt] + AA[:,:,tt].T@lmbda[:,tt+1]
        grdJdu[:,tt] = rr[:,tt] + BB[:,:,tt].T@lmbda[:,tt+1]

    return lmbda, AA, BB, QQext, SSext, RRext, qq, rr, qqT, QQtilde, SStilde, RRtilde, QQT, grdJdu, ll


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


def armijoStepSize(uu, xx, xx_des, uu_des, xx0, ll, deltau, grdJdu, KK, sigma, TT, discretizedDynamicFuntion, stageCostFunction, terminalCostFunction, stepsizeInitialGuess=None):
    """ Armijo's Rule for Step Size Selection """

    armijoMaximumIterations = 10 # sufficient to attempt a stepsize of 0.04
    armijoBeta = 0.7
    armijoC = 0.5
    ns = xx_des.shape[0]
    ni = uu_des.shape[0]
    armijoStepsizes = []
    armijoCosts = []

    armijoLinePendence = dot(squeeze(deltau), squeeze(grdJdu))
    print(" | Armijo line pendence: {:.16f} (alias dot-product between gradJ(u) and deltau)".format(armijoLinePendence))
    stepsizeInitialGuess = float(1 if (stepsizeInitialGuess is None) else stepsizeInitialGuess)
    stepsize = stepsizeInitialGuess
    print(" | Using as initial guess for the Armijo stepsize: {:.10}".format(stepsize))
    for ii in range(armijoMaximumIterations):

        tempuu, tempxx = updateInputStateTrajectory(
            xx0, uu, xx, stepsize, deltau, KK, sigma, TT, ni, ns, discretizedDynamicFuntion
        )
        tempJJ = totalCostFunction(tempxx, tempuu, xx_des, uu_des, TT, stageCostFunction, terminalCostFunction)
        print(" | New cost achieved by moving with stepsize {:.10}: ".format(stepsize), tempJJ)

        armijoStepsizes.append(stepsize)
        armijoCosts.append(tempJJ)

        if tempJJ >= ll + armijoC*stepsize*armijoLinePendence:
            stepsize = armijoBeta*stepsize
        else:
            print(" | Detected Armijo stepsize = {:.10} (in {} iterations)".format(stepsize, ii+1))
            break
        if ii == armijoMaximumIterations-1:
            print(" | WARNING: no stepsize was found applying the Armijo's Rule (not converging in {} iterations)(last stepsize attempted: {:.10})!".format(
                armijoMaximumIterations, stepsize/armijoBeta
            ))
            stepsize = stepsizeInitialGuess

    # Plot the Armijo's Rule stepsize selection behavior
    armijoStepsizes.append(0)
    armijoCosts.append(ll)
    armijoStepsizes = array(armijoStepsizes)
    plt.figure()
    plt.clf()
    plt.title("Armijo's Rule Step Size Selection Behavior")
    plt.plot(armijoStepsizes, armijoCosts, color='k', label='$J(\\mathbf{u}^k+stepsize*d^k)$')
    plt.plot(armijoStepsizes, ll + armijoLinePendence*armijoStepsizes, color='r', label='$J(\\mathbf{u}^k)+stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    plt.plot(armijoStepsizes, ll + armijoC*armijoLinePendence*armijoStepsizes, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k)+stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    plt.scatter(armijoStepsizes, armijoCosts, marker='*')
    plt.grid()
    plt.xlabel('stepsize')
    plt.legend()
    plt.show(block=False)
    plt.pause(0.5)

    return stepsize

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
        linalg.cholesky(matrix)
        return True
    except linalg.LinAlgError:
        return False

def updateInputStateTrajectory(xx0, uu_old, xx_old, stepsize, deltau, KK, sigma, TT, ni, ns, discretizedDynamicFuntion):

    uu_new = zeros((ni, TT))
    xx_new = zeros((ns, TT))
    xx_new[:,0] = xx0
    if xx_old is None or KK is None or sigma is None:
        # Open-loop version
        for tt in range(TT-1):
            uu_new[:,tt] = uu_old[:,tt] + stepsize*deltau[:,tt]
            xx_new[:,tt+1] = discretizedDynamicFuntion(xx_new[:,tt], uu_new[:,tt])[0]
        uu_new[:,TT-1] = uu_old[:,TT-1] + stepsize*deltau[:,TT-1]
        return uu_new, xx_new
    else:
        # Closed-loop version
        for tt in range(TT-1):
            uu_new[:,tt] = uu_old[:,tt] + KK[:,:,tt]@(xx_new[:,tt] - xx_old[:,tt]) + stepsize*sigma[:,tt]
            xx_new[:,tt+1] = discretizedDynamicFuntion(xx_new[:,tt], uu_new[:,tt])[0]
        uu_new[:,TT-1] = uu_old[:,TT-1] + KK[:,:,TT-1]@(xx_new[:,TT-1] - xx_old[:,TT-1]) + stepsize*sigma[:,TT-1]
        return uu_new, xx_new

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
        xx_track[i,0] += noise[i] # Adding the noise on the tracking trajectory initial state

        for tt in range(TT): 
            uu_track[:,tt] = uu_star[:, tt] + KK[:,:,tt]@(xx_track[:,tt] - xx_star[:,tt]) # Computing the controller using LQR gain in a closed loop fashion
            xx_track[:,tt+1] = dynamics(xx_track[:,tt], uu_track[:,tt])

    return xx_track, uu_track

def plots(tx, tu, xx_des, xx_opt, uu_des, uu_opt):
    """
    Generates 4 subplots for desired and optimal trajectories
    and a separate plot for desired and optimal input trajectories.

    Parameters:
    - tx: Time array for the state trajectories
    - tu: Time array for the inpu trajectory
    - xx_des: Array of desired trajectories (ns x len(tx))
    - xx_opt: Array of optimal trajectories (ns x len(tx))
    - uu_des: Array of desired input (len(tx))
    - uu_opt: Array of optimal input (len(tx))
    """
    ns = xx_des.shape[0]  # Number of trajectories
    colormap = cm.get_cmap('tab10', ns)

    # Create 4 subplots for desired and optimal trajectories
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()  # Flatten the 2x2 grid for easier indexing

    for i in range(ns):
        ax = axes[i]  # Select the current subplot
        color = colormap(i)
        ax.plot(tx, xx_des[i, :], color=color, label=f'ϑ{i+1} desired')
        ax.plot(tx, xx_opt[i, :], '--', color=color, label=f'ϑ{i+1} optimal')

        ax.set_title(f'Theta {i}')  # Set subplot title
        ax.legend()  # Add legend
        ax.set_xlabel('Time')  # X-axis label
        ax.set_ylabel('Value')  # Y-axis label

    plt.tight_layout()  # Optimize layout to avoid overlap
    plt.show()  # Display the figure

    # Create a separate plot for desired and optimal input trajectories
    plt.figure(figsize=(8, 6))

    line_input_des, = plt.plot(tu, uu_des, label='Desired input')
    color = line_input_des.get_color()  # Get color of the desired input line
    plt.plot(tu, uu_opt[0, :], '--', color=color, label='Optimal input')

    plt.title('Input Trajectory')  # Set title for input trajectory plot
    plt.xlabel('Time')  # X-axis label
    plt.ylabel('Value')  # Y-axis label
    plt.legend()  # Add legend
    plt.grid(True)  # Add grid for better visualization

    plt.show()  # Display the figure


# Collection of functions that implement the Tegulators used in the Project (and correlated functions)

from miscellaneous import correctStateInputCurvesShapes
from numpy import zeros, std, random
from cvxpy import Variable, Minimize, Problem, quad_form
from solver import solveLQP


def runLQRController(xx_traj, uu_traj, KK, discretizedDynamicFunction, xx0Noise = None): 

    xx_traj, uu_traj, ns, ni, TT = correctStateInputCurvesShapes(xx_traj, uu_traj)
    xx_track = zeros((ns, TT))
    uu_track = zeros((ni, TT))

    xx_track[:,0] = xx_traj[:,0] + (xx0Noise if xx0Noise is not None else zeros(ns))

    for tt in range(TT-1): 
        # Evolving the dynamic of the system using the LQR (in a closed loop fashion)
        uu_track[:,tt] = uu_traj[:, tt] + KK[:,:,tt]@(xx_track[:,tt] - xx_traj[:,tt])
        xx_track[:,tt+1] = discretizedDynamicFunction(xx_track[:,tt], uu_track[:,tt], onlyZeroOrderDynamic = True)

    return xx_track, uu_track

def runMPController(xx_traj, uu_traj, AA, BB, QQ, RR, QQT, MPC_TT, discretizedDynamicFunction, xx0Noise = None, additionalConstraints = False): 

    xx_traj, uu_traj, ns, ni, TT = correctStateInputCurvesShapes(xx_traj, uu_traj)
    xx_track = zeros((ns, TT))
    uu_track = zeros((ni, TT))

    xx_track[:,0] = xx_traj[:,0] + (xx0Noise if xx0Noise is not None else 0)

    for tt in range(TT-MPC_TT-1): 
        xxt = xx_track[:,tt]
        if additionalConstraints:
            KK = solveLQP(AA[:,:,tt:tt+MPC_TT], BB[:,:,tt:tt+MPC_TT], QQ, RR, QQT, MPC_TT, xxt)[0]
            uu_track[:,tt] = uu_traj[:, tt] + KK[:,:,0]@(xx_track[:,tt] - xx_traj[:,tt])
        else:
            uu_track[:,tt] = solveConstraintLQPCVX(
                xxt, xx_traj[:,tt:tt+MPC_TT], uu_traj[:,tt:tt+MPC_TT],
                AA[:,:,tt:tt+MPC_TT], BB[:,:,tt:tt+MPC_TT], QQ, RR, QQT, MPC_TT
            )[2]
        xx_track[:,tt+1] = discretizedDynamicFunction(xx_track[:,tt], uu_track[:,tt], onlyZeroOrderDynamic = True)
    
    return xx_track, uu_track

def solveConstraintLQPCVX(xxt, xx_des, uu_des, AA, BB, QQ, RR, QQT, MPC_TT): 
    '''
        This method solves a linear constrained MPC problem by the use of cvxpy library

        Arguments: 
            1) xxt initial condition at time t
            2) AA, BB: Linearized dynamics matrices
            3) QQ, RR, QQT: Cost matrices
            4) TT_mpc: Prediction time horizon

        Outputs: 
            1) Γ(t) := (xx, uu)(t): Predicted trajectory
            2) Optimal input uut to be applied to xx at istant t
    '''
    xx_des, uu_des, ns, ni, _ = correctStateInputCurvesShapes(xx_des, uu_des)
    xxt = xxt.squeeze()

    # Definition of problem decision variables
    xx = Variable((ns, MPC_TT))
    uu = Variable((ni, MPC_TT))
    # Definition of cost function and constraints variables
    cost = 0
    constr = []
    # Definition the initial condition constraint
    constr += [xx[:,0] == xxt]
    # Definition of the stage cost functions and the equality constraints related to the linearized dynamics
    for tt in range(MPC_TT-1): 
        cost += quad_form((xx[:,tt]-xx_des[:,tt]), QQ) + quad_form((uu[:,tt]-uu_des[:,tt]), RR)
        constr += [xx[:,tt+1] == AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:,tt]]
    # Definition of the terminal cost function
    cost += quad_form((xx[:,MPC_TT-1] - xx_des[:,MPC_TT-1]), QQT)

    # Solving the MPC problem
    problem = Problem(Minimize(cost), constr)
    problem.solve()
    if problem.status in ["infeasible", "infeasible_inaccurate"]: raise RuntimeError("Unfeasible MPC problem!")
    inputActionFirstValue = uu[:,0].value
    return uu.value, xx.value, inputActionFirstValue

def generateInitialStateNoise(xx, noiseStdPercentage, gainK = 1, randomNumberGenerator = None):
    
    ns = xx.shape[0]
    if not noiseStdPercentage or noiseStdPercentage <= 0: return zeros(ns)

    # Fix a local r.n.g. related to a local seed in order to generate the same noise (in the various Tasks)
    if randomNumberGenerator is None: randomNumberGenerator = random.default_rng(2828)  

    # Compute the standard deviation for each state (row-wise)
    stateSD = std(xx, axis = 1)
    # Scale the standard deviation by the percentage
    noiseSD = gainK*stateSD*noiseStdPercentage
    
    # Generate (and return) a Gaussian noise with personalized standard deviation taking
    # samples from a N(0,1) normal distribution and scaling them by noiseStdPercentage
    return randomNumberGenerator.normal(loc = 0.0, scale = noiseSD, size = ns)
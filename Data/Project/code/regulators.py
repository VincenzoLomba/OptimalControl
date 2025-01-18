
# Collection of functions that implement the Tegulators used in the Project (and correlated functions)

from miscellaneous import correctStateInputCurvesShapes
from numpy import zeros, std, random

def runLQRController(xx_traj, uu_traj, KK, discretizedDynamicFunction, xx0Noise = None): 

    xx_traj, uu_traj, ns, ni, TT = correctStateInputCurvesShapes(xx_traj, uu_traj)
    xx_track = zeros((ns, TT))
    uu_track = zeros((ni, TT))

    xx_track[:,0] = xx_traj[:,0] + (xx0Noise if xx0Noise is not None else zeros(ns))

    for tt in range(TT-1): 
        # Evolving the dynamic of the system using the LQR (in a closed loop fashion)
        uu_track[:,tt] = uu_traj[:, tt] + KK[:,:,tt]@(xx_track[:,tt] - xx_traj[:,tt])
        xx_track[:,tt+1] = discretizedDynamicFunction(xx_track[:,tt], uu_track[:,tt])[0]

    return xx_track, uu_track

def generateInitialStateNoise(xx, noiseStdPercentage, K = None, randomNumberGenerator = None):
    
    ns = xx.shape[0]
    if not noiseStdPercentage or noiseStdPercentage <= 0: return zeros(ns)
    if not K or K <= 0: K = 1

    # Fix a local r.n.g. related to a local seed in order to generate the same noise (in the various Tasks)
    if randomNumberGenerator is None: randomNumberGenerator = random.default_rng(2828)  

    # Compute the standard deviation for each state (row-wise)
    stateSD = std(xx, axis = 1)
    # Scale the standard deviation by the percentage
    noiseSD = K*stateSD*noiseStdPercentage
    
    # Generate (and return) a Gaussian noise with personalized standard deviation taking
    # samples from a N(0,1) normal distribution and scaling them by noiseStdPercentage
    return randomNumberGenerator.normal(loc = 0.0, scale = 1.0, size = ns)*noiseSD
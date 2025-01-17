
# Collection of functions that implement the Tegulators used in the Project (and correlated functions)

from miscellaneous import correctStateInputCurvesShapes
from numpy import zeros, std, random, newaxis

def runLQRController(xx_traj, uu_traj, KK, discretizedDynamicFunction, xx0Noise = None): 

    xx_traj, uu_traj, ns, ni, TT = correctStateInputCurvesShapes(xx_traj, uu_traj)
    xx_track = zeros((ns, TT))
    uu_track = zeros((ni, TT))
    
    print(xx_traj[:,0].shape)

    xx_track[:,0] = xx_traj[:,0] + (xx0Noise if xx0Noise is not None else 0)

    for tt in range(TT-1): 
        # Evolving the dynamic of the system using the LQR (in a closed loop fashion)
        uu_track[:,tt] = uu_traj[:, tt] + KK[:,:,tt]@(xx_track[:,tt] - xx_traj[:,tt])
        xx_track[:,tt+1] = discretizedDynamicFunction(xx_track[:,tt], uu_track[:,tt])[0]

    return xx_track, uu_track

def generateNoise(xx, noiseStdPercentage): 
    
    # Compute the standard deviation for each state (row-wise)
    stateSD = std(xx, axis = 1)
    # Scale the standard deviation by the percentage
    noiseSD = stateSD*noiseStdPercentage
    # Generate (and return) Gaussian noise with zero mean and scaled (by percentage) S.D. for each state
    return random.randn(xx.shape[0], 1)*noiseSD[:,newaxis]
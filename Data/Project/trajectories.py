
from numpy import exp, linspace, zeros, ones, squeeze

def sigmoidTrajectory(totalDeltaT, evolutionDeltaT, YstartingValue, YendingValue, dt):
    """
    Generation of a sigmoid trajectory between two constant points
    Arguments:
    - totalDeltaT: scalar total time of the trajectory evolution
    - evolutionDeltaT: scalar time of the trajectory evolution in which the evolution itself is not constant (maximum: 20, minimum 2)
    - YstartingValue: starting value assumed from the trajectory
    - YendingValue: ending value assumed from the trajectory
    - dt: scalar time step of the trajectory evolution
    Returns:
    - Y: the desired sigmoid trajectory
    """

    if evolutionDeltaT < 2:
        evolutionDeltaT = 2
    elif evolutionDeltaT > 20:
        evolutionDeltaT = 20
    if totalDeltaT < evolutionDeltaT:
        totalDeltaT = evolutionDeltaT
    
    error = 1
    tolerance = 1e-4
    k = 0.5
    t = linspace(-evolutionDeltaT/2, evolutionDeltaT/2, int(evolutionDeltaT/dt))
    while error > tolerance:
        error = 1/(1+exp(-k*(-evolutionDeltaT/2)))
        k = k + 0.01
    Ysigmoid = zeros((YstartingValue.shape[0], len(t)))
    for ii in range(0, YstartingValue.shape[0]):
        Ysigmoid[ii] = 1/(1 + exp(-k*t))
    Y = zeros((YstartingValue.shape[0], int(totalDeltaT/dt)))
    Y[:,0:int(((totalDeltaT-evolutionDeltaT)/dt/2))] = zeros((YstartingValue.shape[0], int((totalDeltaT-evolutionDeltaT)/dt/2)))
    Y[:,int(((totalDeltaT-evolutionDeltaT)/dt/2)):int(((totalDeltaT-evolutionDeltaT)/dt/2))+len(t)] = Ysigmoid
    Y[:,int(((totalDeltaT-evolutionDeltaT)/dt/2))+len(t):] = ones((YstartingValue.shape[0], int((totalDeltaT-evolutionDeltaT)/dt/2)))
    for ii in range(0, YstartingValue.shape[0]):
        Y[ii, :] = Y[ii, :]*(YendingValue[ii]-YstartingValue[ii]) + YstartingValue[ii]
    return  squeeze(Y), squeeze(linspace(0, totalDeltaT, int(totalDeltaT/dt)))
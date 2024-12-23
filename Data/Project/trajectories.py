
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

def PascalSnail(totalDeltaT, evolutionDeltaT, dt, a = 0.5, b = 1):
    """
    Generation of a complex smooth trajectory between two constant points
    Arguments:
    - totalDeltaT: scalar total time of the trajectory evolution
    - evolutionDeltaT: scalar time of the trajectory evolution in which the evolution itself is not constant (maximum: 20, minimum 2)
    - YstartingValue: starting value assumed from the trajectory
    - YendingValue: ending value assumed from the trajectory
    - dt: scalar time step of the trajectory evolution
    Returns:
    - Y: the desired smooth trajectory
    """
    if evolutionDeltaT < 2:
        evolutionDeltaT = 2
    elif evolutionDeltaT > 20:
        evolutionDeltaT = 20
    if totalDeltaT < evolutionDeltaT:
        totalDeltaT = evolutionDeltaT
    
    dim = int(totalDeltaT/dt)
    dim_snail = int(evolutionDeltaT/dt)
    
    t = linspace(0,totalDeltaT, dim)
    t_snail = linspace(-evolutionDeltaT/2, evolutionDeltaT/2, int(evolutionDeltaT/dt))

    initialPartEnd = int(((totalDeltaT-evolutionDeltaT)/dt/2))
    snailPartEnd = int(((totalDeltaT-evolutionDeltaT)/dt/2))+len(t_snail)

    theta = linspace(0, 2*pi, int(evolutionDeltaT/dt)) # definition of polar independent coordinate theta
    
    # We define the Pascal snail
    r = a + b*sin(theta)

    # We define the change of variable to polar coordinates
    x = r*cos(theta)
    y = r*sin(theta)

    Ysnail = zeros((2, dim)) # Trajectory initialization
    Y = vstack((x,y)) # Pascal snail 

    print(Ysnail.shape)
    initialValue = Y[:,0]
    finalValue = Y[:,dim_snail-1]

    # Definition of initial part with constant initial snail curve value 
    Ysnail[:,0:initialPartEnd] = initialValue[:, newaxis]
    # Definition of transient part: the effective Pascal snail
    Ysnail[:,initialPartEnd:snailPartEnd] = Y
    # Definition of final part with constant final snail curve value
    Ysnail[:,snailPartEnd:] = finalValue[:, newaxis]

    return squeeze(Ysnail), squeeze(t)

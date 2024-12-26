
from numpy import *
import parameters as params

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

def pascalSnailFRAPositionTrajectory(totalDeltaT, evolutionDeltaT, dt):
    """
    Generation of a complex smooth pascal-snail-like trajectory for the Flexible Robotic Arm (starting from and ending to an equilibrium point)
    Arguments:
    - totalDeltaT: scalar total time of the trajectory evolution
    - evolutionDeltaT: scalar time of the trajectory evolution in which the evolution itself is not constant (maximum: 20, minimum 2)
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
    
    TT = int(totalDeltaT/dt)
    # TTSnail = int(evolutionDeltaT/dt)
    t = linspace(0, totalDeltaT, TT)
    tSnail = linspace(-evolutionDeltaT/2, evolutionDeltaT/2, int(evolutionDeltaT/dt))
    snailBegin = int(((totalDeltaT-evolutionDeltaT)/dt/2))
    snailEnd = int(((totalDeltaT-evolutionDeltaT)/dt/2))+len(tSnail)
    theta = linspace(0, 2*pi, int(evolutionDeltaT/dt)) # Definition of polar independent coordinate theta in the range [0, 2pi]
    
    # Pascal snail definition
    a = 0.4
    b = 0.8
    r = a + b*sin(theta)

    # Change of variable to polar coordinates definition
    x = r*cos(theta)
    y = r*sin(theta)
    x = x[::-1]
    y = y[::-1]
    # Rescaling of the Pascal snail in order to make it fully reachable by the two links in their maximum extension
    y = y*0.4

     # Reorganization of the Pascal snail point in order to make it start from a position in which the second link is vertical (alias equilibrium position)
    maximumXIndex = argmax(x)
    maximumXValue = x[maximumXIndex]
    link1StartingAngle = 45/180*pi
    if (maximumXValue > params.r1*sin(link1StartingAngle)): raise ValueError(
        "The Snail Pascal curve is too large for the first link to be properly placed"
    )
    YvalueForMaximumX = y[maximumXIndex]
    snailYoriginDistance = YvalueForMaximumX + params.r2 + sqrt(params.r1**2 - maximumXValue**2)
    # snailXoriginDistance = 0
    x = concatenate((x[maximumXIndex:],x[:maximumXIndex]))
    y = concatenate((y[maximumXIndex:], y[:maximumXIndex]))

    # Computing FRA angles evolution
    distanceFromAbsoluteOrigin = sqrt(x**2 + (snailYoriginDistance-y)**2)
    if max(distanceFromAbsoluteOrigin) > params.r1 + params.r2: raise ValueError(
        "The Snail Pascal curve is too large for the two link to be able to fully reach it"
    )
    angleFromTheVerticalOfEndEffector = atan2(x, snailYoriginDistance-y)
    x1Reduced = arccos((params.r1**2 + distanceFromAbsoluteOrigin**2 - params.r2**2) / (2*params.r1*distanceFromAbsoluteOrigin))
    x1 = angleFromTheVerticalOfEndEffector + x1Reduced
    x2 = - (pi - arccos((params.r1**2 + params.r2**2 - distanceFromAbsoluteOrigin**2) / (2*params.r1*params.r2)))

    # Definition of additional initial and final part with constant initial value
    x1Extended = zeros(TT)
    x2Extended = zeros(TT)
    x1Extended[:snailBegin] = x1[0]
    x2Extended[:snailBegin] = x2[0]
    x1Extended[snailEnd:] = x1[-1]
    x2Extended[snailEnd:] = x2[-1]
    x1Extended[snailBegin:snailEnd] = x1
    x2Extended[snailBegin:snailEnd] = x2

    return squeeze(x1Extended), squeeze(x2Extended), squeeze(t)

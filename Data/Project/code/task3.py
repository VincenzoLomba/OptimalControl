
# Flexible Robotic Arm Task3: after linearizing the dynamics of the FRA around a given trajectory,
# exploiting the LQR algorithm to define the optimal feedback controller to track the said trajectory.

from miscellaneous import correctStateInputCurvesShapes, saveDataOnFile, loadDataFromFile, TrjTrkCntrlData, RegulatorType
from dynamics import computeLocalLinearization
from solver import solveARE, solveLQP
from regulators import runLQRController, generateInitialStateNoise
from dynamics import discretizedDynamicFRA
from datetime import datetime

taskName = "task3"
def task3(xx_des, uu_des, xx_traj, uu_traj, QQ, RR, lazyExecution = False):
    
    if lazyExecution:
        data = loadDataFromFile(taskName)
        if data is not None : return data

    # Computing the linearization of the dynamics around the given trajectory
    xx_des, uu_des, _, _, TT = correctStateInputCurvesShapes(xx_des, uu_des)
    AAlin, BBlin = computeLocalLinearization(xx_traj, uu_traj)

    # Computing the terminal cost matrix as the solution of the ARE (alias Pinfinity)
    QQT = solveARE(AAlin[:,:,-1], BBlin[:,:,-1], QQ, RR, None)

    # Computing the T.V. LQR (Linear Quadratic Regulator), optimal feedback controller
    KK = solveLQP(AAlin, BBlin, QQ, RR, QQT, TT, xx_traj[:,0])[0]

    # Defining some noise levels (in %)(for the initial state) and then running the LQR on the given trajectory
    xx0noiseLevels = [0.0, 0.2, 0.4]
    startComputingTime = datetime.now()
    xx0noises = [generateInitialStateNoise(xx_traj, np) for np in xx0noiseLevels]
    tracks = [runLQRController(xx_traj, uu_traj, KK, discretizedDynamicFRA, xx0noise) for xx0noise in xx0noises]

    data = TrjTrkCntrlData(xx_traj, uu_traj, tracks, xx0noises, RegulatorType.LQR, startComputingTime, datetime.now())
    saveDataOnFile(data, taskName)
    return data

if __name__ == "__main__": task3()

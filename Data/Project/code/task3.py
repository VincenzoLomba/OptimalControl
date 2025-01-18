
# Flexible Robotic Arm Task3: after linearizing the dynamics of the FRA around a given trajectory,
# exploiting the LQR algorithm to define the optimal feedback controller to track the said trajectory.

from miscellaneous import correctStateInputCurvesShapes, saveDataOnFile, loadDataFromFile, TrjTrkCntrlData, RegulatorType
from dynamics import computeLocalLinearization
from solver import solveARE, solveLQP
from regulators import runLQRController, generateInitialStateNoise
from dynamics import discretizedDynamicFRA

taskName = "task3"
def task3(xx_des, uu_des, xx_traj, uu_traj, QQ, RR, lazyExecution = False):
    
    if lazyExecution:
        data = loadDataFromFile(taskName)
        if data is not None : return data

    xx_des, uu_des, _, _, TT = correctStateInputCurvesShapes(xx_des, uu_des)
    AAlin, BBlin = computeLocalLinearization(xx_traj, uu_traj)

    QQT = solveARE(AAlin[:,:,-1], BBlin[:,:,-1], QQ, RR, None)
    KK = solveLQP(AAlin, BBlin, QQ, RR, QQT, TT, xx_traj[:,0])[0]

    noiseLevels = [0.0, 0.5, 1]
    noises = [generateInitialStateNoise(xx_traj, np) for np in noiseLevels]
    tracks = [runLQRController(xx_traj, uu_traj, KK, discretizedDynamicFRA, noise) for noise in noises]

    data = TrjTrkCntrlData(xx_traj, uu_traj, tracks, noises)
    saveDataOnFile(data, taskName, RegulatorType.LQR)
    return data

if __name__ == "__main__": task3()

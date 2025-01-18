
# Flexible Robotic Arm Task4: after linearizing the dynamics of the FRA around a
# given trajectory, exploiting an MPC algorithm to track the said trajectory.

from miscellaneous import correctStateInputCurvesShapes, saveDataOnFile, loadDataFromFile, TrjTrkCntrlData, RegulatorType
from dynamics import computeLocalLinearization
from solver import solveARE
from regulators import runMPController, generateInitialStateNoise
from dynamics import discretizedDynamicFRA

from miscellaneous import getTime

taskName = "task4"
def task4(xx_des, uu_des, xx_traj, uu_traj, QQ, RR, lazyExecution = False):
    
    if lazyExecution:
        data = loadDataFromFile(taskName)
        if data is not None : return data

    xx_des, uu_des, _, _, _ = correctStateInputCurvesShapes(xx_des, uu_des)
    AAlin, BBlin = computeLocalLinearization(xx_traj, uu_traj)
    QQT = solveARE(AAlin[:,:,-1], BBlin[:,:,-1], QQ, RR, None)

    MPC_TT = 5

    noiseLevels = [0.5]
    noises = [generateInitialStateNoise(xx_traj, np) for np in noiseLevels]
    tracks = [runMPController(xx_traj, uu_traj, AAlin, BBlin, QQ, RR, QQ, MPC_TT, discretizedDynamicFRA, noise) for noise in noises]
    for i in range(len(tracks)):
        xx_track, uu_ttrack = tracks[i]
        xx_track = xx_track[:,:-MPC_TT]
        uu_ttrack = uu_ttrack[:,:-MPC_TT]
        tracks[i] = (xx_track, uu_ttrack)
    xx_traj = xx_traj[:,:-MPC_TT]
    uu_traj = uu_traj[:,:-MPC_TT]
    data = TrjTrkCntrlData(xx_traj, uu_traj, tracks, noises, RegulatorType.MPC)
    saveDataOnFile(data, taskName)
    return data

if __name__ == "__main__": task4()
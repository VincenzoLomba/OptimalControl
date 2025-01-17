
# Flexible Robotic Arm Task3: after linearizing the dynamics of the FRA around a given trajectory,
# exploiting the LQR algorithm to define the optimal feedback controller to track the said trajectory.

from miscellaneous import correctStateInputCurvesShapes, saveDataOnFile, loadDataFromFile
from dynamics import computeLocalLinearization
from solver import solveARE, solveLQP
from regulators import runLQRController, generateNoise
from dynamics import discretizedDynamicFRA

taskName = "task3"
def task3(xx_des, uu_des, xx_traj, uu_traj, QQ, RR, lazyExecution = False):
    
    if lazyExecution:
        tracks = loadDataFromFile(taskName)
        if tracks is not None: return tracks

    xx_des, uu_des, _, _, TT = correctStateInputCurvesShapes(xx_des, uu_des)
    AAlin, BBlin = computeLocalLinearization(xx_traj, uu_traj)
    #print(AAlin[:,:,-1])
    #print(BBlin[:,:,-1])
    #print(QQ)
    #print(RR)
    QQT = solveARE(AAlin[:,:,-1], BBlin[:,:,-1], QQ, RR, None)
    KK = solveLQP(AAlin, BBlin, QQ, RR, QQ, TT, xx_traj[:,0])[0]

    noise = generateNoise(xx_traj, noiseStdPercentage = 0.2)
    xx_track, uu_track = runLQRController(xx_traj, uu_traj, KK, discretizedDynamicFRA, xx0Noise = None)
    xx_track_noise, uu_track_noise = runLQRController(xx_traj, uu_traj, KK, discretizedDynamicFRA, xx0Noise = noise)

    tracks = [xx_track, uu_track, xx_track_noise, uu_track_noise]
    saveDataOnFile(tracks, taskName)

    return xx_track, uu_track, xx_track_noise, uu_track_noise

if __name__ == "__main__": task3()

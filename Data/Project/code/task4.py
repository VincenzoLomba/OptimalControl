
# Flexible Robotic Arm Task4: after linearizing the dynamics of the FRA around a
# given trajectory, exploiting an MPC algorithm to track the said trajectory.

from miscellaneous import correctStateInputCurvesShapes, saveDataOnFile, loadDataFromFile, TrjTrkCntrlData, RegulatorType
from dynamics import computeLocalLinearization
from regulators import runMPController, generateInitialStateNoise
from dynamics import discretizedDynamicFRA
from datetime import datetime
from numpy import diag, eye
import logger

taskName = "task4"
def task4(xx_des, uu_des, xx_traj, uu_traj, lazyExecution = False):
    
    if lazyExecution:
        data = loadDataFromFile(taskName)
        if data is not None:
            logger.log("Task4 data loaded from file")
            return data

    logger.log("Computing the linearization of the dynamics around the given trajectory...")
    xx_des, uu_des, _, ni, _ = correctStateInputCurvesShapes(xx_des, uu_des)
    AAlin, BBlin = computeLocalLinearization(xx_traj, uu_traj)

    logger.log("Defining cost matrices")
    QQ = diag([16.0,16.0,1.0,1.0])
    RR = 0.001*eye(ni)

    logger.log("Defining the prediction time horizon for the MPC (in terms of quantity of time instants)")
    MPC_TT = 50

    # Defining some noise levels (in %)(for the initial state) and then running the MPC on the given trajectory
    xx0noiseLevels = [0.0, 0.2, 0.4]
    startComputingTime = datetime.now()
    xx0noises = []
    tracks = []
    for np in xx0noiseLevels:
        logger.log(f'Running the MPC controller tracking the given trajectory (noise~N(0,p) with p={np}% of state S.D.)...')
        xx0noise = generateInitialStateNoise(xx_traj, np)
        xx0noises.append(xx0noise)
        tracks.append(runMPController(xx_traj, uu_traj, AAlin, BBlin, QQ, RR, QQ, MPC_TT, discretizedDynamicFRA, xx0noise))
    for i in range(len(tracks)):
        xx_track, uu_ttrack = tracks[i]
        xx_track = xx_track[:,:-MPC_TT]
        uu_ttrack = uu_ttrack[:,:-MPC_TT]
        tracks[i] = (xx_track, uu_ttrack)
    xx_traj = xx_traj[:,:-MPC_TT]
    uu_traj = uu_traj[:,:-MPC_TT]

    logger.log("Saving results on file and returning them")
    data = TrjTrkCntrlData(xx_traj, uu_traj, tracks, xx0noises, RegulatorType.MPC, startComputingTime, datetime.now())
    saveDataOnFile(data, taskName)
    return data

if __name__ == "__main__": task4()
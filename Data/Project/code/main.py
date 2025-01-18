
# The main file of the project! It contains the code for the execution of all project's tasks!

from miscellaneous import TrjTrkOCPData, TrjTrkCntrlData
from task1 import task1
from task2 import task2
from task3 import task3
from task4 import task4

# Tasks selection (from one to five, for each task, set zero to avoid the
# execution, any other value to instead execute the particualr task)
tasks = [0, 0, 0, 1, 0]

# Lazyness selection (from one to five, for each task, set zero to avoid the laziness, any other
# value to force the single task to load the results data from the last saved file (if it exists)
lazyness = [False, False, False, False, True]

# Correcting the tasks and lazyness lists (adding an empty element at zero index)
tasks = [None] + tasks
lazyness = [None] + lazyness

if (tasks[1]):
    task1Data: TrjTrkOCPData
    task1Data = task1(lazyness[1])
    task1Data.plotStateInputOptimalTrajectory()
    task1Data.plotStateInputOptimalTrajectoryEvolution([0,1,2, task1Data.K])
    task1Data.plotArmijo([3, 4, 7, 8])
    task1Data.plotDescentDirectionNormEvolution()
    task1Data.plotCostEvolution()

if (tasks[2]):
    task2Data: TrjTrkOCPData
    task2Data, QQ, RR = task2(lazyness[2])
    task2Data.plotStateInputOptimalTrajectory()
    task2Data.plotStateInputOptimalTrajectoryEvolution([0,1,2, task2Data.K])
    task2Data.plotArmijo([3, 4, 7, 8])
    task2Data.plotDescentDirectionNormEvolution()
    task2Data.plotCostEvolution()

if (tasks[3]):
    task3Data: TrjTrkCntrlData
    task2Data, QQ, RR = task2(lazyExecution = True)
    xx_traj, uu_traj = task2Data.getOptimalTrajectory()
    task3Data = task3(
        task2Data.xx_des,
        task2Data.uu_des,
        xx_traj, uu_traj, QQ, RR,
        lazyness[3]
    )
    for i in range(task3Data.getTrackLength()): task3Data.plotTrack(i)

if (tasks[4]):
    MPC_NTT = 2
    task4Data: TrjTrkCntrlData
    task2Data, QQ, RR = task2(lazyExecution = True)
    xx_traj, uu_traj = task2Data.getOptimalTrajectory()
    task4Data = task4(
        task2Data.xx_des,
        task2Data.uu_des,
        xx_traj, uu_traj, QQ, RR, MPC_NTT,
        lazyness[4]
    )
    for i in range(task4Data.getTrackLength()): task4Data.plotTrack(i)



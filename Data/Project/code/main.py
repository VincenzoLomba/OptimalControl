
# The main file of the project! It contains the code for the execution of all project's tasks!

from miscellaneous import TrjTrkOCPData, TrjTrkCntrlData, initSavesFolder
from task1 import task1
from task2 import task2
from task3 import task3
from task4 import task4
from task5 import task5

initSavesFolder()

# Tasks selection (from one to five, for each task, set zero to avoid the
# execution, any other value to instead execute the particualr task)
tasks = [0, 0, 1, 0, 0]

# Lazyness selection (from one to five, for each task, set zero to avoid the laziness, any other
# value to force the single task to load the results data from the last saved file (if it exists)
lazyness = [True, True, True, True, None]

# Correcting the tasks and lazyness lists (adding an empty element at zero index)
tasks = [None] + tasks
lazyness = [None] + lazyness

if (tasks[1]):
    task1Data: TrjTrkOCPData
    task1Data = task1(lazyness[1])
    task1Data.plotStateInputOptimalTrajectory()
    task1Data.plotStateInputOptimalTrajectoryEvolution([0,1,2,3,task1Data.K])
    task1Data.plotArmijo([3, 4, 7, 8])
    task1Data.plotDescentDirectionNormEvolution()
    task1Data.plotCostEvolution()

if (tasks[2]):
    task2Data: TrjTrkOCPData
    task2Data, QQ, RR = task2(lazyness[2])
    task2Data.plotStateInputOptimalTrajectory()
    task2Data.plotStateInputOptimalTrajectoryEvolution([0,1,2,3,task2Data.K])
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
    for i in range(task3Data.getTracksLength()): task3Data.plotTrack(i)

if (tasks[4]):
    task4Data: TrjTrkCntrlData
    task2Data, QQ, RR = task2(lazyExecution = True)
    xx_traj, uu_traj = task2Data.getOptimalTrajectory()
    task4Data = task4(
        task2Data.xx_des,
        task2Data.uu_des,
        xx_traj, uu_traj, QQ, RR,
        lazyness[4]
    )
    for i in range(task4Data.getTracksLength()): task4Data.plotTrack(i)

if (tasks[5]): 
    task2Data, QQ, RR = task2(lazyExecution = True)
    xx_traj, uu_traj = task2Data.getOptimalTrajectory()
    task3Data = task3(task2Data.xx_des, task2Data.uu_des, xx_traj, uu_traj, QQ, RR, lazyExecution = True)
    xx_lqr, _ = task3Data.tracks[1]
    task5(xx_traj, xx_lqr, "Reference Path", "Tracked By LQR Path")
    # task4Data = task4(task2Data.xx_des, task2Data.uu_des, xx_traj, uu_traj, QQ, RR, lazyExecution = True)
    # xx_mpc, _ = task4Data.tracks[0]
    # task5(xx_traj, xx_mpc, "Reference Path", "Tracked By MPC Path")

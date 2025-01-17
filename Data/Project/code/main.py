
# The main file of the project! It contains the code for the execution of all project's tasks!

from task1 import task1
from task2 import task2
from task3 import task3

# Tasks selection (from one to five, set zero to avoid the execution, any other value to instead execute the particualr task)
tasks = [0, 0, 1, 0, 0]
tasks = [None] + tasks

# Should selected tasks be executed lazily? If True, the data will be loaded from the last saved file (if it exists)
globalLazyness = True

if (tasks[1]):
    task1Data = task1(globalLazyness)
    task1Data.plotStateInputOptimalTrajectory()
    task1Data.plotStateInputOptimalTrajectoryEvolution([0,1,2, task1Data.K])
    task1Data.plotArmijo([3, 4, 8, 9])
    task1Data.plotDescentDirectionNormEvolution()
    task1Data.plotCostEvolution()

task2Data = None; RRtask2 = None; QQtask2 = None
if (tasks[2]):
    task2Data, QQ, RR = task2(globalLazyness)
    task2Data.plotStateInputOptimalTrajectory()
    task2Data.plotStateInputOptimalTrajectoryEvolution([0,1,2,4, task1Data.K])
    task2Data.plotArmijo([3, 4, 8, 9])
    task2Data.plotDescentDirectionNormEvolution()
    task2Data.plotCostEvolution()

if (tasks[3]):
    task2Data, QQ, RR = task2(lazyExecution = True)
    xx_traj, uu_traj = task2Data.getOptimalTrajectory()
    xx_track, uu_track, xx_track_noise, uu_track_noise = task3(
        task2Data.xx_des,
        task2Data.uu_des,
        xx_traj, uu_traj, QQ, RR,
        False
    )
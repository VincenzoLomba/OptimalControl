# Flexible Robotic Arm Task1: from a desired step trajectory (that evolves from one equilibrium to
# another) to an optimal one thanks to the regularized Newton's like Method (in its closed-loop version)

from parameters import *
from numpy import *
from trajectories import *
from methods import *
from task1 import task1
from task2 import task2

# Tasks selection
tasks = [0, 1, 0, 0]

# Tasks execution
tasks = [None] + tasks
print(tasks)
if (tasks[1]): task1()
if (tasks[2]): task2()

"""

############################
########## TASK 3 ##########
############################

if(task[3]):
    # Computation of local linearization around optimal trajectory and consequently LQR gain KK
    KK = ComputeLocalLin(xx_opt, uu_opt, QQ, RR, QQ, TT, discretizedDynamicFunction, solveLinearLQP)[0]

    # Design of the LQR with noise
    noise = GenerateNoise(xx_opt, noise_std_percentage=0.2)
    xx_track, uu_track = SolveLQPwithNoise(xx_opt, uu_opt, KK, noise, TT, discretizedDynamicFunction, False)
    xx_track, uu_track = SolveLQPwithNoise(xx_opt, uu_opt, KK, noise, TT, discretizedDynamicFunction, True)


############################
########## TASK 4 ##########
############################

if(task[4]): 
    # Computing again local linearization aroun opt. traj. obtained through Newton's method in task 2
    AA_opt, BB_opt = ComputeLocalLin(xx_opt, uu_opt, QQ, RR, QQ, TT, discretizedDynamicFunction, solveLinearLQP)[1:]
"""

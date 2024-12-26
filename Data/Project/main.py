# Flexible Robotic Arm Task1: from a desired step trajectory (that evolves from one equilibrium to
# another) to an optimal one thanks to the regularized Newton's like Method (in its closed-loop version)

from parameters import *
from numpy import *
from trajectories import *
from methods import *
from tasks.task1 import task1
from tasks.task2 import task2
from tasks.task3 import task3

# Tasks selection
tasks = [0, 1, 0, 0]

# Tasks execution
tasks = [None] + tasks
print(tasks)
if (tasks[1]): task1()
if (tasks[2]): task2()
if (tasks[3]): task3()

"""

############################
########## TASK 4 ##########
############################

if(task[4]): 
    # Computing again local linearization aroun opt. traj. obtained through Newton's method in task 2
    AA_opt, BB_opt = ComputeLocalLin(xx_opt, uu_opt, QQ, RR, QQ, TT, discretizedDynamicFunction, solveLinearLQP)[1:]
"""

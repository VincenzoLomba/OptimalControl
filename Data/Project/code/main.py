
# The main file of the project! It contains the code for the execution of all project's tasks!

from task1 import task1
from task2 import task2

# Tasks selection (from one to five, set zero to avoid the execution, any other value to instead execute the particualr task)
tasks = [1, 0, 0, 0, 0]

lazyness = True

# Tasks execution
tasks = [None] + tasks
if (tasks[1]):
    task1Data = task1(lazyness)
    task1Data.plotStateInputCurves()
if (tasks[2]):
    task2()
#if (tasks[3]):
#    task3()
#if (tasks[4]): 
#    task4()
#if (tasks[5]): 
#    task5()

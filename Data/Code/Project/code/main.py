
# The main file of the project! It contains the code for the execution of all project's tasks!

import task1, task2 # , task3, task4, task5

# Tasks selection (from one to five, set zero to avoid the execution, any other value to instead execute the particualr task)
tasks = [1, 0, 0, 0, 0]

# Tasks execution
tasks = [None] + tasks # print(tasks)
if (tasks[1]):
    task1()
if (tasks[2]):
    task2()
#if (tasks[3]):
#    task3()
#if (tasks[4]): 
#    task4()
#if (tasks[5]): 
#    task5()

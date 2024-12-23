# Optimal Control group 15 code
## Launch instructions
This file gives all the information needed to properly run the code for this project. 

All the tasks can be run from the Main.py file. The parameter ```task_set``` is a python list. By setting the element corresponding to the task to 1 the tasks can be executed. 

Each task can be executed individually. If data from previous tasks is needed it will be automatically recovered from the files ```[]_opt_20.py``` and ```[]real_task3_20.py``` for x and u respectively. 

### Task parameters
The following task parameters can be modified:
 - ```show_desired_trajectories```, if set to ```True```, shows the desired input-state curves to track. If set to false, they are calculated but not shown.
 - ```visu_armijo``` makes the code show the armijo's plots during the iterations in the Newton's method
 - ```save trajectories``` makes the code save the computed trajectories for task 2 and 3 
 - ```use_multiple_initial conditions``` forces the computation for the disturbed trajectories for tasks 3 and 4 with noise on each state component individually. If set to ```False```, it will just show a single disturbed trajectory with noise on all the components. 

### Trajectory parameters
The following trajectory parameters can also be tuned.
- ```tf``` final time in seconds. Modifying this parameter can make the animation look slightly worse.
- ```max_iters``` maximum iterations for newton's algorithm. They shouldn't in principle be reached, but it is a safety value that stops the software from running in case of misbehaviours. 

## Code structure
As mentioned before, the ```Main.py``` function handles the overall workflow of the code. 
- The file ```Cost.py``` contains the stage and terminal costs for the cost function.
- The file ```Dynamics.py``` contains all the necessary information on the dynamics of the system.
- The file ```utils.py``` contains all the useful subroutines we implemented to make the code more readable and modular. 
- The file ```plots.py``` contains the functions used to make the plots and the animation.

A sincere thanks to Dr. Lorenzo Sforni for giving us the code for some routines so that we didn't need to reimplement it, and to Dr. Marco Borghesi for his great contribution to our code and for his time. 
This file is just a report to give an insight on the code structure and how to run it. All of the information about the tasks and the design of the code can be found in the attached report.
 

P.S. do NOT click, enlarge or hover on the animation, it can crash
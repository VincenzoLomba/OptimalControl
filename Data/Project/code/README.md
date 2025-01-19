# Optimal Control of a Flexible Robotic Arm

The current folder contains all the code related to the development of optimal control for a Flexible Robotic Arm, created following the guidelines in the file [_Assignment OptimalControl FlexibleRoboticArm_][1]. For a more detailed explanation of the various methods contained in the files, refer to their content and the comments and descriptions included within them.

### Main File Usage

The file _main.py_ is the most important file for running the code and observing its results.<br>
The **tasks list** at the beginning of the file can and must be used to select which tasks to execute and display in their results through appropriate plots.
The **lazyness list** at the beginning of the file can be used to indicate, for each executed task, whether it should attempt to retrieve the most recent save (if available) and load its data, or perform a new and clean execution of the code.<br>
Afterwards, the file _main.py_ proceeds to execute and manage the requested tasks and display their results.

### Other files brief description

Here is a descriptive list of the other files that make up the project:
- **task1**: from a desired quasi-step state-input curve (that evolves from one equilibrium to another) to an optimal trajectory thanks to the regularized Newton's Like Method (in its closed-loop version)
- **task2**: from a desired smooth state-input curve (that evolves from one equilibrium to another) to an optimal trajectory thanks to the regularized Newton's Like Method (in its closed-loop version)
- **task3**: after linearizing the dynamics of the FRA around a given trajectory, exploiting the LQR algorithm to define the optimal feedback controller to track the said trajectory
- **task4**: after linearizing the dynamics of the FRA around a given trajectory, exploiting an MPC algorithm to track the said trajectory
- **task5**: producing an animation of the robot executing Task 3
- **solver**: solver for an Optimal Control Trajectory Tracking Problem (and ALL the involved functions, as the N.M., or the Armijo StepSize selection rule, or the solver of an LQP, and so on)
- **dynamics**: this file contains the Flexible Robotic Arm discretized dynamics and some related methods, such as a method to evolve forward-in-time (in an open loop fashon) a dynamic system or also a method to compute a local linearization of a NL dynamic around a given trajectory
- **costs**: definition of the cost functions for a Trajectory Tracking Optimal Control Problem
- **curves**: curves generator for states and inputs (in a fashion that a given series of points are properly connected)
- **equilibria**: Flexible Robotic Arm Equilibrium Points Searcher
- **regulators**: collection of the functions that implement the Regulators used in the Project (and correlated functions), alias LQR and MPC
- **miscellaneous**: this file contains some useful functions (and the classes TrjTrkOCPData and TrjTrkCntrlData, the first one used in Tasks 1 and 2 and the second one used in Task 3 and 4 to encapsulate generated data) that are used in the project
- **logger**: project logger, used in the whole project to print, alias log to the console, information about the execution
- **parameters**: this file contains all the parameters of the Flexible Robotic Arm (FRA) system, as well as other parameters used in the code
- **plots**: project plotter, used for all plots generation
- **animation**: project animator, to implement the Task5

[1]: https://github.com/VincenzoLomba/OptimalControl/blob/master/Data/Project/Assignment%20OptimalControl%20FlexibleRoboticArm.pdf

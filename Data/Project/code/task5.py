
# Flexible Robotic Arm Task5: produce a simple animation of the robot executing Task 3

from animation import animateFRA

taskName = "task5"
def task5(xx_ref, xx_lqr, lab1, lab2): return animateFRA(xx_ref, xx_lqr, lab1, lab2)

if __name__ == "__main__": task5()
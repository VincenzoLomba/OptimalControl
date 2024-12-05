from types import SimpleNamespace

ns = 4      # number of states
ni = 1      # number of inputs
m1 = 1.5    # mass of the first link
m2 = 1.5    # mass of the second link 
l1 = 2      # length of the first link
l2 = 2      # length of the second link
r1 = 1      # distance from the pivot point of the first link to its center of mass
r2 = 1      # distance from the pivot point of the second link to its center of mass
I1 = 2      # inertia of the first link
I2 = 2      # inertia of the second link
g = 9.81    # gravity
f1 = 0.1    # friction associated to the pivot of the first link
f2 = 0.1    # friction associated to the pivot of the second link

dtCollection = SimpleNamespace(
    task0_discretizationStep = 1e-3
)

TCollection = SimpleNamespace(
    task1_trajectoryDuration = 10 # The duration of the trajectory (in seconds)
)

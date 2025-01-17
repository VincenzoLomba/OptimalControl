
# Flexible Robotic Arm Task2: from a desired smooth state-input curve (that evolves from one equilibrium to
# another) to an optimal trajectory thanks to the regularized Newton's Like Method (in its closed-loop version)

from parameters import ns, ni
from numpy import pi, array, column_stack, eye
from equilibria import searchFRAInputGivenAnEquilibria
from curves import generateCurves, CurveType
from dynamics import discretizedDynamicFRA as discretizedDynamicFuntion
from methods import runNewtonMethodTrkTrj
from miscellaneous import saveDataOnFile

def task2():
    return None
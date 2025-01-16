
# Some useful functions that are used in the project

import os, parameters as params
from numpy import save, load

def getTimeDifferenceAsString(endingTime, startingTime):
    timeDifference = endingTime - startingTime
    hours, remainder = divmod(timeDifference.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds"

def saveDataOnFile(data, filename):
    if not os.path.exists(params.savesFolder): os.makedirs(params.savesFolder)
    save(os.path.join(params.savesFolder, filename), data)

def loadDataFromFile(filename):
    return load(os.path.join(params.savesFolder, filename + '.npy'))

def correctStateInputCurvesShapes(xx, uu):
    if (uu.ndim == 1):
        ni = 1
        uu = uu.reshape(ni, uu.shape[0])
    else: ni = uu.shape[0]
    if (xx.ndim == 1):
        ns = 1
        xx = xx.reshape(ns, xx.shape[0])
    else: ns = xx.shape[0]
    return xx, uu, ns, ni


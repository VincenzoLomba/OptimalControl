import os, json, parameters as params
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
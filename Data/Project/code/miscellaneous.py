
# Some useful functions (and a class) that are used in the project

import os, parameters as params
from numpy import linspace, zeros, zeros_like
import joblib
from parameters import discretizationStep as dt
from datetime import datetime
import plots as plts

def getTimeDifferenceAsString(endingTime, startingTime):
    timeDifference = endingTime - startingTime
    hours, remainder = divmod(timeDifference.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds"

def saveDataOnFile(data, filename):
    
    if not os.path.exists(params.savesFolder): os.makedirs(params.savesFolder)
    fullFileName = filename + "_" + datetime.now().strftime(params.dateFormat) + ".pkl"
    joblib.dump(data, os.path.join(params.savesFolder, fullFileName))

def loadDataFromFile(filename):

    if not os.path.exists(params.savesFolder): return None
    files = []
    for file in os.listdir(params.savesFolder):
        if file.startswith(filename):
            try:
                date = file[len(filename) + 1 : file.rfind('.')]
                dateStr = datetime.strptime(date, params.dateFormat)
                files.append((file, dateStr))
            except ValueError:
                continue
    if len(files) == 0: return None
    fullFileName = max(files, key=lambda x: x[1])[0]

    return joblib.load(os.path.join(params.savesFolder, fullFileName))

def representDate(value):
    try:
        datetime.fromisoformat(value); return True
    except (ValueError, TypeError): return False

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

class TrjTrkOCPData:

    def __init__(self, ns, ni, xx_des, uu_des, TT, maxIterations):
        self.startingTime = datetime.now()
        self.endingTime = None
        self.ns = ns
        self.ni = ni
        self.TT = TT
        self.T = dt*TT
        self.t = linspace(0, self.T, TT)
        self.xx_des = xx_des
        self.uu_des = uu_des
        self.xxCollection = zeros((ns, TT, maxIterations))
        self.uuCollection = zeros((ni, TT, maxIterations))
        self.grdJJCollection = zeros_like(self.uuCollection)
        self.armijoStepsizesCollection =  [[] for _ in range(maxIterations)]
        self.armijoCostsCollection =  [[] for _ in range(maxIterations)]
        self.K = None

    def setEndingTime(self):
        self.endingTime = datetime.now()
    def getElapsedTime(self):
        if self.endingTime == None: self.setEndingTime()
        return getTimeDifferenceAsString(self.endingTime, self.startingTime)
    
    def getOptimalTrajectory(self):
        return self.xxCollection[:,:,self.K], self.uuCollection[:,:,self.K]
    def getOptimalCostGradient(self):
        return self.grdJJCollection[:,:,self.K]
    def getOptimalTrajectoryErrorsAtFinalTime(self):
        x,u = self.getOptimalTrajectory()
        xd = self.xx_des
        ud = self.uu_des
        return x[:,-1]-xd[:,-1], u[:,-2]-ud[:,-2]
    
    def plotStateInputOptimalTrajectory(self):
        xx_opt, uu_opt = self.getOptimalTrajectory()
        return plts.plotStateInputCurves(self.xx_des, self.uu_des, xx_opt, uu_opt, 'desired', 'optimal', dt)
    def plotStateInputOptimalTrajectoryEvolution(self, indexesCollection = None):
        indexesCollection = self.generateCleanedIndexCollection(indexesCollection)
        xxCollectionCast = self.xxCollection[:,:,indexesCollection]
        uuCollectionCast = self.uuCollection[:,:,indexesCollection]
        return plts.plotStateInputCurvesEvolution(self.xx_des, self.uu_des, xxCollectionCast, uuCollectionCast, 'desired', 'optimal', dt, indexesCollection)
    def plotStateGradientEvolution(self, upToIndex = -1, recoverFromIndex = -1):
        indexesCollection = self.generateCleanIndexCollection(indexesCollection)
        upToIndex, recoverFromIndex, amount = self.castCollectionsIndexes(upToIndex, recoverFromIndex)
        grdJJCollectionCast = zeros_like((self.ni, self.TT, amount))

    def generateCleanedIndexCollection(self, dirtyIndexCollection):
        if not dirtyIndexCollection: return list(range(self.K+1))
        indexesCollection = [int(i) for i in dirtyIndexCollection]
        indexesCollection.sort()
        maxAmount = self.K+1
        if len(indexesCollection) > maxAmount: indexesCollection = indexesCollection[:maxAmount]
        if indexesCollection[0] != 0: indexesCollection[0] = 0
        if indexesCollection[-1] != self.K: indexesCollection[-1] = self.K
        return indexesCollection






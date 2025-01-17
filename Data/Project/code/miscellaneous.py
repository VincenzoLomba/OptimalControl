
# Some useful functions (and a class) that are used in the project

import os, parameters as params
from numpy import linspace, zeros, zeros_like
import joblib
from parameters import discretizationStep as dt
from datetime import datetime
from plots import plotStateInputCurves

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
        self.t = linspace(0, dt*TT, TT)
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
    def plotStateInputCurves(self):
        xx_opt, uu_opt = self.getOptimalTrajectory()
        return plotStateInputCurves(self.xx_des, self.uu_des, xx_opt, uu_opt, 'desired', 'optimal', dt)


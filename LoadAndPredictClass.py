# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:03:37 2024

@author: i_bab
"""
import cv2 as cv
import os

import numpy as np
from helper import adjust_number
from augment_utils import normListBy
from tensorflow.train import Checkpoint
from metrics import act_from_pred, refine_thresh
from metrics import get_contours, find_cent_on_im, draw_cent_on_im_v2

class ModelLoad:
    def __init__(self, iModel, iOptimizer=None):
        self.mModel= iModel
        self.mOptimizer = iOptimizer
        self.mPredDict = {}
    
    def getModel(self):
        return self.mModel
    
    def getOptimizer(self):
        return self.mOptimizer
    
    def setLoadDir(self, iLoadDir):
        self.mLoadDir = iLoadDir
    
    def getLoadDir(self):
        return self.mLoadDir

    def setDataDir(self, iDataDir):
        self.mDataDir = iDataDir
    
    def getDataDir(self):
        return self.mDataDir
    
    def loadData(self):
        self.mNameList=[]
        wDataDir = self.getDataDir()
        wDataList = []
        for wFile in os.listdir(wDataDir):
            self.mNameList.append(wFile)
            wIm=cv.imread(os.path.join(wDataDir,wFile), flags=cv.IMREAD_COLOR)
            wDataList.append(wIm)
        self.mData = wDataList
    
    def getNameList(self):
        return self.mNameList
    def getData(self):
        return self.mData
    
    def getDataAtIdx(self, iIdx):
        return self.getData()[iIdx]
    
    def loadFromCkpt(self, iLoadFile):
        wLoadPath = os.path.join(self.getLoadDir(), iLoadFile)
        wLoader = Checkpoint(model = self.getModel())#, optimizer= self.getOptimizer())
        wLoader.read(wLoadPath).expect_partial()
        self.getModel().trainable = False
        
    def setNorm(self, iNorm = 0):
        self.mNorm = iNorm
    
    def getNorm(self):
        return self.mNorm
        
    def setImageProcess(self, iImageProcess = None):
        #iImageProcess customizable function
        self.mImageProcess = iImageProcess

        
    def getImageProcess(self):
        return self.mImageProcess
    
    def processImages(self, ioImList):
        wNorm = self.getNorm()
        if wNorm:
            ioImList = normListBy(ioImList, wNorm)
        wImageProcess = self.getImageProcess()
        ioImList = wImageProcess(
            np.array(ioImList, dtype=np.float32)[..., ::-1])     
        return ioImList
    
    
    def predict(self):
        wPredDict={}
        iImArray = self.processImages(self.getData())
        wPredList= self.getModel()(iImArray, training=False)
        wNameList = self.getNameList()
        wType='pred'
        self.mResolutions = [i for i in range(len(wPredList))]
        for wRes, wPredAtRes in zip(self.mResolutions, wPredList):
            for wName, wPred in zip(wNameList, wPredAtRes):
                
                if wName not in wPredDict.keys():
                    wPredDict.update({wName:{}})
       
                if wType not in wPredDict[wName].keys():
                    wPredDict[wName].update({wType:{}})
                wPredDict[wName][wType].update({wRes: wPred.numpy()[...,0, None]})
        
        self.mPredDict = wPredDict
        
                
    def getResolutions(self):
        return self.mResolutions
    
    def setSaveDir(self, iSaveDir):
        self.mSaveDir = iSaveDir
    
    def getSaveDir(self):
        return self.mSaveDir            
    
    def getDict(self):
        return self.mPredDict
    
    def getNameAtIndex(self, iIdx):
        return self.getNameList()[iIdx]
    
    def getPredListAtIndex(self, iIdx, iType='pred'):
        wDict = self.getDict()[self.getNameAtIndex(iIdx)][iType]
        return [wDict[wRes] for wRes in wDict.keys()]
    
    def getPredAtIndexAtRes(self, iIdx, iRes, iType='pred'):
        if iRes in self.getResolutions():
            return self.getPredListAtIndex(iIdx, iType)[iRes]
        else:
            return
    
    def thresholdPreds(self, iThresh): 
        wType ='thresh'
        wPredDict = self.getDict()
        for wRes in self.getResolutions():
            for wName in self.getNameList():
                wDict = wPredDict[wName]
                
                if wType not in wDict.keys():
                    wDict.update({wType: {}})
                wPred = wPredDict[wName]['pred'][wRes]
                wThresh = cv.threshold(wPred.copy(), iThresh, 1.0, cv.THRESH_TOZERO)[1][...,None]
                wThresh = refine_thresh(wPred.copy(), wThresh)
                wDict[wType].update({wRes: wThresh})

    def extractPredActivations(self):
        wType='act'
        wPredDict = self.getDict()
        for wRes in self.getResolutions():
            for wName in self.getNameList():
                wDict = wPredDict[wName]
                if wType not in wDict.keys():
                    wDict.update({wType:{}})
                wThresh = wPredDict[wName]['thresh'][wRes]
                wPredAct = act_from_pred(wThresh)
                wDict[wType].update({wRes: wPredAct})

    
    def computePredCentroids(self):
        wType='cent'
        wPredDict = self.getDict()
        for wRes in self.getResolutions():
            for wName in self.getNameList():
                wDict = wPredDict[wName]
                if wType not in wDict.keys():
                    wDict.update({wType:{}})
                # wPred = wPredDict[wName]['pred'][wRes]
                wThresh = wPredDict[wName]['thresh'][wRes]
                wPredCntrs = get_contours(wThresh)
                wCentersOnIm = find_cent_on_im(wPredCntrs)
                wPredCents = draw_cent_on_im_v2(wCentersOnIm, None, wThresh.shape)
                wDict[wType].update({wRes: wPredCents})


    def savePlots(self, iSize):
        wNameList = self.getNameList()
        for wIdx, wName in zip(range(len(wNameList)), wNameList):
            wTypeList = ["pred", "thresh", "act", "cent"]
            for wNo, wType in zip(range(len(wTypeList)), wTypeList):
                for wRes in self.getResolutions():
                    wPred = self.getPredAtIndexAtRes(wIdx, wRes, wType)
                    wInterpolation = cv.INTER_NEAREST
                    wSaveDir = os.path.join(self.getSaveDir(), "%s_%s"%(adjust_number(wNo, 2), wType), 'resolution_%s'%(adjust_number(wRes, 2)))
                    os.makedirs(wSaveDir, exist_ok=True)
                    wIm = wPred[...,0]
                    wIm = np.uint8(np.clip(wIm*255.0, 0, 255))
                    cv.imwrite(os.path.join(wSaveDir, wName), cv.resize(wIm, iSize, interpolation=wInterpolation))
        
            
if __name__ == '__main__':
    pass









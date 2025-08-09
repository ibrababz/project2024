#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:33:32 2025

@author: ibabi
"""

import tensorflow as tf
from dataLoad import show_batch
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import Callback
import numpy as np
import os
import time
import gc



class ModelFunctions():
    def __init__(self, iDecoderStartIdx):
        self.mLayerStateDict = {}
        self.mLossLvlSched = {}
        self.mLRSched = {}
        self.mDecoderStartIdx = iDecoderStartIdx
        self.mDecoderName = 'Decoder'
        
    def getModel(self):
        return self.model
    
    def getBackBone(self):
        return self.getModel().layers[:self.mDecoderStartIdx]
 
    def getDecoder(self):
        return self.getModel()
    
    def getDecoderName(self):
        return self.mDecoderName
    
    def getDecoderLayers(self):
        return self.getModel().layers[self.mDecoderStartIdx:]
    
    def setDecoderResolutionsDict(self):
        self.mResolutions={} 
        for wLayer in self.getDecoderLayers():
            wOutputHW = tuple(wLayer.output.shape[1:3])
            if wOutputHW not in self.mResolutions.keys():
                self.mResolutions.update({wOutputHW: []})
            self.mResolutions[wOutputHW].append(wLayer.name)
            
    def setGradualUnfreeze(self, iStart, iRate, iLayerNames, iDecoderFlag = False):
        ## iRate = noLayersToUnfreeze/noEpochsToUnfreeze
        wLayersToUnfreeze = []
        wCounter0 = iStart
        wCounter = wCounter0
        while True:
            wDelta = wCounter - wCounter0
            wNoToUnfreeze = iRate*wDelta         
            if wNoToUnfreeze - np.floor(wNoToUnfreeze) <= 1e-7: 
                for i in range(int(wNoToUnfreeze)):
                    if len(iLayerNames) > 0:
                        wLayersToUnfreeze.append(iLayerNames.pop(0))
                wFreezeDict = {}
                wFreezeDict.update(self.genLayerStateDict(wLayersToUnfreeze, [True]*len(wLayersToUnfreeze)))
                if iDecoderFlag:
                    wFreezeDict = {self.getDecoderName(): wFreezeDict}
                self.setLayerFreezeScheduleByName(wCounter0, wFreezeDict)
                wLayersToUnfreeze = []
                wCounter0 = wCounter
                if len(iLayerNames) == 0:
                    # print("breaking at epoch: %s"%wCounter)
                    break                
            wCounter +=1
    
    def checkLayerStates(self, iEpoch):
        wKey = str(iEpoch)
        if wKey in self.mLayerStateDict.keys():
            wDict = self.mLayerStateDict[wKey]
            wModel = self.getModel()
            wDecoderName = self.getDecoderName()
            if wDecoderName in wDict.keys():
                wDecoder = self.getDecoder()
                wDecoderDict = wDict[wDecoderName]
                for wDecoderLayer in wDecoderDict.keys():
                    wDecoder.get_layer(wDecoderLayer).trainable = wDecoderDict[wDecoderLayer]
            for wLayer in wDict.keys():
                if wLayer != wDecoderName:
                    wModel.get_layer(wLayer).trainable = wDict[wLayer]
            print("Updated States:")
            self.printCurrentStates()
            
    def setLayerFreezeScheduleByName(self, iEpoch, iLayerStateDict):
        wKey = str(iEpoch)
        wDict = self.getLayerStateDict()
        if wKey in wDict.keys():
            wDict[wKey].update(iLayerStateDict)
        else:
            wDict.update({str(iEpoch): iLayerStateDict})
            
    def genLayerStateDict(self, iLayerNames, iLayerStates):
        wDict = {}
        for wName, wState in zip(iLayerNames, iLayerStates):
            wDict.update({wName: wState})
        return wDict
    
    def getLayerStateDict(self):
        return self.mLayerStateDict

    def printCurrentStates(self):
        wHeader = f"{'Name':^10} {'Trainable':^12}"
        print(wHeader)
        print('-'*len(wHeader))
        print(f"{'Backbone':^{len(wHeader)}}")
        wPrevName =  ''
        for wLayer in self.getBackBone():
            wCurrentName = wLayer.name.split('_')[0]
            if wCurrentName != wPrevName:
                print(f"{wLayer.name:10.10s}|{wLayer.trainable:12}")
                print(f"{'...':^{len(wHeader)}}")
            wPrevName = wCurrentName
            
        print(f"{'Decoder':^{len(wHeader)}}")
        for wLayer in self.getDecoderLayers():
            if wLayer.name.split('_')[0] not in ['dropout', 'concatenate', 'softmax']:
                print(f"{wLayer.name:10.8s}|{wLayer.trainable:12}")
                
    def setLossLvlScheduleFromFlagList(self, iEpochList, iFlagList):
        for wEpoch, wFlag in zip(iEpochList, iFlagList):
            self.setLossLvlScheduleFromFlag(wEpoch, wFlag)
    
    def setLossLvlScheduleFromFlag(self, iEpoch, iFlag):
        if abs(iFlag)>=len(self.getDecoderResolutions()):
            wResList=[]        
        elif iFlag>=0:
            wResList= [iFlag]
        elif iFlag<0:
            wResList = list(range(abs(iFlag)+1))
    
        self.setLossLvlScheduleByRes(iEpoch, wResList)
    
    def setLossLvlScheduleByRes(self, iEpoch, iResList):
        wNoRes = len(self.getDecoderResolutions())
        wLossLvl = [0]*wNoRes
        for i in iResList:
            wLossLvl[i] = 1
        self.setLossLvlSchedule(iEpoch, wLossLvl)
        
    def setLossLvlSchedule(self, iEpoch, iLossLvl):
        self.mLossLvlSched.update({str(iEpoch):iLossLvl})
        self.setLayerStatesFromLossLvlSched(iEpoch, iLossLvl)
        
    def setLayerStatesFromLossLvlSched(self, iEpoch, iLossLvl):
        wDecoderDict = {}
        for wLvl, wRes in zip(iLossLvl, self.getDecoderResolutions()):
            wLayerNames = self.getDecoderResolutionsDict()[wRes]
            wDecoderDict.update(self.genLayerStateDict(wLayerNames, [bool(wLvl)]*len(wLayerNames)))
        wLayerStateDict = {self.getDecoderName():wDecoderDict}
        self.setLayerFreezeScheduleByName(iEpoch, wLayerStateDict)  
            
    def getDecoderResolutionsDict(self):
        return self.mResolutions
    
    def getDecoderResolutions(self):
        return list(self.getDecoderResolutionsDict().keys())
    
    def checkLossLvlSchedule(self, iEpoch, iStart=0):
        wKey = str(iEpoch)
        if wKey in self.mLossLvlSched.keys():
            wLossLvl = self.mLossLvlSched[wKey]
            self.setLossLvl(wLossLvl)
            # if self.getEpoch()>=iStart:
            #     self.resetLossWatchDog()
    
    def setLossLvl(self, iLossLvl):
        #can be [1, 0], [0, 1] or [1, 1]
        self.mLossLvl = iLossLvl
        self.nLvls = len(iLossLvl)
         
    def checkLRSched(self, iEpoch):
        wKey = str(iEpoch)
        if wKey in self.mLRSched.keys():
            wLR = self.mLRSched[wKey]
            tf.keras.backend.set_value(self.getModel().optimizer.learning_rate, wLR)
            print("Learning rate set to: %s"%wLR)
            
    def setLRSched(self, iEpoch, iLR):
        self.mLRSched.update({str(iEpoch): iLR})
        

class CustomLRSchedulerFunction:
    def __init__(self, iScheduleTupleList=[], iSaveLasNCallBack=None):
        self.mScheduleTupleList = iScheduleTupleList
        self.mSaveLastNCallBack = iSaveLasNCallBack
        
    def getSchedule(self):
        return self.mScheduleTupleList
        
    def setLRScheduleFromLists(self, iEpochList, iLRList):
        self.mScheduleTupleList = [(wEpoch, wLR) for wEpoch, wLR in zip(iEpochList, iLRList)]
    
    def setLRScheduleFromDict(self, iEpochLRDict):
        self.mScheduleTupleList = [(wEpoch, wLR) for wEpoch, wLR in iEpochLRDict.items()]
    
    def setLRSchedule(self, iScheduleTupleList):
        self.mScheduleTupleList = iScheduleTupleList
        
    def setSaveLastNCallBack(self, iSaveLasNCallBack):
        self.mSaveLastNCallBack = iSaveLasNCallBack
        
    def __call__(self, iEpoch, iLR):
        wList= self.getSchedule()
        if iEpoch < wList[0][0] or  iEpoch > wList[-1][0]:
            return iLR
        for i in range(len(wList)):
            if iEpoch == wList[i][0]:
                if self.mSaveLastNCallBack is not None:
                    self.mSaveLastNCallBack.resetMinValLossList([float('inf')], ['inf'])
                print(f"\nLearning Rate set to: {wList[i][1]:.2e}")
                return wList[i][1]
        return iLR
            
        
                
class LayerUnfreezeCallback(Callback, ModelFunctions):
    def __init__(self, iDecoderStartIdx, *args, **kwargs):
        Callback.__init__(self, *args, **kwargs)
        ModelFunctions.__init__(self, iDecoderStartIdx)
    
    def setGradualUnfreezeArgs(self, iArgs):
        self.mGradUnfreezeArgs=iArgs
        
    def getGradualUnfreezeArgs(self):
        return self.mGradUnfreezeArgs
    
    def  setLossLvlScheduleFromFlagListArgs(self, iArgs):
        self.mLossLvlScheduleFromFlagListArgs = iArgs
    
    def getLossLvlScheduleFromFlagListArgs(self):
        return self.mLossLvlScheduleFromFlagListArgs
           
    def on_train_begin(self, logs=None):
        self.setGradualUnfreeze(*self.getGradualUnfreezeArgs())
        self.setDecoderResolutionsDict()
        self.setLossLvlScheduleFromFlagList(*self.getLossLvlScheduleFromFlagListArgs())
        
    def on_epoch_begin(self, epoch, logs=None):
        self.checkLayerStates(epoch) 
        self.checkLossLvlSchedule(epoch)
        # self.checkLRSched(epoch)
        

class SleepCallBack(Callback):
    def __init__(self, iSleepTime=1.,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mSleepTime = iSleepTime
        
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        time.sleep(self.mSleepTime)
        
        
class PlotCallBack(Callback):
    def __init__(self, iTFDataSegment, iFreq=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mDataBatch = iTFDataSegment
        self.mFreq = iFreq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%self.mFreq ==0 or epoch==0:
            for wX in self.mDataBatch:
                pass
            wIm, wTruth= wX
            # show_batch(tf.cast(wIm, dtype=tf.uint8))
            show_batch(wIm)
            wOutput = self.model.predict(preprocess_input(wIm[...,::-1]))
            for i in range(len(wTruth)):
                show_batch(wOutput[i])
                show_batch(wTruth[i])
                show_batch(tf.where(wTruth[i]<1., 0., 1.))
            # print(logs['val_loss'])

class SaveEveryN(Callback):
    def __init__(self, iCheckpoint, iSaveDir, iFreq=25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mCheckpoint = iCheckpoint
        self.mFreq = iFreq
        self.mSaveDir = iSaveDir
        
    def getSaveDir(self):
        return self.mSaveDir

    def on_epoch_end(self, epoch, logs=None):
        wSaveDir = self.getSaveDir()
        if (epoch+1)%self.mFreq == 0 and epoch>0:
            wName= f'cp-{epoch:04d}'
            self.mCheckpoint.write(os.path.join(wSaveDir,wName))
            
class LogEveryN(Callback):
    def __init__(self, iFileName, iSaveDir, iFreq=25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mFileName = iFileName
        self.mFreq = iFreq
        self.mSaveDir = iSaveDir
        self.mBuffer = []
        self.mFile = open(os.path.join(self.mSaveDir, self.mFileName), 'w')
        
    def getSaveDir(self):
        return self.mSaveDir
    
    def getFileName(self):
        return self.mFileName
    
    def getFrequency(self):
        return self.mFreq

    def on_epoch_end(self, epoch, logs=None):
        wFreq = self.getFrequency()
        self.mBuffer.append(','.join([f"{epoch}"]+[f"{logs[wKey]:.3f}" for wKey in logs if wKey != 'learning_rate']+[f"{logs['learning_rate']:.2e}".replace('e-0','e-')]))
        if (epoch+1)%wFreq==0:
            if epoch+1 == wFreq:
                self.mBuffer = [','.join(['epoch']+list(logs.keys()))] + self.mBuffer
            with open(os.path.join(self.mSaveDir, self.mFileName), 'a') as wFile:
                wFile.write('\n'.join(self.mBuffer)+'\n')
                self.mBuffer=[]

class StopTraining(Callback):
    def __init__(self, iMinEpochs, iMaxTrainLoss, iMaxValLoss, *args, iPatience=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.mMinEpochs = iMinEpochs
        self.mMaxTrainLoss = iMaxTrainLoss
        self.mMaxValLoss = iMaxValLoss
        self.mPatience = iPatience
        self.mTrainCounter = 0
        self.mValCounter = 0
        
    def getMinEpochs(self):
        return self.mMinEpochs
    
    def getMaxTrainLoss(self):
        return self.mMaxTrainLoss
    
    def getMaxValLoss(self):
        return self.mMaxValLoss
    
    def getPatience(self):
        return self.mPatience
    
    def getTrainCounter(self):
        return self.mTrainCounter
    
    def incrementTrainCounter(self):
        self.mTrainCounter+=1
        
    def incrementValCounter(self):
        self.mValCounter+=1
        
    def resetTrainCounter(self):
        self.mTrainCounter = 0
        
    def resetValCounter(self):
        self.mValCounter = 0
    
    def getValCounter(self):
        return self.mValCounter
            
    def on_epoch_end(self, epoch, logs=None):

        if epoch+1 >= self.getMinEpochs():
            
            if logs['loss'] >= self.getMaxTrainLoss():
                self.incrementTrainCounter()
            else:
                self.resetTrainCounter()
                
            if logs['val_loss'] >= self.getMaxValLoss():
                self.incrementValCounter()
            else:
                self.resetValCounter()
            
            wPatience = self.getPatience()
                
            if self.getTrainCounter() >= wPatience or self.getValCounter() >= wPatience:
                print("\nStopping training due to exploding loss(es)")
                self.model.stop_training = True
            
class SaveLastN(Callback):
    def __init__(self, iCheckpoint, iSaveDir, iLastN=5, iPeriodicReset=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mCheckpoint = iCheckpoint
        self.mLastN = iLastN
        self.mSaveDir = iSaveDir
        self.mPeriodicReset = iPeriodicReset
        
    def on_epoch_end(self, epoch, logs=None):
        wPeriodicReset = self.getPeriodicReset()
        wValidLossPerEpoch = logs["val_loss"]
        if wPeriodicReset and (epoch+1)%wPeriodicReset==0:
            self.resetMinValLossList([float('inf')], ['inf'])
        self.addRemoveSavedModels(epoch, wValidLossPerEpoch)

    
    def addRemoveSavedModels(self, iEpoch, iValidLossPerEpoch):
        wSaveDir = self.getSaveDir()
        wMinValLossList, wMinValLossNameList = self.getMinValLossList()

        wMaxIndexesSorted = np.flip(np.argsort(wMinValLossList))
        wMaxIdx = wMaxIndexesSorted[0]
        wMaxMinValLoss = wMinValLossList[wMaxIdx]
        
        if iValidLossPerEpoch <= wMaxMinValLoss:
            wMinValLossList.append(iValidLossPerEpoch)
            wSavePath=f"cp-{iEpoch:04d}_list_{iValidLossPerEpoch:.3f}".replace('.', 'p')
            wMinValLossNameList.append(wSavePath)
            self.mCheckpoint.write(os.path.join(wSaveDir, wSavePath))

            if len(wMinValLossList)  > self.mLastN:
                wMinValLossList.pop(wMaxIdx)
                wDeleteName = wMinValLossNameList.pop(wMaxIdx)
                self.removeOldModel(wDeleteName)
            
            self.resetMinValLossList(wMinValLossList, wMinValLossNameList)
            # self.saveMeta(wSavePath)
            
    def getPeriodicReset(self):
        return self.mPeriodicReset
        
    def getMinValLossList(self):
        return self.mMinValLossList, self.mMinValLossNameList
    
    def resetMinValLossList(self, iMinValLossList, iMinValLossNameList):
        self.mMinValLossList, self.mMinValLossNameList = iMinValLossList, iMinValLossNameList
        
    def getSaveDir(self):
        return self.mSaveDir
    
    def removeOldModel(self, iOldName, iExtList = ['.data-00000-of-00001', '.index']):
        wSaveDir = self.getSaveDir()
        
        #Check and remove listed extensions for removal
        for wExt in iExtList:
            wPath = os.path.join(wSaveDir, iOldName+ wExt)
            if os.path.isfile(wPath): 
                os.remove(wPath)
        
        #Check unlisted extensions for removal
        for wFile in os.listdir(wSaveDir):
            if iOldName in wFile:
                os.remove(os.path.join(wSaveDir, wFile))

        time.sleep(0.1)
    

from keras.src import backend
from keras.src.callbacks.callback import Callback
from keras.src.utils import io_utils


class CustomLearningRateScheduler(Callback):
    """Learning rate scheduler.

    At the beginning of every epoch, this callback gets the updated learning
    rate value from `schedule` function provided at `__init__`, with the current
    epoch and current learning rate, and applies the updated learning rate on
    the optimizer.

    Args:
        schedule: A function that takes an epoch index (integer, indexed from 0)
            and current learning rate (float) as inputs and returns a new
            learning rate as output (float).
        verbose: Integer. 0: quiet, 1: log update messages.

    Example:

    >>> # This function keeps the initial learning rate for the first ten epochs
    >>> # and decreases it exponentially after that.
    >>> def scheduler(epoch, lr):
    ...     if epoch < 10:
    ...         return lr
    ...     else:
    ...         return lr * ops.exp(-0.1)
    >>>
    >>> model = keras.models.Sequential([keras.layers.Dense(10)])
    >>> model.compile(keras.optimizers.SGD(), loss='mse')
    >>> round(model.optimizer.learning_rate, 5)
    0.01

    >>> callback = keras.callbacks.LearningRateScheduler(scheduler)
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=15, callbacks=[callback], verbose=0)
    >>> round(model.optimizer.learning_rate, 5)
    0.00607

    """

    def __init__(self, schedule, iBegin = 100, iPatience = 50, iDivOutOfList =  np.cbrt(10.), verbose=0):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.mBegin = iBegin
        self.mPatience= iPatience
        self.mCounter=0
        self.mDivOutOfList = iDivOutOfList
        self.mMinTrainLoss = np.inf
      

    def on_epoch_begin(self, epoch, logs=None):
        

        if self.mCounter > self.mPatience and self.mPatience:
            wScheduleList = self.schedule.getSchedule()
            wIdx = -1
            for i, wTuple in enumerate(wScheduleList):
                if wTuple[0]>epoch:
                    wIdx = i
                    break
            if wIdx == -1:
                wNewScheduleList = wScheduleList + [(epoch, wScheduleList[wIdx][1]/self.mDivOutOfList)]
            
            else:
                wNewScheduleList = []
                for i, wTuple in enumerate(wScheduleList):
                    if i != wIdx:
                        wNewScheduleList.append(wTuple)
                    elif i == wIdx:
                        wNewScheduleList.append((epoch, wTuple[1]))
                
            
            self.schedule.setLRSchedule(wNewScheduleList)
            self.mCounter=0
            print("\nNew LR Schedule:")
            print(wNewScheduleList,'\n')
                
  
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')

        try:  # new API
            learning_rate = float(
                backend.convert_to_numpy(self.model.optimizer.learning_rate)
            )
            learning_rate = self.schedule(epoch, learning_rate)
        except TypeError:  # Support for old API for backward compatibility
            learning_rate = self.schedule(epoch)

        if not isinstance(learning_rate, (float, np.float32, np.float64)):
            raise ValueError(
                "The output of the `schedule` function should be a float. "
                f"Got: {learning_rate}"
            )

        self.model.optimizer.learning_rate = learning_rate
        if self.verbose > 0:
            io_utils.print_msg(
                f"\nEpoch {epoch + 1}: LearningRateScheduler setting learning "
                f"rate to {learning_rate}."
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        wLoss = logs['loss']
        if epoch>=self.mBegin:
            if wLoss < self.mMinTrainLoss:
                self.mMinTrainLoss = wLoss
                self.mCounter = 0
            else:
                self.mCounter +=1 
                
            
        logs["learning_rate"] = float(
            backend.convert_to_numpy(self.model.optimizer.learning_rate)
        )
    
    
class a:
    def __init__(self):
        self.apple=1
        print('a')
        
class b: 
    def __init__(self):
        self.orange=1
        print('b')
        
class c(a,b):
    def __init__(self):
        a.__init__(self)
        b.__init__(self)
        self.juice = self.apple+self.orange
        print('c')





















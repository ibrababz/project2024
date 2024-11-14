# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:03:37 2024

@author: i_bab
"""
import cv2 as cv
import os
from joblib import dump, load
import tensorflow as tf
import numpy as np
from dataLoad import getImageListFromBatch, getNameListFromBatch, getMapListsFromBatch, ProcessMapList3D, show_batch, flat_map_list_v2
from dataLoad import file_writing
from helper import adjust_number
# from loss_functions import ComputeLosses
from augment_utils import copyTemp, augmentXData, normListBy
from loss_functions import tensor_pos_neg_loss, dist_loss_from_list, tensor_dice_loss, tensor_map_loss, act_list_3D
from tensorflow.keras import backend as B  
from tensorflow.train import Checkpoint
from metrics import act_from_pred_list, threshold_list, refine_thresh_list
from metrics import get_cntrs_list, draw_cntrs_list, draw_cntrs_exp_list, find_cent_list, draw_cent_on_im_list
from metrics import TF_Metrics_from_batch, Metrics_from_TF_batch

class ModelTrainer:
    def __init__(self, iModel, iOptimizer):
        self.mModel= iModel
        self.mOptimizer = iOptimizer
        self.mMinValCkpt = Checkpoint(model = self.mModel, optimizer= self.mOptimizer)
        self.mLastNCkpt = Checkpoint(model = self.mModel, optimizer= self.mOptimizer)
        self.mNCkpt = Checkpoint(model = self.mModel, optimizer= self.mOptimizer)
        self.mShape = tuple(iModel.input.shape[1:])
        H, W, C = self.mShape
        self.mDim = (W,H)
        self.mLossLvlSched = {}
        self.mLayerStateDict = {}
        self.mLRSched = {}
        self.mMinValLoss = None
        self.mMinTrainLoss = None
        self.setLoadFlag(False)
        self.setLoadEpoch(None)
        
        
    
    def writeInitData(self):
        wLRSched = self.getLRSched()
        wLossLvlSched = self.getLossLvlSchedule()
        wLayerStateDict = self.getLayerStateDict()
        wInitDict = {'loss types': self.getLossDict(), 'break': {'train': self.getBreakEpochsTrain(), 'val': self.getBreakEpochsVal()}}
        wSchedDict = {'lr': wLRSched,
                      'loss level': wLossLvlSched,
                      'layer states': wLayerStateDict}
        wInitDict.update({'sched': wSchedDict})
        dump(wInitDict, os.path.join(self.getSaveDir(), 'init.dump'))
        
    def setCkptFreq(self, iEveryNEpochs = 25):
        self.mEveryNEpochs = iEveryNEpochs
        
    def setLossDict(self, iLossDict = {'Pos': 5., 'Neg': 5.*5., 'Dice': 1.}):
        self.mLossDict = iLossDict
    
    def getModel(self):
        return self.mModel
    
    def getOptimizer(self):
        return self.mOptimizer
    
    def getLossDict(self):
        return self.mLossDict
    
    def setLossLvlSchedule(self, iEpoch, iLossLvl):
        self.mLossLvlSched.update({str(iEpoch):iLossLvl})
        
    def setLossLvlScheduleFromDict(self, iDict):
        self.mLossLvlSched = iDict
        
    def getLossLvlSchedule(self):
        return self.mLossLvlSched
   
    def genLayerStateDict(self, iLayerNames, iLayerStates):
        wDict = {}
        for wName, wState in zip(iLayerNames, iLayerStates):
            wDict.update({wName: wState})
        return wDict
    
    def setLayerFreezeScheduleByName(self, iEpoch, iLayerStateDict):
        self.mLayerStateDict.update({str(iEpoch): iLayerStateDict})
        
    def setLayerFreezeScheduleFromDict(self, iDict):
        self.mLayerStateDict = iDict
    
    def setDecoderName(self, iDecoderName):
        self.mDecoderName = iDecoderName
    
    def getDecoderName(self):
        return self.mDecoderName
    
    def setLRSched(self, iEpoch, iLR):
        self.mLRSched.update({str(iEpoch): iLR})
        
    def setLRSchedFromDict(self, iDict):
        self.mLRSched = iDict
        
    def getLRSched(self):
        return self.mLRSched
        
    def checkLRSched(self):
        wKey = str(self.getEpoch())
        if wKey in self.mLRSched.keys():
            wLR = self.mLRSched[wKey]
            B.set_value(self.mOptimizer.learning_rate, wLR)
            print("Learning rate set to: %s"%wLR)
        
    def checkLossLvlSchedule(self):
        wKey = str(self.getEpoch())
        if wKey in self.mLossLvlSched.keys():
            wLossLvl = self.mLossLvlSched[wKey]
            self.setLossLvl(wLossLvl)
            self.resetLossWatchDog()
            
    def updateLossLvlScheduleFromLoaded(self):
        wKeyList = list(self.getLossLvlSchedule().keys())
        wKeyList.reverse()
        for wKey in wKeyList:
            if int(wKey) < self.getEpoch():
                wLossLvl = self.mLossLvlSched[wKey]
                self.setLossLvl(wLossLvl)
                break
     
    def checkLayerStates(self):
        wKey = str(self.getEpoch())
        if wKey in self.mLayerStateDict.keys():
            wDict = self.mLayerStateDict[wKey]
            wModel = self.getModel()
            wDecoderName = self.getDecoderName()
            if wDecoderName in wDict.keys():
                wDecoder = wModel.get_layer(wDecoderName)
                wDecoderDict = wDict[wDecoderName]
                for wDecoderLayer in wDecoderDict.keys():
                    wDecoder.trainable = wDecoderDict[wDecoderLayer]
            for wLayer in wDict.keys():
                if wLayer != wDecoderName:
                    wModel.get_layer(wLayer).trainable = wDict[wLayer]
            
            
    def updateLayerStatesFromLoaded(self):
        wKeyList = list(self.getLayerStateDict().keys())
        wCurrentEpoch = self.getEpoch()
        for wKey in wKeyList:
            if int(wKey) < wCurrentEpoch:
                wDict = self.getLayerStateDict()[wKey]
                wModel = self.getModel()
                wDecoderName = self.getDecoderName()
                if wDecoderName in wDict.keys():
                    wDecoder = wModel.get_layer(wDecoderName)
                    wDecoderDict = wDict[wDecoderName]
                    for wDecoderLayer in wDecoderDict.keys():
                        wDecoder.get_layer(wDecoderLayer).trainable = wDecoderDict[wDecoderLayer]
                for wLayer in wDict.keys():
                    if wLayer != wDecoderName:
                        wModel.get_layer(wLayer).trainable = wDict[wLayer]
                
            
            
    def setLossLvl(self, iLossLvl = [1, 0]):
        #can be [1, 0], [0, 1] or [1, 1]
        self.mLossLvl = iLossLvl
        self.nLvls = len(iLossLvl)
    
    def getLossLvl(self):
        return self.mLossLvl
        
    def getNLvls(self):
        return self.nLvls
    
    
    def setData(self, iTrainData, iValidData, iBatchSize):
        self.mTrainData = iTrainData
        self.mValidData = iValidData
        self.mBatchSize = iBatchSize
        self.mBatchLossTracker =[]
        self.mBatchAccTracker = []
        self.mTrainSize = len(iTrainData)
        self.mValidSize = len(iValidData)

    def getDataNameLists(self, iTrain = True):
        oNameLists = []
        for wData in self.getData(iTrain):
            oNameLists.append(wData.getNamesList())
        return oNameLists
        
    def genLogDataNames(self, iTrain = True, iWithPath = True):
        wLogNames = []
        for wData, wNameList in zip(self.getData(), self.getDataNameLists(iTrain)):
            for wName in wNameList:
                wLogName = wName
                if iWithPath:
                    wLogName = os.path.join(wData.getLoadDir(), wLogName)
                wLogNames.append(wLogName)
        return wLogNames

    def logDataNames(self, iWithPath = True):
        wLogNames = self.genLogDataNames(iTrain=True, iWithPath=iWithPath)
        
        with open(os.path.join(self.getSaveDir(), "train_files.txt"), 'w') as file:
            fwriting = file_writing(file)
            for wName in wLogNames:
                fwriting.write_file(wName)
                
        wLogNames = self.genLogDataNames(iTrain=False, iWithPath=iWithPath) 
        with open(os.path.join(self.getSaveDir(), "valid_files.txt"), 'w') as file:
            fwriting = file_writing(file)
            for wName in wLogNames:
                fwriting.write_file(wName)
        
    def getTrainSize(self):
        return self.mTrainSize
    
    def getValidSize(self):
        return self.mValidSize
                
    def getData(self, iTrain = True):
        if iTrain:
            return self.getTrainData()
        else:
            return self.getValidData()
        
    def getTrainData(self):
        return self.mTrainData
    
    def getValidData(self):
        return self.mValidData
    
    def getBatchSize(self):
        return self.mBatchSize
    
    def setNorm(self, iNorm = 0):
        self.mNorm = iNorm
    
    def getNorm(self):
        return self.mNorm
        
    def setSaveDir(self, iSaveDir):
        self.mSaveDir = iSaveDir
    
    def getSaveDir(self):
        return self.mSaveDir
               
    def setImageProcess(self, iImageProcess = None, iModelFlag = None):
        #iImageProcess customizable function
        self.mImageProcess = iImageProcess
        self.mModelFlag = iModelFlag
        
    def getImageProcess(self):
        return self.mImageProcess
    
    def getModelFlag(self):
        return self.mModelFlag
        
    def processImages(self, ioXDataAug):
        wModelFlag = self.getModelFlag()
        wImageProcess = self.getImageProcess()
        
        if wModelFlag == 'swin':
            ioXDataAug = wImageProcess(np.array(ioXDataAug), return_tensors="tf")[
                'pixel_values']
        elif wModelFlag == 'resnet':
            ioXDataAug = wImageProcess(
                np.array(ioXDataAug, dtype=np.float32)[..., ::-1])     
        elif wModelFlag == 'vgg':
            ioXDataAug = wImageProcess(
                np.array(ioXDataAug, dtype=np.float32)[..., ::-1])
            
        return ioXDataAug
        
    def setUnfreeze(self, iUnfreezeIndexList = None):
        self.mUnfreezeIndexList = iUnfreezeIndexList
        
    def setAugments(self, iAugments = None):
        self.mAugments = iAugments #make sure this is not an instance but a class defintion
        
    def setBatchGen(self, iBatchGen = None):
        self.mBatchGen = iBatchGen
    
    def getBatchGen(self):
        return self.mBatchGen
    
    def setEpoch(self, iEpoch):
        self.mCounter = iEpoch
        
    def incrementEpoch(self):
        self.mCounter += 1
    
    def getEpoch(self):
        return self.mCounter
    
    def getAugments(self, iTrain = True):
        if iTrain:
            oAugments = self.mAugments()
            oAugments.seed_(self.getEpoch())
        else:
            oAugments = None
        return oAugments
    
    def resetBatchCounter(self):
        self.mBatchCounter = 0
   
    def incrementBatchCounter(self):
        self.mBatchCounter+=1
        
    def getBatchCounter(self):
        return self.mBatchCounter
    
    def batchGenerator(self, iTrain = True):
        wBatchGenerator = self.getBatchGen()
        wData = self.getData(iTrain)
        if iTrain:
            wSeed = self.getEpoch()
        else:
            wSeed = 0
        return wBatchGenerator(wData, self.getBatchSize(), wSeed)

    def computeLosses(self, iXData, iMapLists, iTrain):
        wLossList = [tf.constant(0.) for i in range(self.getNLvls())]
        wLossUpdateList = [tf.constant(0.) for i in range(self.getNLvls())]
        self.resetLoss(iLossList = wLossList, iLossUpdateList = wLossUpdateList)     
        self.resetAugMaps()
        self.resetAugWeights()
        self.resetAugActs()
        
        wAugments = self.getAugments(iTrain)
        wTemp = copyTemp(wAugments)
        wXDataAug = augmentXData(iXData, wAugments)
        wNorm = self.getNorm()
        if wNorm:
            wXDataAug = normListBy(wXDataAug, wNorm)
        
        wXDataAug = self.processImages(wXDataAug)
        
        iPredList = self.mModel(wXDataAug, training= iTrain)
        self.processMaps(iPredList, iMapLists, wTemp)
        wLossKeys = list(self.getLossDict().keys())
        if 'Pos' in wLossKeys and 'Neg' in wLossKeys:
            self.computePosNeg(iPredList)
        if 'Dice' in wLossKeys:
            self.computeDice(iPredList)
        if 'Cart' in wLossKeys:
            self.computeCart(iPredList, 1)
        if 'Pix' in wLossKeys:
            self.computePix(iPredList)
            
        wBatchCounter = self.getBatchCounter()
        wEpoch = self.getEpoch()
        
        if iTrain:
            wDataSize = self.getTrainSize()
        else:
            wDataSize = self.getValidSize()
        
        if self.getLoadFlag():
            if wEpoch - self.getLoadEpoch() == 1:
                if (wBatchCounter+1)*self.getBatchSize() >= wDataSize:
                    self.showPlots(iXData, iPredList)
                
        if (wBatchCounter+1)*self.getBatchSize() >= wDataSize and wEpoch%self.getPlotFreq() == 0:
            self.showPlots(iXData, iPredList)
            
    def setPlotFreq(self, iPlotFreq = 100, iPlotAll = False):
        self.mPlotFreq = iPlotFreq
        self.mPlotAll = iPlotAll
        
    def getPlotFreq(self):
        return self.mPlotFreq
    
    def getPlotAll(self):
        return self.mPlotAll
    
    def getTrainableWeights(self, iLayers, ioTrainableWeights = []):
        for wLayer in iLayers:
            if wLayer.name == self.getDecoderName():
                self.getTrainableWeights(wLayer.layers, ioTrainableWeights)
            ioTrainableWeights.extend(wLayer.trainable_weights)

        return ioTrainableWeights
    
    def trainEpoch(self, iTrain = True):

        wBatchLossTracker = []
        self.resetBatchCounter()

        for wBatch in self.batchGenerator(iTrain):
    
            wXData = getImageListFromBatch(wBatch)
            wMapLists = getMapListsFromBatch(wBatch)
    
            if iTrain:
    
                with tf.GradientTape() as tape:
                    self.computeLosses(wXData, wMapLists, iTrain)
               
                wLossList, wLossUpdateList = self.getLoss()
                # wLayers = self.getModel().layers
                # wWeights = [wLayer.trainable_weights for wLayer in wLayers]
                # wTrainableWeights = []
                # for wWeight in wWeights:
                #     wTrainableWeights.extend(wWeight)
                wTrainableWeights = self.getTrainableWeights(self.getModel().layers, [])
                # grads = tape.gradient(wLossUpdateList, self.mModel.trainable_weights)
                # self.mOptimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, self.mModel.trainable_weights) if grad is not None)
                grads = tape.gradient(wLossUpdateList, wTrainableWeights)
                self.mOptimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, wTrainableWeights) if grad is not None)
     
            else:
                self.computeLosses(wXData, wMapLists, iTrain)
                wLossList, _ = self.getLoss()

            wBatchLossTracker.append(np.sum(wLossList))
            self.incrementBatchCounter()
        
        self.setLossPerEpoch(np.mean(wBatchLossTracker))

    def resetLossWatchDog(self):
        print('Resetting loss watchdog')
        self.setMinValLoss(np.inf)
        self.setMinTrainLoss(np.inf)
        self.resetMinTrainLossCounter(0)
        self.resetMinValLossCounter(0)
        self.setMinValLossName(None)
        wList, wName = [np.inf], ['inf']
        self.resetMinValLossList(wList, wName)    
        
     
    def train(self, iEpochs = (0,100), iSaveLastN = 5):

        if not self.getLoadFlag():
            self.resetLossWatchDog()
            self.resetLossTrackers([], [])
            self.writeInitData()
            wStart, wEnd = iEpochs
        else:
            wStart = self.getEpoch()
            _, wEnd = iEpochs
            if self.getSaveDir() != self.getLoadDir():
                self.writeInitData()
            
        self.setMinValSaveLastN(iSaveLastN)
        wLastN = self.getMinValSaveLastN()
        
        
        for i in range(wStart, wEnd):
            self.setEpoch(i)
            self.checkLRSched()
            self.checkLossLvlSchedule()
            self.checkLayerStates()
            self.trainEpoch(True)
            self.updateLossTracker(True)
            wTrainLossPerEpoch = self.getLossPerEpoch()
            self.checkTrainLoss(wTrainLossPerEpoch)
            
            self.trainEpoch(False)
            self.updateLossTracker(False)
            wValidLossPerEpoch = self.getLossPerEpoch()
            self.checkValLoss(wValidLossPerEpoch)

            self.addRemoveSavedModels(wLastN, wValidLossPerEpoch)
                        
            B.set_value(self.mOptimizer.iterations, i)
            print('ep:', self.getEpoch(), 'tr L:', np.round(wTrainLossPerEpoch,3), 'val L:', np.round(wValidLossPerEpoch,3), 'min_V:', np.round(self.getMinValLoss(), 3))
            self.saveEveryN()
            if self.checkBreak():
                break 
    
    def getCkptFreq(self):
        return self.mEveryNEpochs
    
    def saveEveryN(self):
        wEpoch = self.getEpoch()
        if (wEpoch+1)%self.getCkptFreq() == 0 and wEpoch != 0:
            wName = self.savePrint(iFlag = 'ckpt', verbose = 0)
            self.mNCkpt.write(wName)
            self.saveMeta(wName)
        
    
    def computePosNeg(self, iPredList):
        wLossLvl = self.getLossLvl()
        wLossDict = self.getLossDict()
        
        for i, wLvl in zip(range(len(wLossLvl)), wLossLvl):
            if wLvl:
                wPos, wNeg = tensor_pos_neg_loss(iPredList[i][..., 0],  self.getAugMaps(i), self.getAugActs(i))
                wLoss = wPos + wNeg
                wLossUpdate = wLossDict['Pos']*wPos + wLossDict['Neg']*wNeg
                self.incrementLoss(i, wLoss, wLossUpdate)

                
    def computeDice(self, iPredList):
        wLossLvl = self.getLossLvl()
        wLossDict = self.getLossDict()

        for i, wLvl in zip(range(len(wLossLvl)), wLossLvl):
            if wLvl:
                wDice = tensor_dice_loss(iPredList[i][..., 0],  self.getAugMaps(i), self.getAugWeights(i))
                wLoss = wDice
                wLossUpdate = wLossDict['Dice']*wDice
                self.incrementLoss(i, wLoss, wLossUpdate)

    def computeCart(self, iPredList, iFlag = 1):
        wLossLvl = self.getLossLvl()
        wLossDict = self.getLossDict()
        
        for i, wLvl in zip(range(len(wLossLvl)), wLossLvl):
            if wLvl:
                wDist = dist_loss_from_list(self.getAugWeights(i), iPredList[i][..., 0], iFlag)
                wLoss = wDist
                wLossUpdate = wLossDict['Cart']*wDist
                self.incrementLoss(i, wLoss, wLossUpdate)
                
    
    def computePix(self, iPredList):
        wLossLvl = self.getLossLvl()
        wLossDict = self.getLossDict()

        for i, wLvl in zip(range(len(wLossLvl)), wLossLvl):
            if wLvl:
                wPix = tensor_map_loss(iPredList[i][..., 0], self.getAugMaps(i), self.getAugWeights(i))
                wLoss = wPix
                wLossUpdate = wLossDict['Pix']*wPix
                self.incrementLoss(i, wLoss, wLossUpdate)
    
    def processMaps(self, iPredList, iMapLists, iTemp):
        wLossLvl = self.getLossLvl()

        for i, wLvl in zip(range(len(wLossLvl)), wLossLvl):
            if self.getPlotAll() and self.getEpoch()%self.getPlotFreq() == 0:
                wDimGrid = tuple(iPredList[i][0, ..., 0].shape[0:2])
                wMapAugList, wWeightList, _ = ProcessMapList3D(iMapLists[i], self.mDim, wDimGrid, copyTemp(iTemp))
                wActList = act_list_3D(wMapAugList)
                self.updateAugMaps(i, wMapAugList)
                self.updateAugWeights(i, wWeightList)
                self.updateAugActs(i, wActList)
            elif self.getPlotAll() and self.getLoadFlag():
                if self.getEpoch() - self.getLoadEpoch() == 1:
                    wDimGrid = tuple(iPredList[i][0, ..., 0].shape[0:2])
                    wMapAugList, wWeightList, _ = ProcessMapList3D(iMapLists[i], self.mDim, wDimGrid, copyTemp(iTemp))
                    wActList = act_list_3D(wMapAugList)
                    self.updateAugMaps(i, wMapAugList)
                    self.updateAugWeights(i, wWeightList)
                    self.updateAugActs(i, wActList)          
                elif wLvl:
                    wDimGrid = tuple(iPredList[i][0, ..., 0].shape[0:2])
                    wMapAugList, wWeightList, _ = ProcessMapList3D(iMapLists[i], self.mDim, wDimGrid, copyTemp(iTemp))
                    wActList = act_list_3D(wMapAugList)
                    self.updateAugMaps(i, wMapAugList)
                    self.updateAugWeights(i, wWeightList)
                    self.updateAugActs(i, wActList)
            elif wLvl:
                wDimGrid = tuple(iPredList[i][0, ..., 0].shape[0:2])
                wMapAugList, wWeightList, _ = ProcessMapList3D(iMapLists[i], self.mDim, wDimGrid, copyTemp(iTemp))
                wActList = act_list_3D(wMapAugList)
                self.updateAugMaps(i, wMapAugList)
                self.updateAugWeights(i, wWeightList)
                self.updateAugActs(i, wActList)
                
    def resetAugMaps(self, iMapAugLists = None):
        
        if iMapAugLists is None:
            wMapAugLists = [None]*self.getNLvls()
        else:
            wMapAugLists = iMapAugLists
        
        self.mMapAugLists = wMapAugLists
        
    def resetAugActs(self, iActAugLists = None):
        
        if iActAugLists is None:
            wActAugLists = [None]*self.getNLvls()
        else:
            wActAugLists = iActAugLists
        
        self.mActAugLists = wActAugLists
    
    def updateAugActs(self, iIdx, iActAug):
        self.mActAugLists[iIdx] = iActAug
        
    
    def resetAugWeights(self, iWeightAugLists = None):
        
        if iWeightAugLists is None:
            wWeightAugLists = [None]*self.getNLvls()
        else:
            wWeightAugLists = iWeightAugLists
        
        self.mWeightAugLists = wWeightAugLists
        
    def updateAugMaps(self, iIdx, iMapAug):
        self.mMapAugLists[iIdx] = iMapAug
        
    def updateAugWeights(self, iIdx, iWeightAug):
        self.mWeightAugLists[iIdx] = iWeightAug
    
    def getAugMaps(self, iIdx):
        return self.mMapAugLists[iIdx]
    
    def getAugWeights(self, iIdx):
        return self.mWeightAugLists[iIdx]
    
    def getAugActs(self, iIdx):
        return self.mActAugLists[iIdx]
    
    def resetLoss(self, iLossList = [tf.constant(0.), tf.constant(0.)], iLossUpdateList = [tf.constant(0.), tf.constant(0.)]):
        self.mLossList = iLossList
        self.mLossUpdateList = iLossUpdateList  
    
    def incrementLoss(self, iIdx, iLoss, iLossUpdate):
        self.mLossList[iIdx] += iLoss
        self.mLossUpdateList[iIdx] += iLossUpdate
        
    def getLoss(self):
        return self.mLossList, self.mLossUpdateList
        
     
    def showPlots(self, iXData, iPredList):
        show_batch(list(iXData))
        wLossLvl = self.getLossLvl()
        for i, wLvl in zip(range(len(wLossLvl)), wLossLvl):
            if self.getPlotAll():
                show_batch(flat_map_list_v2(self.getAugMaps(i)))
                show_batch(flat_map_list_v2(self.getAugActs(i)))
                show_batch(iPredList[i][..., 0].numpy()[..., None])
            elif wLvl:
                show_batch(flat_map_list_v2(self.getAugMaps(i)))
                show_batch(flat_map_list_v2(self.getAugActs(i)))
                show_batch(iPredList[i][..., 0].numpy()[..., None])
              
    def checkTrainLoss(self, iTrainLossPerEpoch):
        if(iTrainLossPerEpoch < self.getMinTrainLoss()):
            self.setMinTrainLoss(iTrainLossPerEpoch)
            self.resetMinTrainLossCounter()
        else:
            self.incrementMinTrainLossCounter()

    def checkValLoss(self, iValidLossPerEpoch):
        if(iValidLossPerEpoch < self.getMinValLoss()):
          self.setMinValLoss(iValidLossPerEpoch)
          wCurrentName = self.savePrint(iFlag = 'min_val')
          self.mMinValCkpt.write(wCurrentName)
          wOldName = self.getMinValLossName()
          if wOldName is not None:
              self.removeOldModel(wOldName)
          self.setMinValLossName(wCurrentName)
          self.resetMinValLossCounter()
          self.saveMeta(wCurrentName)
        else:
            self.incrementMinValLossCounter()

    def saveMeta(self, iName):
        dump(self.genMetaDict(),iName+'_meta.dump')
        
    def getMinValLoss(self):
        return self.mMinValLoss
    
    def setMinValLoss(self, iMinValLoss):
        self.mMinValLoss = iMinValLoss
        
    def resetMinValLossCounter(self, iCount = 0):
        self.mMinValLossCounter = iCount
    
    def getMinValLossCounter(self):
        return self.mMinValLossCounter
    
    def incrementMinValLossCounter(self):
        self.mMinValLossCounter +=1
        
    def getMinTrainLoss(self):
        return self.mMinTrainLoss
    
    def setMinTrainLoss(self, iMinTrainlLoss):
        self.mMinTrainLoss = iMinTrainlLoss
        
    def resetMinTrainLossCounter(self, iCount = 0):
        self.mMinTrainLossCounter = iCount
    
    def incrementMinTrainLossCounter(self):
        self.mMinTrainLossCounter +=1
        
    def getMinTrainLossCounter(self):
        return self.mMinTrainLossCounter
        
    def resetLossTrackers(self, iTrainTracker, iValTracker):
        self.mTrainLossTracker = iTrainTracker
        self.mValLossTracker = iValTracker
        
    def updateLossTracker(self, iTrain = True):
        if iTrain:
            self.mTrainLossTracker.append(self.getLossPerEpoch())
        else:
            self.mValLossTracker.append(self.getLossPerEpoch())
            
    def getLossTracker(self, iTrain = True):
        if iTrain:
            return self.mTrainLossTracker
        else:
            return self.mValLossTracker
   
    def setBreakEpochsTrain(self, iNoEpochs):
        self.mBreakEpochsTrain = iNoEpochs
    
    def setBreakEpochsVal(self, iNoEpochs):
        self.mBreakEpochsVal = iNoEpochs
        
    def getBreakEpochsTrain(self):
        return self.mBreakEpochsTrain
    
    def getBreakEpochsVal(self):
        return self.mBreakEpochsVal
    
    def getLossPerEpoch(self):
        return self.mLossPerEpoch
    
    def setLossPerEpoch(self, iLossPerEpoch):
        self.mLossPerEpoch = iLossPerEpoch
    
    def resetMinValLossList(self, iList = [np.inf], iName = ['inf']):
        self.mMinValLossList = iList
        self.mMinValLossNameList = iName
    
    def getMinValLossList(self):
        return self.mMinValLossList, self.mMinValLossNameList
    
    def setMinValSaveLastN(self, iSaveLastN = 5):
        self.mSaveLastN = iSaveLastN
    
    def getMinValSaveLastN(self):
        return self.mSaveLastN
    
    def setMinValLossName(self, iName):
        self.mOldName = iName
        
    def getMinValLossName(self):
        return self.mOldName
        
    
    def genMetaDict(self):
        wMinValLossList, wMinValLossNameList = self.getMinValLossList()
        wMinValLossCounter, wMinTrainLossCounter = self.getMinValLossCounter(), self.getMinTrainLossCounter()
        wTrainLossTracker, wValLossTracker = self.getLossTracker(), self.getLossTracker(False)
        wMinValLoss, wMinTrainLoss = self.getMinValLoss(), self.getMinTrainLoss()
        wMinValLossName = self.getMinValLossName()
        wValDict = {'min': wMinValLoss,
                    'name': wMinValLossName,
                    'list': wMinValLossList,
                    'names': wMinValLossNameList,
                    'counter': wMinValLossCounter,
                    'tracker': wValLossTracker}
        wTrainDict = {'min': wMinTrainLoss,
                    'counter': wMinTrainLossCounter,
                    'tracker': wTrainLossTracker} 
                    
        return {'val': wValDict, 'train': wTrainDict}
    
    def addRemoveSavedModels(self, iLastN, iValidLossPerEpoch):
        wMinValLossList, wMinValLossNameList = self.getMinValLossList()

        wMaxIndexesSorted = np.flip(np.argsort(wMinValLossList))
        wMaxIdx = wMaxIndexesSorted[0]
        wMaxMinValLoss = wMinValLossList[wMaxIdx]
        
        if iValidLossPerEpoch <= wMaxMinValLoss:
            wMinValLossList.append(iValidLossPerEpoch)
            wSavePath = self.savePrint(iFlag='list_'+str(np.round(iValidLossPerEpoch,3)).replace('.', 'p'), verbose=0)
            wMinValLossNameList.append(wSavePath)
            self.mLastNCkpt.write(wSavePath)

            if len(wMinValLossList)  > iLastN:
                wMinValLossList.pop(wMaxIdx)
                wDeleteName = wMinValLossNameList.pop(wMaxIdx)
                self.removeOldModel(wDeleteName)
            
            self.resetMinValLossList(wMinValLossList, wMinValLossNameList)
            self.saveMeta(wSavePath)
            # oDataDict = self.genMetaDict()
            # wSavePath = wSavePath + '_meta.dump'
            # dump(oDataDict, wSavePath)
        
   
    def savePrint(self, iFlag = '', verbose = 1):
        if verbose:
            print('-----saving weights------')
        wSaveDir = self.getSaveDir()
        wNumber = adjust_number(self.getEpoch())
        oSavePath = os.path.join(wSaveDir, wNumber + '_' + iFlag)
        return oSavePath
    
    def removeOldModel(self, iOldName):
        wSaveDir = self.getSaveDir()
        try:
            wExtList = ['.data-00000-of-00001', '.index', '_meta.dump']
            for wExt in wExtList:
                wDeleteName = iOldName + wExt
                wDeletePath = os.path.join(wSaveDir, wDeleteName)
                os.remove(wDeletePath)
        except:
            wFileList = os.listdir(wSaveDir)
            print("Could not find file with extension from extension list")
            for wFile in wFileList:
                wDeleteName = os.path.join(wSaveDir, wFile)
                if iOldName in wDeleteName:
                    if os.path.exists(wDeleteName):
                        os.remove(wDeleteName)
                    else:
                        print("Cannot remove a file that does not exist") 
            
    def checkBreak(self):
          if self.getMinTrainLossCounter() >= self.getBreakEpochsTrain(): 
              print('BREAKING NEWS: no improvement training')
              return 1
          elif self.getMinValLossCounter() >= self.getBreakEpochsVal(): #break from training
              print('BREAKING NEWS: no improvement validating')
              return 1
          else:
              return 0    
          
    def setGradualUnfreezeByNoEpochs(self, iStart, iNoEpochs, iLayerNames):
        pass
    
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
                    if len(iLayerNames) != 0:
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
                
    def getLayerStateDict(self):
        return self.mLayerStateDict
    
    def setLoadDir(self, iLoadDir):
        self.mLoadDir = iLoadDir
    
    def getLoadDir(self):
        return self.mLoadDir
    
    def loadInit(self, iMetaFile):
        wDict = load(os.path.join(self.getLoadDir(), 'init.dump'))
        # wDict = load(os.path.join(wTrainer.getLoadDir(), 'init.dump'))
        wLossDict = wDict['loss types']
        wTrainToBreak, wValToBreak = wDict['break']['train'],  wDict['break']['val']
        self.setLossDict(wLossDict)
        self.setBreakEpochsTrain(wTrainToBreak)
        self.setBreakEpochsVal(wValToBreak)
        wSchedDict = wDict['sched']
        self.loadSchedules(wSchedDict)
        self.loadMeta(iMetaFile)
        
    def loadSchedules(self, iSchedDict):
        self.setLRSchedFromDict(iSchedDict['lr'])
        self.setLossLvlScheduleFromDict(iSchedDict['loss level'])
        self.setLayerFreezeScheduleFromDict(iSchedDict['layer states'])
        
    def loadMeta(self, iLoadFile):
        wLoadPath = os.path.join(self.getLoadDir(), iLoadFile)
        wDataDict = load(wLoadPath + '_meta.dump')
        wValDict = wDataDict['val']
        wTrainDict = wDataDict['train']
        self.setMinValLoss(wValDict['min'])
        self.setMinValLossName(wValDict['name'])
        self.resetMinValLossList(wValDict['list'], wValDict['names'])
        self.resetMinValLossCounter(wValDict['counter'])
        self.setMinTrainLoss(wTrainDict['min'])
        self.resetMinTrainLossCounter(wTrainDict['counter'])
        self.resetLossTrackers(wTrainDict['tracker'], wValDict['tracker'])
        
    def setLoadFlag(self, iBool):
        self.mLoadFlag = iBool
            
    def getLoadFlag(self):
        return self.mLoadFlag
    
    def setLoadEpoch(self, iEpoch):
        self.mLoadedEpoch = iEpoch
    
    def getLoadEpoch(self):
        return self.mLoadedEpoch
    
    def loadFromCkpt(self, iLoadFile):
        self.setLoadFlag(True)
        self.setLoadEpoch(int(iLoadFile.split('_')[0]))
        self.setEpoch(self.getLoadEpoch()+1)
        wLoadPath = os.path.join(self.getLoadDir(), iLoadFile)
        wLoader = Checkpoint(model = self.getModel(), optimizer= self.getOptimizer())
        # print('load path is :%s'%wLoadPath)
        wLoader.read(wLoadPath)
        self.loadInit(iLoadFile)
        self.updateLayerStatesFromLoaded()
        self.updateLossLvlScheduleFromLoaded()
        
            
    def printSetupInfo(self):
        print('Model Flag: %s'%self.getModelFlag())
        print("\nLR rate sched:")
        print(self.getLRSched())
        print("\nLoss Types:")
        print(self.getLossDict())
        print("\nLoss level sched:")        
        print(self.getLossLvlSchedule())
        print("\nLayer State sched:")        
        wLayerStateDict = self.getLayerStateDict()
        wKeyList = wLayerStateDict.keys()
        wStateLen = len(wKeyList)
        for i, wKey in zip(range(wStateLen),wKeyList):
            wNames=[]
            for wKey2 in wLayerStateDict[wKey]:
                wNames.append(wKey2)
            wState = wLayerStateDict[wKey][wKey2]
            if wStateLen > 20:
                wDen = 8
                wP1, wP2, wP3, wP4 = int(wStateLen/wDen) , int((wDen-1)*wStateLen/(2*wDen)), int((wDen+1)*wStateLen/(2*wDen)), int((wDen-1)*wStateLen/wDen)
                if i<wP1 or (i>wP2 and i<wP3) or i>wP4:
                    print(wKey, wNames, wState)
                elif i==wP1 or i== int((wP2+wP3)/2) or i==wP4: 
                    print("...")
            else: 
                print(wKey, wNames, wState)
            
            
class ModelEvaluator(ModelTrainer):
    def __init__(self, iModel, iOptimizer):
        super().__init__(iModel, iOptimizer)
        self.getModel().trainable = False
        self.setAugments(None)
        self.setResolution()
        
    def setResolution(self):
        wRes = len(self.getModel().output)
        self.setLossLvl([1]*wRes)
    
    def setData(self, iTestData, iBatchSize):
        self.mTestData = iTestData
        self.mBatchSize = iBatchSize   
        self.mTestSize = len(iTestData)
                
    def getData(self):
        return self.mTestData
            
    def getTestData(self):
        return self.mTestData
    
    def getValidData(self):
        #wrapper
        return self.getTestData()
    
    def getTestSize(self):
        return self.mTestSize
    
    def getValidSize(self):
        #wrapper
        return self.getTestSize()
    
        
    def loadFromCkpt(self, iLoadFile):
        self.setLoadFlag(True)
        wLoadPath = os.path.join(self.getLoadDir(), iLoadFile)
        wLoader = Checkpoint(model = self.getModel(), optimizer= self.getOptimizer())
        wLoader.read(wLoadPath)
        self.loadInit()
        self.getModel().trainable = False

        
    def loadInit(self):
        wDict = load(os.path.join(self.getLoadDir(), 'init.dump'))
        wLossDict = wDict['loss types']
        self.setLossDict(wLossDict)
        self.setBreakEpochsTrain(np.inf)
        self.setBreakEpochsVal(np.inf)
        wSchedDict = wDict['sched']
        self.loadSchedules(wSchedDict)
        
    def showPlots(self, iXData, iPredList, iIdx):
        if self.getShowPlots():
            show_batch(list(iXData))
            show_batch(flat_map_list_v2(self.getAugMaps(iIdx)))
            show_batch(flat_map_list_v2(self.getAugActs(iIdx)))
            show_batch(self.getThreshPreds(iIdx))
            
    
    def setSaveSize(self, iDim):
        self.mSaveDim = iDim
        
    def getSaveSize(self):
        return self.mSaveDim
    
    def setSavePlotType(self, iType ='maps'):
        self.mType = iType
        
    def getSavePlotType(self):
        return self.mType
        
    def savePlots(self, iIdx):
        if self.getSavePlots():
            wType = self.getSavePlotType()
            if wType == 'maps':
                wPlots = self.getThreshPreds(iIdx)
                wInterpolation = cv.INTER_NEAREST
            elif wType == 'act':
                wPlots = self.getPredActs(iIdx)
                wInterpolation = cv.INTER_NEAREST     
            elif wType == 'cent':
                wPlots = self.getPredCents(iIdx)
                wInterpolation = cv.INTER_NEAREST
            elif wType == 'truth_acts':
                wPlots = self.getFlatActs(iIdx)
                wInterpolation = cv.INTER_NEAREST
            elif wType == 'truth_maps':
                wPlots = self.getFlatMaps(iIdx)
                wInterpolation = cv.INTER_NEAREST
            wSaveDir = os.path.join(self.getLoadDir(), 'test_output_' + wType)
            os.makedirs(wSaveDir, exist_ok=True)
            for wPlt, wName in zip(wPlots, self.getNames()):
                wIm = wPlt[..., ::-1][...,0]
                wIm = np.uint8(np.clip(wIm*255.0, 0, 255))
                cv.imwrite(os.path.join(wSaveDir, wName[0]), cv.resize(wIm, self.getSaveSize(), interpolation=wInterpolation))
            
            
    def printMetrics(self, iIdx, iThresh, iRound=3):
        wPrec, wReca, wF1, wMIoU = self.getPrecMetrics(iIdx, iRound)
        print('Treshold: %s, Prec: %s, Reca: %s, F1: %s, MIoU: %s'%(np.round(iThresh, 4), wPrec, wReca, wF1, wMIoU))
        
        
    def setShowPlots(self, iBool):
        self.mShowPlots = iBool
    
    def getShowPlots(self):
        return self.mShowPlots
    
            
    def setSavePlots(self, iBool):
        self.mSavePlots = iBool
    
    def getSavePlots(self):
        return self.mSavePlots
                

    def batchGenerator(self):
        wBatchGenerator = self.getBatchGen()
        wData = self.getData()
        wSeed = 0
        return wBatchGenerator(wData, self.getBatchSize(), wSeed)
    
    def setImageFolder(self, iFolder):
        self.mImageFolder = iFolder
    
    def getImageFolder(self):
        return self.mImageFolder
    
    def computeMetrics(self, iThresh, iResIdx = 0, iType = 'maps'):
        
        wModel = self.getModel()
        wBatchGen = self.batchGenerator()
        
        self.resetTFMetricsList()
        self.resetPrecMetricsList()
        for wBatch in wBatchGen:

            wXData = getImageListFromBatch(wBatch)
            wNorm = self.getNorm()
            if wNorm:
                wXData = normListBy(wXData, wNorm)            
            wXNames = getNameListFromBatch(wBatch)
            wMapLists = getMapListsFromBatch(wBatch)            
            wXDataProcess = self.processImages(wXData)

            wPredList = wModel(np.array(wXDataProcess))
            self.resetNames()
            self.resetTFMetricsBatch()
            self.resetPrecMetricsBatch()
            self.resetAugMaps()
            self.resetAugWeights()
            self.resetAugActs()
            self.resetFlatMaps()
            self.resetFlatActs()
            self.resetThreshPreds()
            self.resetPredActs()
            self.resetPredCents()

            
            self.processMaps(wPredList, wMapLists, iResIdx)
            self.flatten(iResIdx)
            self.thresholdPreds(wPredList, iResIdx, iThresh)
            self.extractPredActivations(iResIdx)
            self.computePredCentroids(iResIdx)
            self.computeMetricsBatch(iIdx=iResIdx, iFlag=iType)
            self.updateMetricsLists(iResIdx)
            self.updateNames(wXNames)
            self.showPlots(wXData, wPredList, iResIdx)
            self.savePlots(iResIdx)
            
        """
            wPredListIdx = list(wPredList[iResIdx][...,0].numpy()[...,None])
            
            act_list1 = act_list_3D(truth_map_list)
            flat_act_list1 = flat_map_list_v2(act_list1)
    
            #show_batch(blue_act_list1)
            flat_map_list1 = flat_map_list_v2(truth_map_list)
    
            pred_list_thresh = threshold_list(wPredListIdx.copy(), iThresh, 1.0)
                    
            pred_list_thresh = refine_thresh_list(wPredListIdx.copy(), pred_list_thresh.copy())
            
            #=======================FOR METRICS START==================================
            truth_batch1 = flat_map_list1.copy()
            truth_act_batch1 = flat_act_list1.copy()
            
            pred_batch1 = pred_list_thresh.copy()
            

            #=======================FOR METRICS END====================================
            
            #=======================RESIZING START=============================
            
            flat_map_list1 = resize_list(flat_map_list1, (H,W))
            wPredListIdx = resize_list(wPredListIdx,(H,W))
            pred_list_thresh = resize_list(pred_list_thresh,(H,W))
            flat_act_list1 = resize_list(flat_act_list1, (H,W))
            
            #=======================RESIZING END=============================
            
            
            #show_batch(flat_act_list1)
            clr_act_list1 = colorMap_list(flat_act_list1.copy(), cv.COLORMAP_HOT)
            #show_batch(clr_act_list1)
            blue_act_list1 = zero_clr_channel_list(clr_act_list1.copy(), [1,2])
    

            #show_batch(colorMap_list(pred_list_thresh.copy(), cv.COLORMAP_HOT))
            
            
            
            #flat_map_list1_thresh = threshold_list(flat_map_list1, 0.0, 1.0) 
            #show_batch(pred_list_thresh)
            shape_grid2= wPredListIdx[0].shape
            h_grid = shape_grid2[0]
    
            
            pred_thresh_clr = colorMap_list(pred_list_thresh, cv.COLORMAP_HOT)
            truth_clr = colorMap_list(flat_map_list1, cv.COLORMAP_OCEAN)
            blend_list = blend_pred_truth_list(pred_thresh_clr, truth_clr, 0.4, 0.6)
            blend_with_original = blend_pred_truth_list(wXData, blend_list, 0.4, 0.6)
            cmb_blend_lists = cmb_3_images_list(wXData, blend_with_original, blend_list)
            
            
            
            #show_batch(blend_with_original)
            #show_batch(cmb_blend_lists)
            
            blend_with_original_pred = blend_pred_truth_list(wXData, pred_thresh_clr, 0.5, 0.5)
            cmb_blend_lists_pred = cmb_3_images_list(wXData, blend_with_original_pred, pred_thresh_clr)
            cmb_pred_act_lists = cmb_3_images_list(cmb_blend_lists_pred, blue_act_list1, blue_act_list1)
            
            #show_batch(blend_with_original_pred)
            #show_batch(cmb_blend_lists_pred)
            
            
         
            #==============================METRICS START======================================================
            pred_cntrs_batch1 = get_cntrs_list(pred_batch1)
            pred_drawn_list1 = draw_cntrs_list(pred_batch1.copy(), pred_cntrs_batch1, thickness = -1, on_black = 1.)
            pred_drawn_expanded_list1 = draw_cntrs_exp_list(pred_batch1.copy(), pred_cntrs_batch1, thickness = -1)
            centers_on_im_list1 = find_cent_list(pred_cntrs_batch1)
            drawn_cent_on_im_list1 = draw_cent_on_im_list(centers_on_im_list1, None, pred_batch1[0].shape)
            flat_pred_act_list1 = act_from_pred_list(pred_batch1)
            
            TPi_batch, FPi_batch, TNi_batch, FNi_batch = TF_Metrics_from_batch(truth_batch1, pred_batch1); prefix = 'map'
            # TPi_batch, FPi_batch, TNi_batch, FNi_batch = TF_Metrics_from_batch(truth_act_batch1, drawn_cent_on_im_list1); prefix = 'cent'
            # TPi_batch, FPi_batch, TNi_batch, FNi_batch = TF_Metrics_from_batch(truth_act_batch1, flat_pred_act_list1); prefix = 'act'         
            Prec_batch, Reca_batch, F1_batch, MIoU_batch = Metrics_from_TF_batch(TPi_batch, FPi_batch, TNi_batch, FNi_batch)
        
            #==============================METRICS END======================================================
            
            
            drawn_cent_resize_list1 = resize_list(drawn_cent_on_im_list1, (H,W))
            pred_act_resize_list1 = resize_list(flat_pred_act_list1, (H,W))
            #===========================SAVE IMAGE START============================================
            # show_batch(wXData)
            # save_images_from_list(wXData, wXNames, save_im_dir, '_00O', 0)
            # save_images_from_list(colorMap_list(flat_map_list1.copy(), cv.COLORMAP_OCEAN), wXNames, save_im_dir, '_01T', 0)
            # save_images_from_list(colorMap_list(wPredListIdx.copy(), cv.COLORMAP_HOT), wXNames, save_im_dir, '_02P', 0)
            
            # blue_act_list1 = zero_clr_channel_list(colorMap_list(flat_act_list1.copy(), cv.COLORMAP_OCEAN), [1,2])
            # save_images_from_list(blue_act_list1, wXNames, save_im_dir, '_01T_act', 0)
            
            # red_pred_cent_list1 = zero_clr_channel_list(colorMap_list(drawn_cent_resize_list1.copy(), cv.COLORMAP_HOT), [0,1])
            # save_images_from_list(red_pred_cent_list1, wXNames, save_im_dir, '_02P_cent', 0)
            
            # green_pred_act_list1 = zero_clr_channel_list(colorMap_list(pred_act_resize_list1.copy(), cv.COLORMAP_HOT), [0,2])
            # save_images_from_list(green_pred_act_list1, wXNames, save_im_dir, '_02P_act', 0)
           
            # save_images_from_list(colorMap_list(pred_list_thresh.copy(), cv.COLORMAP_HOT), wXNames, save_im_dir, '_03P', 0)
            # save_images_from_list(blend_list, wXNames, save_im_dir, '_04B', 0)
            # #save_images_from_list(blend_with_original, wXNames, save_im_dir, '_05B', 0)
            # save_images_from_list(cmb_pred_act_lists, wXNames, save_im_dir, '_05B', 0)
            
            # cmb_actcent_lists = add_clr_im_list(blue_act_list1, red_pred_cent_list1) 
            # save_images_from_list(cmb_actcent_lists, wXNames, save_im_dir, '_06C_cent', 0)
            
            # cmb_actact_lists = add_clr_im_list(blue_act_list1, green_pred_act_list1) 
            # save_images_from_list(cmb_actact_lists, wXNames, save_im_dir, '_06C_act', 0)
            # show_batch(pred_list_thresh)
            pred_cntrs_to_draw = get_cntrs_list(pred_list_thresh)
            pred_cntrs_drawn = draw_cntrs_list(wPredListIdx.copy(), pred_cntrs_to_draw, -1, 1,4,1)
            # show_batch(pred_cntrs_drawn)
            pred_cntrs_drawn_red = zero_clr_channel_list(colorMap_list(pred_cntrs_drawn.copy(), cv.COLORMAP_HOT), [0,1])
            # show_batch(pred_cntrs_drawn)
            
            # save_images_from_list(pred_cntrs_drawn_red, wXNames, save_im_dir, '_02P_contour', 0)
            
            #===========================SAVE IMAGE END==============================================
            
            #===========================SHOW START=================================================
            # show_batch(colorMap_list(flat_map_list1.copy(), cv.COLORMAP_OCEAN))
            # show_batch(colorMap_list(wPredListIdx.copy(), cv.COLORMAP_HOT))
            # show_batch(colorMap_list(pred_list_thresh.copy(), cv.COLORMAP_HOT))
            
            # show_batch(blend_list)
            # show_batch(cmb_pred_act_lists)
    
            # show_batch(truth_batch1)            
            # show_batch(truth_act_batch1) 
            # show_batch(pred_batch1)
            # show_batch(pred_drawn_list1)
            # show_batch(drawn_cent_on_im_list1)
            # show_batch(red_pred_cent_list1) 
            # show_batch(green_pred_act_list1)
            #===========================SHOW END=================================================
            
            
            TPi_list.extend(TPi_batch)
            FPi_list.extend(FPi_batch)
            TNi_list.extend(TNi_batch)
            FNi_list.extend(FNi_batch)
    
            Prec_list.extend(Prec_batch)
            Reca_list.extend(Reca_batch)
            F1_list.extend(F1_batch)
            MIoU_list.extend(MIoU_batch)
            #==============================METRICS END======================================================
           
            # cntrs_list = get_cntrs_list(pred_list_thresh_resized)
            # truth_cntrs_list = get_cntrs_list(flat_map_list1_thresh_resized)
            #break
            # show_batch(colorMap_list(pred_list_thresh_resized, cv.COLORMAP_HOT))
            
            # cmb_list = draw_cntrs_list(colorMap_list(resize_list(flat_map_list1.copy(), (H,W)), cv.COLORMAP_OCEAN), cntrs_list, colour= (0., 0., 1.), thickness =5, on_black = 0.)
            # #cmb_list = draw_cntrs_list(cmb_list, truth_cntrs_list, colour= (0., 0., 1.), thickness =5)
            # show_batch(cmb_list)
            # #show_batch(draw_cntrs_list(pred_list_thresh, cntrs_list))
            #break
        Prec_mean = np.mean(Prec_list)
        Reca_mean = np.mean(Reca_list)
        F1_mean = np.mean(F1_list)
        MIoU_mean = np.mean(MIoU_list)
        """
    def processMaps(self, iPredList, iMapLists, iIdx):
        wDimGrid = tuple(iPredList[iIdx][0, ..., 0].shape[0:2])
        wMapAugList, wWeightList, _ = ProcessMapList3D(iMapLists[iIdx], self.mDim, wDimGrid, None)
        wActList = act_list_3D(wMapAugList)
        self.updateAugMaps(iIdx, wMapAugList)
        self.updateAugWeights(iIdx, wWeightList)
        self.updateAugActs(iIdx, wActList)
        
    def resetNames(self):
        self.mNameList = None
        
    def updateNames(self, iNameList):
        self.mNameList = iNameList
        
    def getNames(self):
        return self.mNameList
        
    def resetFlatMaps(self, iFlatMapsLists = None):
        
        if iFlatMapsLists is None:
            wFlatMapsLists = [None]*self.getNLvls()
        else:
            wFlatMapsLists = iFlatMapsLists
        
        self.mFlatMapsLists = wFlatMapsLists
            
    def updateFlatMaps(self, iIdx, iFlatMaps):
        self.mFlatMapsLists[iIdx] = iFlatMaps
     
        
    def getFlatMaps(self, iIdx):
        return self.mFlatMapsLists[iIdx]
    
    
    def resetFlatActs(self, iFlatActsLists = None):
        
        if iFlatActsLists is None:
            wFlatActsLists = [None]*self.getNLvls()
        else:
            wFlatActsLists = iFlatActsLists
        
        self.mFlatActsLists = wFlatActsLists
            
    def updateFlatActs(self, iIdx, iFlatActs):
        self.mFlatActsLists[iIdx] = iFlatActs
     
    def getFlatActs(self, iIdx):
        return self.mFlatActsLists[iIdx]
    
    def flatten(self, iIdx):
        wMaps = self.getAugMaps(iIdx)
        wActs = self.getAugActs(iIdx)
        wFlatMaps = flat_map_list_v2(wMaps)
        wFlatActs = flat_map_list_v2(wActs)
        self.updateFlatMaps(iIdx, wFlatMaps)
        self.updateFlatActs(iIdx, wFlatActs)
        
    def thresholdPreds(self, iPredList, iIdx, iThresh):
        wPredListIdx = list(iPredList[iIdx][...,0].numpy()[...,None])
        wThreshPreds = threshold_list(wPredListIdx.copy(), iThresh, 1.0)        
        wThreshPreds = refine_thresh_list(wPredListIdx.copy(), wThreshPreds.copy())
        self.updateThreshPreds(iIdx, wThreshPreds)
        
    def extractPredActivations(self, iIdx):
        wPredActs = act_from_pred_list(self.getThreshPreds(iIdx))
        self.updatePredActs(iIdx, wPredActs)
    
    def computePredCentroids(self, iIdx):
        pred_cntrs_batch1 = get_cntrs_list(self.getThreshPreds(iIdx))
        centers_on_im_list1 = find_cent_list(pred_cntrs_batch1)
        wPredCents = draw_cent_on_im_list(centers_on_im_list1, None, self.getThreshPreds(iIdx)[0].shape)
        self.updatePredCents(iIdx, wPredCents)

        
    
        
    def resetPredActs(self, iPredActsLists = None):
        
        if iPredActsLists is None:
            wPredActsLists = [None]*self.getNLvls()
        else:
            wPredActsLists = iPredActsLists
        
        self.mPredActsLists = wPredActsLists
            
    def updatePredActs(self, iIdx, iPredActs):
        self.mPredActsLists[iIdx] = iPredActs
     
    def getPredActs(self, iIdx):
        return self.mPredActsLists[iIdx]
    
    def resetPredCents(self, iPredCentsLists = None):
        
        if iPredCentsLists is None:
            wPredCentsLists = [None]*self.getNLvls()
        else:
            wPredCentsLists = iPredCentsLists
        
        self.mPredCentsLists = wPredCentsLists
            
    def updatePredCents(self, iIdx, iPredCents):
        self.mPredCentsLists[iIdx] = iPredCents
     
    def getPredCents(self, iIdx):
        return self.mPredCentsLists[iIdx]   
    
    def resetThreshPreds(self, iThreshPredsLists = None):
        
        if iThreshPredsLists is None:
            wThreshPredsLists = [None]*self.getNLvls()
        else:
            wThreshPredsLists = iThreshPredsLists
        
        self.mThreshPredsLists = wThreshPredsLists
    
    def updateThreshPreds(self, iIdx, iThreshPreds):
        self.mThreshPredsLists[iIdx] = iThreshPreds   

    def getThreshPreds(self, iIdx):
        return self.mThreshPredsLists[iIdx]
    
    def computeMetricsBatch(self, iIdx, iFlag = 'maps'):
        self.resetTFMetricsBatch()
        self.resetPrecMetricsBatch()
        
        if iFlag == 'maps':
            wTruth = self.getFlatMaps(iIdx)            
            wPred = self.getThreshPreds(iIdx)
        else:
            wTruth = self.getFlatActs(iIdx)
            if iFlag == 'act':
                wPred = self.getPredActs(iIdx)
            elif iFlag == 'cent':
                wPred = self.getPredCents(iIdx)
     
        wTPBatch, wFPBatch, wTNBatch, wFNBatch = TF_Metrics_from_batch(wTruth, wPred)
        self.updateTFMetricsBatch(iIdx, wTPBatch, wFPBatch, wTNBatch, wFNBatch)
        self.updatePrecMetricsBatch(iIdx)
    
    def resetLists(self, iLists):
        if iLists is None:
            oLists = [None]*self.getNLvls()
        else:
            oLists = iLists
        return oLists
    
    def resetTFMetricsBatch(self, iTPBatch = None, iFPBatch = None, iTNBatch = None, iFNBatch = None):
        self.mTPBatch = self.resetLists(iTPBatch)
        self.mFPBatch = self.resetLists(iFPBatch)
        self.mTNBatch = self.resetLists(iTNBatch)
        self.mFNBatch = self.resetLists(iFNBatch)
    
    def resetPrecMetricsBatch(self, iPrecBatch = None, iRecaBatch = None, iF1Batch = None, iMIoUBatch = None):
        self.mPrecBatch = self.resetLists(iPrecBatch)
        self.mRecaBatch = self.resetLists(iRecaBatch)
        self.mF1Batch = self.resetLists(iF1Batch)
        self.mMIoUBatch = self.resetLists(iMIoUBatch)
        
    def updateTFMetricsBatch(self, iIdx, iTPBatch, iFPBatch, iTNBatch, iFNBatch):
        self.mTPBatch[iIdx] = iTPBatch
        self.mFPBatch[iIdx] = iFPBatch
        self.mTNBatch[iIdx] = iTNBatch
        self.mFNBatch[iIdx] = iFNBatch
        
    def updatePrecMetricsBatch(self, iIdx):
        wTPBatch, wFPBatch, wTNBatch, wFNBatch = self.getTFMetricsBatch(iIdx)
        wPrecBatch, wRecaBatch, wF1Batch, wMIoUBatch = Metrics_from_TF_batch(wTPBatch, wFPBatch, wTNBatch, wFNBatch)
        self.mPrecBatch[iIdx] = wPrecBatch
        self.mRecaBatch[iIdx] = wRecaBatch
        self.mF1Batch[iIdx] = wF1Batch
        self.mMIoUBatch[iIdx] = wMIoUBatch
        
    def getTFMetricsBatch(self, iIdx):
        return  self.mTPBatch[iIdx], self.mFPBatch[iIdx], self.mTNBatch[iIdx], self.mFNBatch[iIdx]
    
    def getPrecMetricsBatch(self, iIdx):
        return  self.mPrecBatch[iIdx], self.mRecaBatch[iIdx], self.mF1Batch[iIdx], self.mMIoUBatch[iIdx]
    
    def resetTFMetricsList(self, iTPList = None, iFPList = None, iTNList = None, iFNList = None):
        self.mTPList = self.resetLists(iTPList)
        self.mFPList = self.resetLists(iFPList)
        self.mTNList = self.resetLists(iTNList)
        self.mFNList = self.resetLists(iFNList)
        
    def resetPrecMetricsList(self, iPrecList = None, iRecaList = None, iF1List = None, iMIoUList = None):
        self.mPrecList = self.resetLists(iPrecList)
        self.mRecaList = self.resetLists(iRecaList)
        self.mF1List = self.resetLists(iF1List)
        self.mMIoUList = self.resetLists(iMIoUList)
        
    def updateTFMetricsList(self, iIdx):
        wTPBatch, wFPBatch, wTNBatch, wFNBatch = self.getTFMetricsBatch(iIdx)
        
        if self.mTPList[iIdx] is None:
            self.mTPList[iIdx] = wTPBatch
        else:
            self.mTPList[iIdx].extend(wTPBatch)
            
        if self.mFPList[iIdx] is None:
            self.mFPList[iIdx] = wFPBatch
        else:
            self.mFPList[iIdx].extend(wFPBatch)
            
        if self.mTNList[iIdx] is None:
            self.mTNList[iIdx] = wTNBatch
        else:
            self.mTNList[iIdx].extend(wTNBatch)
            
        if self.mFNList[iIdx] is None:
            self.mFNList[iIdx] = wFNBatch
        else:
            self.mFNList[iIdx].extend(wFNBatch)
            
    def updatePrecMetricsList(self, iIdx):
        wPrecBatch, wRecaBatch, wF1Batch, wMIoUBatch = self.getPrecMetricsBatch(iIdx)
        
        if self.mPrecList[iIdx] is None:
            self.mPrecList[iIdx] = wPrecBatch
        else:
            self.mPrecList[iIdx].extend(wPrecBatch)
            
        if self.mRecaList[iIdx] is None:
            self.mRecaList[iIdx] = wRecaBatch
        else:
            self.mRecaList[iIdx].extend(wRecaBatch)
            
        if self.mF1List[iIdx] is None:
            self.mF1List[iIdx] = wF1Batch
        else:
            self.mF1List[iIdx].extend(wF1Batch)
            
        if self.mMIoUList[iIdx] is None:
            self.mMIoUList[iIdx] = wMIoUBatch
        else:
            self.mMIoUList[iIdx].extend(wMIoUBatch)    

    def getTFMetricsList(self, iIdx):
        return  self.mTPList[iIdx], self.mFPList[iIdx], self.mTNList[iIdx], self.mFNList[iIdx]
    
    def getPrecMetricsList(self, iIdx):
        return  self.mPrecList[iIdx], self.mRecaList[iIdx], self.mF1List[iIdx], self.mMIoUList[iIdx]

    def getTFMetrics(self, iIdx):
        wTPList, wFPList, wTNList, wFNList = self.getTFMetricsList(iIdx)
        return np.mean(wTPList), np.mean(wFPList), np.mean(wTNList), np.mean(wFNList)

    def getPrecMetrics(self, iIdx, iRound = None):
        wPrecList, wRecaList, wF1List, wMIoUList = self.getPrecMetricsList(iIdx)
        oPrec, oReca, oF1, oMIoU = np.mean(wPrecList), np.mean(wRecaList), np.mean(wF1List), np.mean(wMIoUList)
        if iRound is not None:
            oPrec, oReca, oF1, oMIoU = np.round(oPrec, iRound), np.round(oReca, iRound), np.round(oF1, iRound), np.round(oMIoU, iRound)
        return oPrec, oReca, oF1, oMIoU
    
    def updateMetricsLists(self, iIdx):
        self.updateTFMetricsList(iIdx)
        self.updatePrecMetricsList(iIdx)
        
    def getDataNameLists(self):
        oNameLists = []
        for wData in self.getData():
            oNameLists.append(wData.getNamesList())
        return oNameLists
        
    def genLogDataNames(self, iWithPath = True):
        wLogNames = []
        for wData, wNameList in zip(self.getData(), self.getDataNameLists()):
            for wName in wNameList:
                wLogName = wName
                if iWithPath:
                    wLogName = os.path.join(wData.getLoadDir(), wLogName)
                wLogNames.append(wLogName)
        return wLogNames

    def logDataNames(self, iWithPath = True):
        wLogNames = self.genLogDataNames(iWithPath)
        with open(os.path.join(self.getLoadDir(), "test_files.txt"), 'w') as file:
            fwriting = file_writing(file)
            for wName in wLogNames:
                fwriting.write_file(wName)
        
            
if __name__ == '__main__':
    pass









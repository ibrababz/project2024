#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 20:51:43 2025

@author: ibabi
"""
import os
import sys
import numpy as np
import math

import tensorflow as tf

from tfMapUtils import tfFlatMapBatch, tfScale3DBatch
from models import makeYoloTypeFlat
from models import save_model_summary
from customLoss import CustomLoss4, tfResizeFunc
from customCallbacks import PlotCallBack, LayerUnfreezeCallback, CustomLRSchedulerFunction
from customCallbacks import SaveEveryN, SaveLastN
from customCallbacks import SleepCallBack, LogEveryN, StopTraining
from customCallbacks import CustomLearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import plot_model
from tensorflow.train import Checkpoint

from tfAugmentations import chooseAugmentV2

from functools import partial

from pathlib import Path
from parsingFunctions import logArgs, decodeParserSchedAsDict

import argparse

def dictToTupleV4(x, iResizeFunctionList, iTrain=True, iPreprocess=True):
    wIm, wMap = x['images'], x['segmentation_masks']#dividing before preprocessing is incorrect but it was done when previously training in TrainerClass.py
    wIm = tf.cast(wIm, dtype=tf.float32)#/255.
    if iPreprocess:
        wIm = preprocess_input(wIm[...,::-1])
    wMapTuple = tuple(tfFlatMapBatch(tfScale3DBatch(wResize(wMap))) for wResize in wResizeFunctionList)
    if 'weights' in x.keys():
        return wIm, wMapTuple, x['weights']
    return wIm, wMapTuple 


def preprocessIm(iDataTuple):
    oIm, oMapTuple = iDataTuple
    oIm = preprocess_input(oIm[...,::-1])
    return oIm, oMapTuple
                   
def saveHistory(iHistoryDict, iSaveDir, iName='history', iKeyList = None):
    
    if iKeyList is None:
        wDict = iHistoryDict
    else:
        wDict = {wKey: iHistoryDict[wKey] for wKey in iKeyList}
        
    with open(os.path.join(iSaveDir, f'{iName}.csv'), 'w') as wFile:
        
        wKeyList = ['epoch'] + list(wDict.keys())
        wFile.write(','.join(wKeyList))
        wFile.write('\n')
        wValuesList = list(wDict.values())
        wValuesArray = np.array(wValuesList).T
        for i, wValues in enumerate(wValuesArray):
            wValuesStr = ','.join([str(i)]+[str(wVal) for wVal in wValues])
            wFile.write(wValuesStr)
            wFile.write('\n')

if sys.stdin and sys.stdin.isatty():
    gFromShell=True
    ROOT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
    gScriptName=os.path.splitext(os.path.basename(__file__))[0]
else:
    gFromShell=False
    ROOT_DIR = os.path.abspath(os.path.join(os.getcwd()))
    gScriptName='IDE'
	
gParentDir = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))
gDefaultTrainSubDir = os.path.join('data2019', 'synth', 'test_01_tr_7000_dim_448_sample_4_frac_1e+00_res_3_blend_True_clr_True_uint',)
gDefaultTrainDir = os.path.join(gParentDir, 'tfdataset', gDefaultTrainSubDir)
gDefaultValidSubDir = os.path.join('data4K', 'valid_real_448_res2')
gDefaultValidDir = os.path.join(gParentDir, 'tfdataset', gDefaultValidSubDir)

parser = argparse.ArgumentParser(
                    prog='Testing Data Augmentation',
                    description='Testing different combinations of augmentation pipelines for best generalizing behaviour',
                    epilog='Text at the bottom of help')

parser.add_argument('-t', '--mTrainDir', type=Path, nargs='+', default = gDefaultTrainDir, 
                    help='Path to directory for training data')

parser.add_argument('-v', '--mValidDir', type=Path, nargs='+', default = gDefaultValidDir, 
                    help='Path to directory for validation data')

parser.add_argument('-lp', '--mLoadCkpt', type=Path,
                    help='Path to load checkpoint')

parser.add_argument('-f', '--mSaveFolderName', type=str, 
                    help='Name of Subfolder to create for saving data')

parser.add_argument('-b', '--mBatchSize', default= 8, type=int, 
                    help='Training Batch size')

parser.add_argument('-s', '--mSampleSize', default=0, type=int, 
                    help='Amount of to subsample from original dataset')

parser.add_argument('-e', '--mEpochs', default=15, type=int, 
                    help='Amount of epochs to train for')

parser.add_argument('-lr', '--mLearningRate', default=1e-4, type=float, 
                    help='Amount of epochs to train for')

parser.add_argument('-n', '--mNbPts', default=-1, type=int, 
                    help='Amount of datapoints to simulate, if -1 will simulate original dataset size that we have sampled from')

parser.add_argument('-ag', '--mAugType', default=0, type=int, 
                    help='Augmentation protocol to choose from')

parser.add_argument('-dp', '--mDeeper', default=0, type=int, 
                    help='Depth of decoder model')# is lighter version #0 is the original #latest was 2

parser.add_argument('-res', '--mMapRes', default=3, type=int, 
                    help='Number of output map resolutions')

parser.add_argument('-lg', '--mLegacy', action='store_true',
                    help='Use old model output (i.e. does not slice final tensor output within model)')

parser.add_argument('-hf', '--mHistoryFolder', type=str, default = 'test', 
                    help='Name of Subfolder to create for saving data')

parser.add_argument('-ckpt', '--mCkptFreq', default=50, type=int, 
                    help='Checkpoint save frequency')

parser.add_argument('-svln', '--mSaveLastN', default=10, type=int, 
                    help='Save N Best models')

parser.add_argument('-prt', '--mSaveLastNPeriodicReset', default=0, type=int, 
                    help='Reset min val list every N epochs')

parser.add_argument('-ufr', '--mUnfreezeRate', nargs = '+', type=float, default = [300, 500],
                    help='Start epoch and rate at which to unfreeze backbone network when fine-tuning')

parser.add_argument('-lrs', '--mLRSched', nargs = '*', default=None, type=float, 
                    help='LR schedule as a list [epoch_1, LR_1, epoch_2, LR_2, etc...]')

parser.add_argument('-lvl', '--mLossLvlSched', nargs = '+', default=[0, 0, 100, 1, 200, 2, 300, -2], type=int, 
                    help='Loss level scheduling [epoch_1, losslvlflag_2, epoch_2, losslvlflag_2, etc...]')

parser.add_argument('-stp', '--mStopTrainBegin', default=100, type=int, 
                    help='When to begin examining losses for stopping training')

parser.add_argument('-lrp', '--mLRPatience', nargs = 2,  default=[100, 50], type=int, 
                    help='When to begin examining losses for stopping training')


def makeNewDirV2(iParDir, iFolderName, iKey, iIdx=0, iPrefix='test'):
    wPrefix = iPrefix +'_'+ f"{iIdx:02}"
    wNew=True
    for wFile in os.listdir(iParDir)[::-1]: #reverse it because it's usually ordered
                                            #so might as well search from bottom up
        try:
            if iKey == wFile.split('_')[2] and wPrefix == '_'.join(wFile.split('_')[0:2]):
                wNew=False
                break
        except IndexError:
            pass
        
    if wNew:
        return os.path.join(iParDir, "_".join([wPrefix, iKey, iFolderName]))
    else:
        return makeNewDirV2(iParDir, iFolderName, iKey, iIdx+1, iPrefix)

if __name__ == '__main__':

    wArgs= parser.parse_args()
    wTrainDir = wArgs.mTrainDir
    wValidDir = wArgs.mValidDir
    wDeeper = wArgs.mDeeper
    wRes= wArgs.mMapRes
    wLegacy = wArgs.mLegacy
    wBatchSize = wArgs.mBatchSize
    nEpochs = wArgs.mEpochs
    wSampleSize = wArgs.mSampleSize
    wNbPts = wArgs.mNbPts
    wAugType= wArgs.mAugType
    
    ChooseAugment = partial(chooseAugmentV2, iType=wAugType)
    wStartEpoch = 0
    wLoadCkpt = wArgs.mLoadCkpt
    wLastN = wArgs.mSaveLastN
    wPeriodicReset = wArgs.mSaveLastNPeriodicReset
    wCkptFreq = wArgs.mCkptFreq
    wStopTrainBegin = wArgs.mStopTrainBegin
    wLRBegin, wLRPatience = tuple(wArgs.mLRPatience)
    
    wSaveDir = makeNewDirV2(ROOT_DIR, f'ep_{nEpochs:04}_aug_{wAugType}', gScriptName, iPrefix=wArgs.mHistoryFolder)
    os.makedirs(wSaveDir, exist_ok=True)
    print('\nSave dir: %s'%wSaveDir)
    wArgLogName=gScriptName +'_args.csv'
    logArgs(wArgs, wSaveDir, wArgLogName)
    print('\nSaved argument Log to: %s'%wArgLogName)
    with open(os.path.join(wSaveDir, wArgLogName), 'a') as wFile:
        wFile.write('\n\n'+' '.join(sys.argv))
    
    
    wLRSched = wArgs.mLRSched
    wTFDataList = []
    for wDir in wTrainDir:
        wTFDataList.append(tf.data.Dataset.load(os.path.abspath(wDir)))
    # wTFData = tf.data.Dataset.load(os.path.abspath(wTrainDir))
    wTFData = wTFDataList[0]
    for wData in wTFDataList[1:]:
        wTFData = wTFData.concatenate(wData)
    
    wTotLen = len(wTFData)

    # wValidLoadSubDir = os.path.join('data4K', 'valid_real_448_res2')
    wValidTFDataList = []
    for wDir in wValidDir:
        wValidTFDataList.append(tf.data.Dataset.load(os.path.abspath(wDir)))
    wValidTFData = wValidTFDataList[0]
    for wData in wValidTFDataList[1:]:
        wValidTFData = wValidTFData.concatenate(wData)
    wImInfo, wMapInfo = wTFData.element_spec['images'], wTFData.element_spec['segmentation_masks']
    wImInputShape, wMapInputShape = wImInfo.shape, wMapInfo.shape
    wImInputDim, wMapInputDim = wImInputShape[:2], wMapInputShape[:2]    

    wModel = makeYoloTypeFlat(iShape=wImInputShape, iRes=wRes, iDeeper=wDeeper, iLegacy=wLegacy)
    wDecoderStartIdx = 175
    wBaseLayers, wDecoderLayers = wModel.layers[:wDecoderStartIdx], wModel.layers[wDecoderStartIdx:]
    for wLayer in wBaseLayers:
        wLayer.trainable=False
    for wLayer in wDecoderLayers:
        wLayer.trainable=True
    
    wLossWeightDict = {'Pos':tf.constant([5.]), 'Neg':tf.constant([25.]), 'Dice':tf.constant([1.])}
    wOutputSizes = [iLayerOutput.shape[1:3] for iLayerOutput in wModel.outputs]
    wResizeFunctionList = [tfResizeFunc(wOutput) for wOutput in wOutputSizes]

    
    wPlotBatchSize = 4
    wPlotData = tf.data.Dataset.from_tensor_slices(next(iter(wTFData.batch(wPlotBatchSize))))
    wPlotValidData = tf.data.Dataset.from_tensor_slices(next(iter(wValidTFData.batch(wPlotBatchSize))))
   
    wPlotFreq =10
    nBatches = math.ceil(len(wTFData)/wBatchSize)
    
    wPartial = partial(dictToTupleV4, iResizeFunctionList=wResizeFunctionList, iTrain=True, iPreprocess=True)
    if wSampleSize > 0:
        if wNbPts <0:
            wRepeat = len(wTFData)//wSampleSize
        else:
            wRepeat = wNbPts//wSampleSize    
        wTFData = wTFData.take(count=wSampleSize)
    
        wTFData = wTFData.repeat(wRepeat)
    
    wTFData = wTFData.shuffle(wTFData.cardinality(), reshuffle_each_iteration=True)
    wTFData = wTFData.batch(wBatchSize)
    wTFData = ChooseAugment(wTFData)
    wTFData = wTFData.map(wPartial, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    wPartialVal = partial(dictToTupleV4, iResizeFunctionList=wResizeFunctionList, iTrain=False, iPreprocess=True)
    wValidTFData = wValidTFData.batch(wBatchSize)
    wValidTFData = wValidTFData.map(wPartialVal, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    wPartialPlot = partial(dictToTupleV4, iResizeFunctionList=wResizeFunctionList, iTrain=True, iPreprocess=False)
    wPartialPlotVal = partial(dictToTupleV4, iResizeFunctionList=wResizeFunctionList, iTrain=False, iPreprocess=False)
    
    wPlotData = wPlotData.batch(wPlotBatchSize)
    wPlotData = ChooseAugment(wPlotData)
    wPlotData = wPlotData.map(wPartialPlot, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    wPlotValidData = wPlotValidData.batch(wPlotBatchSize).map(wPartialPlotVal, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    if wLRSched is not None:
        wLearningRateSchedDict = decodeParserSchedAsDict(wLRSched, iValKey=float)
        wLearningRate = {list(wLearningRateSchedDict.keys())[0]: list(wLearningRateSchedDict.values())[0]}
        wOptimizer = Adam(learning_rate=list(wLearningRate.values())[0], clipvalue=0.5)

    else:
        wLearningRateSchedDict = None
        wLearningRate = wArgs.mLearningRate
        wOptimizer = Adam(learning_rate=wLearningRate, clipvalue=0.5)
        
    wModel.compile(optimizer=wOptimizer, 
                   loss={
                             'slice_1': CustomLoss4([tf.constant([1.]), tf.constant([5.])]), 
                             'slice_2': CustomLoss4([tf.constant([1.]), tf.constant([5.])]), 
                             'slice_3': CustomLoss4([tf.constant([1.]), tf.constant([5.])])
                         }
                   )  
    
    wCheckpoint = Checkpoint(model=wModel,optimizer= wOptimizer)
        
    if wLoadCkpt is not None:
        wCheckpoint.read(wLoadCkpt)
        wCkptName=str(wLoadCkpt).split('cp-')[-1].split('.')[0]
        wStartEpoch = int(wCkptName.split('_')[0])
        print(f"\nLoaded checkpoint cp-{wCkptName}. Starting from epoch: {wStartEpoch}\n")
    
    wCallBacks = []
    wSchedulerCallback = LayerUnfreezeCallback(wDecoderStartIdx)
    wBaseModelLayerNames = [wLayer.name for wLayer in wBaseLayers[::-1] if not('batch' in str(type(wLayer)).lower()  and 'normalization' in str(type(wLayer)).lower())]
    wSchedulerCallback.setGradualUnfreezeArgs([300, len(wBaseModelLayerNames), wBaseModelLayerNames])
    wSchedulerCallback.setLossLvlScheduleFromFlagListArgs([[0], [-2]])     
    
    wCallBacks.append(wSchedulerCallback)

    wLRSchedule = CustomLRSchedulerFunction()
    
    if wLearningRateSchedDict is not None:
        wLRSchedule.setLRScheduleFromDict(wLearningRateSchedDict)
    else:
        wLRSchedule.setLRScheduleFromDict({wStartEpoch: wLearningRate})
    
    wCallBacks.append(CustomLearningRateScheduler(wLRSchedule, iBegin = wLRBegin, iPatience = wLRPatience))

    
    wSaveEveryNCkpt = SaveEveryN(wCheckpoint, wSaveDir, iFreq=wCkptFreq)
    wCallBacks.append(wSaveEveryNCkpt)
    wSaveLastNCkpt = SaveLastN(wCheckpoint, wSaveDir, iLastN=wLastN, iPeriodicReset=wPeriodicReset)
    wSaveLastNCkpt.resetMinValLossList([float('inf')], ['inf'])
    wCallBacks.append(wSaveLastNCkpt)
    wLRSchedule.setSaveLastNCallBack(wSaveLastNCkpt)
    wCallBacks.append(LogEveryN('loss_log.txt', wSaveDir, 5))
    wCallBacks.append(StopTraining(wStopTrainBegin, 10., 10., iPatience = 5))
    wCallBacks.append(SleepCallBack(2.))
    
    plot_model(wModel, os.path.join(wSaveDir, 'full_model.png'), show_shapes = True, show_layer_names=True)
    save_model_summary(wSaveDir, wModel)
    
    wHistory= wModel.fit(wTFData, epochs=nEpochs, initial_epoch=wStartEpoch, validation_data=wValidTFData, callbacks = wCallBacks)
    saveHistory(wHistory.history, wSaveDir,f'history_aug_{wAugType:02}', iKeyList=None)#['loss', 'val_loss'])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
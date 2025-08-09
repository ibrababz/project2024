#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 20:51:43 2025

@author: ibabi
"""

# from dataLoad import loadDataFilesAsObjectsMP
import os
import sys
import numpy as np
# os.environ["KERAS_BACKEND"] = "tensorflow"
import math
# import keras_cv
import tensorflow as tf
# from tfAugmentations import Augments, getAugments, getAffine, getNoise
from tfMapUtils import tfFlatMapBatch, tfFlatMap, tfAct3DBatch, tfScale3DBatch, tfFlatMapBatchEager, tfAct3DBatchEager
from tfMapUtils import tfAct3D, tfScale3D
from models import makeYoloType, makeYoloTypeFlat
from models import save_model_summary
from timeit import time
from customLoss import CustomLoss, CustomLoss2, CustomLoss3, CustomLoss4, tfResizeFunc
from customCallbacks import PlotCallBack, LayerUnfreezeCallback, CustomLRSchedulerFunction
from customCallbacks import SaveEveryN, SaveLastN
from customCallbacks import SleepCallBack, LogEveryN, StopTraining
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import plot_model
from tensorflow.train import Checkpoint

from tfAugmentations import chooseAugmentV2

from functools import partial

from dataLoad import show_batch
from pathlib import Path
from parsingFunctions import decodeParserSched, logArgs
from parsingFunctions import decodeParserSchedAsDict


if sys.stdin and sys.stdin.isatty():
    gFromShell=True
    ROOT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
else:
    gFromShell=False
    ROOT_DIR = os.path.abspath(os.path.join(os.getcwd()))

def dictToTuple(x, iOutputNum=3):
    wIm, wMap = x['images'], x['segmentation_masks']
    return wIm, (wMap,)*iOutputNum

def _dictToTuple(x):
    return x['images'], x['segmentation_masks']

def dictToTupleV2(x, iResizeFunctionList):
    wIm, wMap = x['images'], x['segmentation_masks']
    wMapTuple = tuple(tfFlatMapBatch(tfAct3DBatch(wResize(wMap))) for wResize in wResizeFunctionList)
    return wIm, wMapTuple

# wNoiseAugments = getNoise()

# def addNoise(wIm, iTrain=True):
#     return wNoiseAugments(wIm, training=iTrain)#*255.

    
# @tf.function
# def dictToTupleV3(x, iResizeFunctionList, iTrain=True, iPreprocess=True, iRate=0.5):
#     wIm, wMap = x['images'], x['segmentation_masks']
#     wIm = tf.cast(wIm, dtype=tf.float32)
#     wIm= addNoise(wIm/255., iTrain)  #dividing before preprocessing is incorrect but it was done when previously training in TrainerClass.py
#     if iPreprocess:
#         wIm = preprocess_input(wIm[...,::-1])

#     wMapTuple = tuple(tfFlatMapBatch(tfScale3DBatch(wResize(wMap))) for wResize in wResizeFunctionList)
#     return wIm, wMapTuple #CHANGE THIS BACK MAYBE TO PREPROCESSING IF (/255.0)

# def mapNoise(x, iTrain=True):
#     wIm, wMap = x['images'], x['segmentation_masks']
#     wIm = tf.cast(wIm, dtype=tf.float32)
#     wIm= addNoise(wIm/255., iTrain)*255. #dividing before preprocessing is incorrect but it was done when previously training in TrainerClass.py
#     return {'images':wIm, 'segmentation_masks': wMap}


def dictToTupleV4(x, iResizeFunctionList, iTrain=True, iPreprocess=True):
    wIm, wMap = x['images'], x['segmentation_masks']#dividing before preprocessing is incorrect but it was done when previously training in TrainerClass.py
    wIm = tf.cast(wIm, dtype=tf.float32)#/255.
    if iPreprocess:
        wIm = preprocess_input(wIm[...,::-1])
    wMapTuple = tuple(tfFlatMapBatch(tfScale3DBatch(wResize(wMap))) for wResize in wResizeFunctionList)
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
            
        
    

import matplotlib.pyplot as plt
import argparse

if sys.stdin and sys.stdin.isatty():
    gFromShell=True
    ROOT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
    gScriptName=os.path.splitext(os.path.basename(__file__))[0]
else:
    gFromShell=False
    ROOT_DIR = os.path.abspath(os.path.join(os.getcwd()))
    gScriptName='IDE'
    

# ROOT_DIR = os.path.abspath(os.path.join(os.getcwd()))
gParentDir = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))
gDefaultTrainSubDir = os.path.join('data4K', 'test_real_448_res2')
gDefaultTrainDir = os.path.join(gParentDir, 'tfdataset', gDefaultTrainSubDir)
gDefaultValidSubDir = os.path.join('data4K', 'valid_real_448_res2')
gDefaultValidDir = os.path.join(gParentDir, 'tfdataset', gDefaultValidSubDir)

parser = argparse.ArgumentParser(
                    prog='Testing Data Augmentation',
                    description='Testing different combinations of augmentation pipelines for best generalizing behaviour',
                    epilog='Text at the bottom of help')
if gFromShell:
    parser.add_argument('mModelCkpt', type=Path,
                        help='Path to load checkpoint')
    
parser.add_argument('-tp', '--mModelCkptType', nargs = '+', type=str, default = ['cp', 'list'],
                    help='Type of checkpoint to look for')

parser.add_argument('-ex', '--mModelCkptTypeExclude', nargs = '+', type=str,
                    help='Type of checkpoint to look for')

parser.add_argument('-gt', '--mModelCkptGreaterThan', type=int, default = 0,
                    help='Type of checkpoint to look for')

parser.add_argument('-t', '--mTrainDir', type=Path, default = gDefaultTrainDir, 
                    help='Path to directory for training data')

parser.add_argument('-v', '--mValidDir', type=Path, default = gDefaultValidDir, 
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

parser.add_argument('-ufr', '--mUnfreezeRate', nargs = '+', type=float, default = [300, 500],
                    help='Start epoch and rate at which to unfreeze backbone network when fine-tuning')


parser.add_argument('-lrs', '--mLRSched', nargs = '*', default=None, type=float, 
                    help='LR schedule as a list [epoch_1, LR_1, epoch_2, LR_2, etc...]')

parser.add_argument('-lvl', '--mLossLvlSched', nargs = '+', default=[0, 0, 100, 1, 200, 2, 300, -2], type=int, 
                    help='Loss level scheduling [epoch_1, losslvlflag_2, epoch_2, losslvlflag_2, etc...]')



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

import cv2 as cv
        
def threshold_list(img_list, thresh_val, max_val, flag = cv.THRESH_TOZERO):

    return [cv.threshold(im, thresh_val, max_val, flag)[1][...,None] for im in img_list]

def refine_thresh(pred, thresh):
    thresh = tf.where(thresh > 0., 1., 0.)
    thresh = cv.dilate(np.float32(thresh[...,0]), np.ones((3,3)))[...,None]
    return pred*thresh

def refine_thresh_list(pred_list, thresh_list):

    return [refine_thresh(pred, thresh) for pred, thresh in zip(pred_list, thresh_list)]

def get_cntrs_list(img_list, ret_flag = cv.RETR_TREE, app_flag = cv.CHAIN_APPROX_SIMPLE, thresh_flag = cv.THRESH_BINARY):

    return [cv.findContours(np.uint8(cv.threshold(im, 0.01, 1., thresh_flag)[1]), ret_flag, app_flag)[0] for im in img_list]

def act_from_pred_list(pred_batch):
    pred_cntrs_batch = get_cntrs_list(pred_batch)
    pred_drawn_expanded_list = draw_cntrs_exp_list(pred_batch.copy(), pred_cntrs_batch, thickness = -1)
    pred_exp_list = expand_pred_list(pred_batch, pred_drawn_expanded_list)
    pred_act_list = [tfAct3D(pred_exp) for pred_exp in pred_exp_list]
    flat_pred_act_list = [tfFlatMap(pred_act) for pred_act in pred_act_list]
    return flat_pred_act_list

def  draw_cntrs_exp_list(img_list_copy, cntrs_list, colour = 1, thickness= -1):
    im_list_cntrs = []
    for im, cntrs in zip(img_list_copy, cntrs_list):
        layers = []
        for i, cnt in zip(range(len(cntrs)),cntrs):
            blank = np.zeros(im.shape, dtype = im.dtype)
            layers.append(cv.drawContours(blank, cntrs, i, colour, thickness))
        
        if len(layers)>0:
            expanded = np.concatenate(layers, axis = -1)
        else:
            expanded = np.zeros(im.shape, dtype = im.dtype)
        
        im_list_cntrs.append(expanded)
    return im_list_cntrs
    
def expand_pred_list(pred_batch, pred_drawn_expanded_list):
    return [pred*exp for pred, exp in zip(pred_batch, pred_drawn_expanded_list)]

def find_cent_on_im(contours):
    centers_on_image = []
    for c in contours:
    
        M = cv.moments(c)
        if  M["m00"] != 0: 
            cX = round(M["m10"] / M["m00"])
            cY = round(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
  
        centers_on_image.append([cX, cY])

    return centers_on_image

def find_cent_list(contours_on_image_list):
    return [find_cent_on_im(cntrs) for cntrs in contours_on_image_list]

def draw_cent_on_im(im, centers_on_image, color = (1.,1.,1.)):

    if len(im.shape)<3:
        color = color[0]
    elif len(im.shape) == 3:
        if im.shape[2] == 1:
            color = color[0]
    im_copy = im.copy()        
    for c in centers_on_image:
        cX, cY = c[0],  c[1]
        
        im_copy[cY, cX] = color
    
    return im_copy

def draw_cent_on_im_list(centers_on_im_list, im_list = None, shape = None):
    drawn_cent_list = [] 
    for i in range(len(centers_on_im_list)):
        if im_list is not None:
            im = im_list[i].copy()
        else:
            im = np.zeros(shape)
            
        drawn = draw_cent_on_im(im, centers_on_im_list[i])
        drawn_cent_list.append(drawn)
    return drawn_cent_list



def compute_pred_centroids(iThreshList):
    pred_cntrs_batch1 = get_cntrs_list(iThreshList)
    centers_on_im_list1 = find_cent_list(pred_cntrs_batch1)
    return draw_cent_on_im_list(centers_on_im_list1, None, iThreshList[0].shape)

def TFmetrics(truth_map_i, pred_map_i):
    TPi = np.logical_and(truth_map_i, pred_map_i)*1.
    
    FPi = np.greater(pred_map_i,0.)*1. - TPi
    
    TNi = np.logical_and(np.logical_not(truth_map_i), np.logical_not(pred_map_i))*1.
    
    FNi = np.logical_not(pred_map_i)*1. - TNi
    
    
    TPi_no = np.sum(TPi)
    FPi_no = np.sum(FPi)
    
    TNi_no = np.sum(TNi)
    FNi_no = np.sum(FNi)
    
    return TPi_no, FPi_no, TNi_no, FNi_no

def Metrics_from_TF(TPi_no, FPi_no, TNi_no, FNi_no):
    if TPi_no == 0.:
        Prec = 0.
        Reca = 0.
        F1 = 0.
        MIoU = 0.
    else:
        Prec = TPi_no/(TPi_no + FPi_no)
        Reca = TPi_no/(TPi_no + FNi_no)
        F1 = 2.*Prec*Reca/(Prec+Reca)
        MIoU = TPi_no/(TPi_no + FPi_no + FNi_no)

    
    return Prec, Reca, F1, MIoU

def Metrics_from_TF_batch(TPi_list, FPi_list, TNi_list, FNi_list):
    Prec_list = []
    Reca_list = []
    F1_list = []
    MIoU_list = []
    
    for TPi_no, FPi_no, TNi_no, FNi_no in zip(TPi_list, FPi_list, TNi_list, FNi_list):
        Prec, Reca, F1, MIoU = Metrics_from_TF(TPi_no, FPi_no, TNi_no, FNi_no)
        Prec_list.append(np.round(Prec, 2))
        Reca_list.append(np.round(Reca, 2))
        F1_list.append(np.round(F1, 2))
        MIoU_list.append(np.round(MIoU, 2))

    
    return Prec_list, Reca_list, F1_list, MIoU_list

def TF_Metrics_from_batch(truth_batch1, pred_batch1):
    TPi_list = []
    FPi_list = []
    TNi_list = []
    FNi_list = []

    for act_i, coi_i in zip(truth_batch1, pred_batch1):
        
        TPi_no, FPi_no, TNi_no, FNi_no = TFmetrics(act_i,coi_i) 
        
        TPi_list.append(TPi_no)
        FPi_list.append(FPi_no)
        TNi_list.append(TNi_no)
        FNi_list.append(FNi_no)

    return TPi_list, FPi_list, TNi_list, FNi_list

def compute_metrics_batch(iTruth, iPred):

    # if iFlag == 'maps':
    #     wTruth = self.getFlatMaps(iIdx)            
    #     wPred = self.getThreshPreds(iIdx)
    # else:
    #     wTruth = self.getFlatActs(iIdx)
    #     if iFlag == 'act':
    #         wPred = self.getPredActs(iIdx)
    #     elif iFlag == 'cent':
    #         wPred = self.getPredCents(iIdx)
 
    wTPBatch, wFPBatch, wTNBatch, wFNBatch = TF_Metrics_from_batch(iTruth, iPred)
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
    wSaveDir = makeNewDirV2(ROOT_DIR, f'ep_{nEpochs:04}_type_{wAugType}', 'aug', iPrefix='_'.join([wArgs.mHistoryFolder, gScriptName]))
    ChooseAugment = partial(chooseAugmentV2, iType=wAugType)
    wModelCkptType = wArgs.mModelCkptType
    wModelCkptGreaterThan = wArgs.mModelCkptGreaterThan
    wModelCkptTypeExclude = wArgs.mModelCkptTypeExclude
    wStartEpoch = 0
    if gFromShell:
        wModelCkpt = wArgs.mModelCkpt
    wLoadCkpt = wArgs.mLoadCkpt
    
    os.makedirs(wSaveDir, exist_ok=True)
    print('\nSave dir: %s'%wSaveDir)
    wArgLogName=gScriptName +'_args.csv'
    logArgs(wArgs, wSaveDir, wArgLogName)
    print('\nSaved argument Log to: %s'%wArgLogName)
        
    
    
    if wStartEpoch:
        print("LOADING")
    
    wLRSched = wArgs.mLRSched
    
    if wLRSched is not None:
        wLearningRateSchedDict = decodeParserSchedAsDict(wLRSched, iValKey=float)
        wLearningRate = {list(wLearningRateSchedDict.keys())[0]: list(wLearningRateSchedDict.values())[0]}
    else:
        wLearningRateSchedDict = None
        wLearningRate = wArgs.mLearningRate

    def mapNoiseV2(x):
        wIm, wMap = x['images'], x['segmentation_masks']
        wIm = tf.cast(wIm, dtype=tf.float32)
        # wIm= wNoise(wIm/255., training=True)*255. #dividing before preprocessing is incorrect but it was done when previously training in TrainerClass.py
        return {'images':wIm, 'segmentation_masks': wMap}
    

    wTFData = tf.data.Dataset.load(os.path.abspath(wTrainDir))
    wTotLen = len(wTFData)

    wValidLoadSubDir = os.path.join('data4K', 'valid_real_448_res2')
    wValidTFData = tf.data.Dataset.load(os.path.abspath(wValidDir))
    # wTFData, wValidTFData = tf.keras.utils.split_dataset(wTFData, left_size= 0.8)
    wImInfo, wMapInfo = wTFData.element_spec['images'], wTFData.element_spec['segmentation_masks']
    wImInputShape, wMapInputShape = wImInfo.shape, wMapInfo.shape
    wImInputDim, wMapInputDim = wImInputShape[:2], wMapInputShape[:2]    
    
    # wDeeper= 0# is lighter version #0 is the original #latest was 2
    # wModel = makeYoloType(iShape=wImInputShape, iRes=3, iDeeper=wDeeper, iLegacy=False)
    
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
    
    # wTFData = wTFData.shuffle(wTFData.cardinality(), reshuffle_each_iteration=True)
    wTFData = wTFData.batch(1)
    # wTFData = ChooseAugment(wTFData)
    wTFData = wTFData.map(wPartial, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    # wPartialVal = partial(dictToTupleV4, iResizeFunctionList=wResizeFunctionList, iTrain=False, iPreprocess=True)
    # wValidTFData = wValidTFData.batch(wBatchSize)
    # wValidTFData = wValidTFData.map(wPartialVal, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    # wPartialPlot = partial(dictToTupleV4, iResizeFunctionList=wResizeFunctionList, iTrain=True, iPreprocess=False)
    # wPartialPlotVal = partial(dictToTupleV4, iResizeFunctionList=wResizeFunctionList, iTrain=False, iPreprocess=False)
    
    # wPlotData = wPlotData.batch(wPlotBatchSize)
    # wPlotData = ChooseAugment(wPlotData)
    # wPlotData = wPlotData.map(wPartialPlot, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    # wPlotValidData = wPlotValidData.batch(wPlotBatchSize).map(wPartialPlotVal, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    
    # wCallBacks = []
    # wSchedulerCallback = LayerUnfreezeCallback(wDecoderStartIdx)
    # wBaseModelLayerNames = [wLayer.name for wLayer in wBaseLayers[::-1] if not('batch' in str(type(wLayer)).lower()  and 'normalization' in str(type(wLayer)).lower())]
    # wSchedulerCallback.setGradualUnfreezeArgs([wStartEpoch, len(wBaseModelLayerNames), wBaseModelLayerNames])
    # wSchedulerCallback.setLossLvlScheduleFromFlagListArgs([[0], [-2]])     
    
    # wCallBacks.append(wSchedulerCallback)

    # wLRSchedule = CustomLRSchedulerFunction()
    # if wLearningRateSchedDict is not None:
    #     wLRSchedule.setLRScheduleFromDict(wLearningRateSchedDict)
    #     wOptimizer = Adam(learning_rate=list(wLearningRate.values())[0], clipvalue=0.5)
    #     wCallBacks.append(LearningRateScheduler(wLRSchedule))
    # else:
    #     wLRSchedule.setLRScheduleFromDict({wStartEpoch: wLearningRate})
    wOptimizer = Adam(learning_rate=1e-4, clipvalue=0.5)
    #     wCallBacks.append(LearningRateScheduler(wLRSchedule))


    wModel.compile(optimizer=wOptimizer, 
                   loss={
                             'slice_1': CustomLoss4([tf.constant([1.]), tf.constant([5.])]), 
                             'slice_2': CustomLoss4([tf.constant([1.]), tf.constant([5.])]), 
                             'slice_3': CustomLoss4([tf.constant([1.]), tf.constant([5.])])
                         }
                   )  
    if not gFromShell:
        wModelCkpt = os.path.join('.', 'test_tfFineTuneNew_00_aug_ep_0500_type_5', 'cp-0447_list_2p008')
        
    wListDir = os.listdir(wModelCkpt)
    wTemp = [wFile.split('.')[0] for wFile in wListDir if all(wType in wFile for wType in wModelCkptType) and int(wFile.split('.')[0].split('_')[0].split('cp-')[1]) > wModelCkptGreaterThan]
    if wModelCkptTypeExclude is not None:
        wTemp = [wFile for wFile in wTemp if not any(wType in wFile for wType in wModelCkptTypeExclude)]
    wModels = []
    for wFile in wTemp:
        if wFile not in wModels:
            wModels.append(wFile)
    
    wCheckpoint = Checkpoint(model=wModel,optimizer= wOptimizer)
    for wFile in wModels:
        wCheckpoint.read(os.path.join(wModelCkpt, wFile))
        wCkptName=str(wModelCkpt).split('cp-')[-1].split('.')[0]
        print(f"\nLoaded model {wFile}\n")
            
        if wLoadCkpt is not None:
            wCheckpoint.read(wLoadCkpt)
            wCkptName=str(wLoadCkpt).split('cp-')[-1].split('.')[0]
            wStartEpoch = int(wCkptName.split('_')[0])
            print(f"\nLoaded checkpoint cp-{wCkptName}. Starting from epoch: {wStartEpoch}\n")
            
        # wSaveEveryNCkpt = SaveEveryN(wCheckpoint, wSaveDir, iFreq=50)
        # wCallBacks.append(wSaveEveryNCkpt)
        # wLastN = 10
        # wSaveLastNCkpt = SaveLastN(wCheckpoint, wSaveDir, iLastN=wLastN)
        # wSaveLastNCkpt.resetMinValLossList([float('inf')], ['inf'])
        # wCallBacks.append(wSaveLastNCkpt)
        # wCallBacks.append(LogEveryN('loss_log.txt', wSaveDir, 5))
        # wCallBacks.append(StopTraining(100, 10., 10., iPatience = 5))
        # wCallBacks.append(SleepCallBack())
        plot_model(wModel, os.path.join(wSaveDir, 'full_model.png'), show_shapes = True, show_layer_names=True)
        save_model_summary(wSaveDir, wModel)
        
        wPredList= wModel.predict(wTFData)      
        # wResIdx = 2
        # wSample = wPredList[wResIdx][0][...,0]
        # plt.imshow(wSample)
        # plt.show()
        # plt.close()
        # saveHistory(wHistory.history, wSaveDir,f'history_aug_type_{wAugType:02}', iKeyList=None)#['loss', 'val_loss'])
        

            
        wDataNumpy = list(wTFData.as_numpy_iterator())
        wImages = [wData[0] for wData in wDataNumpy]
        wNRes = len(wDataNumpy[0][1])
        wMapLists = [[wData[1][i][0] for wData in wDataNumpy] for i in range(wNRes)] 
    
        # print(type(wDataNumpy))
        # print(len(wDataNumpy))
        # print(type(wDataNumpy[0]))
        # print(len(wDataNumpy[0]))
        
        # for wMaps in wMapLists:
        #     print(type(wMaps))
        #     print(len(wMaps))
        #     print(wMaps[0].shape)
        wBuffer = 2
        wHeader = ['Thresh ', 'Prec', 'Reca', 'F1', 'MIoU']
        
        
        wHeaderMaxLen = len(max(wHeader, key=len))+wBuffer
        wHeaderValList = ['{:^'+str(wHeaderMaxLen)+'}' for wHead in wHeader]
        wStrValList = ['{:^'+str(wHeaderMaxLen)+'.4f}']+['{:^'+str(wHeaderMaxLen)+'.2f}' for wHead in wHeader[1:]]
        wHeader = ' '.join(wHeaderValList).format(*wHeader)
        
        wHeaderLen = len(wHeader)
    
        wHeader1List, wHeader2List =[], []
        wRowListList = []
        for wResIdx in range(wNRes):
            wResolutionStr = f"Resolution: {wResIdx}"
            wHeader1List.append(f'{wResolutionStr:^{wHeaderLen}}')
            wHeader2List.append(wHeader)    
            wTruth = wMapLists[wResIdx]
            wRowList = []
            wStep = 0.00625#/2
            for thresh_val in np.arange(0.9, 1. - wStep, wStep):
                wThresh = threshold_list(list(wPredList[wResIdx]), thresh_val, 1.)
                wThresh = refine_thresh_list(list(wPredList[wResIdx]).copy(), wThresh.copy())
                wAct = act_from_pred_list(wThresh)
                wCent = compute_pred_centroids(wThresh)
                TP, FP, TN, FN = TF_Metrics_from_batch(np.array(wTruth), np.array(wThresh))
                Prec, Recall, F1, MIoU = Metrics_from_TF_batch(TP, FP, TN, FN)
                wValueList = [thresh_val, np.mean(Prec)*100, np.mean(Recall)*100, np.mean(F1)*100, np.mean(MIoU)*100]
                wRowList.append('|'.join(wStrValList).format(*wValueList))
            wRowListList.append(wRowList)
        wHeader1 = '  '.join(wHeader1List)
        wHeader2 = '  '.join(wHeader2List)
        
        wTableRows = []
        for j in range(len(wRowListList[0])):
            wCurrentRow = []
            for i in range(len(wRowListList)):
                wCurrentRow.append(wRowListList[i][j])
            wTableRows.append('  '.join(wCurrentRow))
        wTableRows.append('\n')
        print(f"\n\n{wFile:^{len(wHeader1)}}")
        print(wHeader1)
        print(wHeader2)
        for wRow in wTableRows:
            print(wRow)

            
            

            # print(f"shape act: {wAct[0].shape}, shape cent: {wCent[0].shape}")
    
        
        
        # plt.imshow(wThresh[0])
        # plt.show()
        # plt.imshow(wAct[0])
        # plt.show()
        # plt.imshow(wCent[0])
        # plt.show()
        # plt.close()
        
        
        
        
        
        
        
        
        
        
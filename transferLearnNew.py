# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:09:29 2024

@author: i_bab
"""
import tensorflow as tf
import os
import numpy as np

from dataLoad import loadDataFilesAsObjects, generate_batch, makeNewDirV2
from models import makeYoloType
from tensorflow.keras.utils import plot_model
from models import save_model_summary
from TrainerClass import ModelTrainer, ModelTransLearn
from augment_utils import chooseAugments
import sys
import argparse
from pathlib import PurePath
if sys.stdin and sys.stdin.isatty():
    gFromShell=True
    ROOT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
else:
    gFromShell=False
    ROOT_DIR = os.path.abspath(os.path.join(os.getcwd()))
    
print("ROOT DIR: %s"%ROOT_DIR)
#%%


gParentDir = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))


def getArguments():
    parser = argparse.ArgumentParser(
                        prog='Synthetic Dandelion Image Generator',
                        description='Generate synthetic training and validation data sets',
                        epilog='Text at the bottom of help')
    
    parser.add_argument('-b', '--mBatchSize', nargs = '+', default= [0, 32, 1000, 16], type=int, 
                        help='Training Batch size')
    
    parser.add_argument('-e', '--mNoEpochs', default=600, type=int, 
                        help='Amount of validation data to generate')
    
    parser.add_argument('-s', '--mStart', default=0, type=int, 
                        help='Amount of validation data to generate')
    
    parser.add_argument('-m', '--mModelFlag', default='resnet',
                        help='Backbone model')
    
    parser.add_argument('-t', '--mTrainDir', default=os.path.join(*[gParentDir, "data4k", "train_real_448_res2"]),
                        help='Full path training data folder')
     
    parser.add_argument('-v', '--mValidDir', default=os.path.join(*[gParentDir, "data4k", "valid_real_448_res2"]),
                        help='Full path validation data folder')
    
    parser.add_argument('-g', '--mTransferLearnLoadPath', help='Full path to pre-trained model for transfer learning checkpoint')
    
    parser.add_argument('-n', '--mNorm', default=255., type=float, 
                        help='Normalize input pixels by')
    
    parser.add_argument('-r', '--mMapRes', default=3, type=int, 
                        help='Number of output map resolutions')
    
    parser.add_argument('-p', '--mPlotFreq', default=25, type=int, 
                        help='Output plot frequency')
    
    parser.add_argument('-c', '--mCkptFreq', default=25, type=int, 
                        help='Checkpoint save frequency')
    
    parser.add_argument('-a', '--mPlotAll', default=1, type=int, 
                        help='Plot all resolutions 1 or only active loss level 0')
    
    parser.add_argument('-f', '--mUnfreezeRate', default=500, type=float, 
                        help='Rate by which to unfreeze backbone network when fine-tuning')

    parser.add_argument('-i', '--mLRSched', nargs = '+', default=[0, 1e-4, 301, 1e-5, 500, 1e-6, 549, 1e-7], type=float, 
                        help='LR schedule as a list [epoch_1, LR_1, epoch_2, LR_2, etc...]')
    
    parser.add_argument('-x', '--mSaveLastN', default=3, type=int, 
                        help='Save N Best models')
    
    parser.add_argument('-z', '--mLogFreq', default=5, type=int, 
                        help='Freq at which to log to output file')
    
    parser.add_argument('-y', '--mWithPath', default=1, type=int, 
                        help='Log data names with path 0 or 1 (t/f)')
    
    parser.add_argument('-u', '--mCkpt', default=None,
                        help='Full path checkpoint file without extension')
    
    parser.add_argument('-w', '--mAugments', default=1, type=int, 
                        help='type of augment to use')
    
    parser.add_argument('-q', '--mLossLvlSched', nargs = '+', default=[0, 0, 100, 1, 200, 2, 300, -3], type=int, 
                        help='Loss level scheduling [epoch_1, losslvlflag_2, epoch_2, losslvlflag_2, etc...]')
    
    parser.add_argument('-k', '--mDepthKernelList', nargs = '+', default=[128, 1, 256, 3, 256, 1, 256, 3, 128, 1], type=int, 
                        help='Loss level scheduling [epoch_1, losslvlflag_2, epoch_2, losslvlflag_2, etc...]')
    
       
    return parser

def decodeParserSched(iList, iEpKey=int, iValKey= int):
    oEpList, oValList = [], []
    for i in range(len(iList)//2):
        oEpList.append(iEpKey(iList[2*i]))
        oValList.append(iValKey(iList[2*i+1]))
    return oEpList, oValList
#%%
if __name__ =='__main__':
    
    if gFromShell:
        print('From Shell!')
    physical_devices = tf.config.list_physical_devices('GPU')
    
    wArgs = getArguments().parse_args()
    wNorm = wArgs.mNorm
    iSrcPath = wArgs.mTrainDir
    wRes = wArgs.mMapRes
    iValidSrcPath = wArgs.mValidDir
    wModelFlag = wArgs.mModelFlag
    wStartEpoch, wEndEpoch = wArgs.mStart, wArgs.mNoEpochs
    wEpochs = wStartEpoch, wEndEpoch
    wBatchEpochList, wBatchSizeList = decodeParserSched(wArgs.mBatchSize)
    wPlotFreq = wArgs.mPlotFreq
    wPlotAll = bool(wArgs.mPlotAll)
    wEpList, wLRList = decodeParserSched(wArgs.mLRSched, iValKey=float) 
    wCheckPointFreq = wArgs.mCkptFreq
    wUnfreezeRate = wArgs.mUnfreezeRate
    wLastN = wArgs.mSaveLastN
    wLogFreq = wArgs.mLogFreq
    wWithPath = bool(wArgs.mWithPath)
    wAugments = wArgs.mAugments
    wLossLvlEpList, wLossLvlFlagList = decodeParserSched(wArgs.mLossLvlSched)
    wTransferLearnLoadPath = wArgs.mTransferLearnLoadPath
    wDepthList, wKernelList = decodeParserSched(wArgs.mDepthKernelList)
    
#%%
    print('\nLoading Training Data')
    wDataObjectList = loadDataFilesAsObjects(iSrcPath)
    print('\nDone!')
# #%%
#     from helper import show_wait
#     for wDataObj in wDataObjectList[:5]:
#         show_wait(wDataObj.getImage(), 2)
        
#%%  
    print('\nLoading Validation Data')
    wValidDataObjectList = loadDataFilesAsObjects(iValidSrcPath)        
    print('\nDone!')
#%%
    wModel = makeYoloType(wDataObjectList[0].getShape(), wModelFlag, wRes)
    wOptimizer = tf.keras.optimizers.Adam(learning_rate= wLRList[0], clipnorm=1., clipvalue=0.5) 
    
#%%
    wSaveFolder = "ep_{}-{}_lr_{:.0e}".format(wEpochs[0], wEpochs[1], wLRList[0])
    wSaveDir = makeNewDirV2(ROOT_DIR, wSaveFolder, wModelFlag, 0)
    os.makedirs(wSaveDir, exist_ok=True)
    print(wSaveDir)

#%%
    plot_model(wModel.layers[-1], os.path.join(wSaveDir, 'top_model.png'), show_shapes = True)
    plot_model(wModel, os.path.join(wSaveDir, 'wModel.png'), show_shapes = True)
    save_model_summary(wSaveDir, wModel.layers[-1])
    save_model_summary(wSaveDir, wModel)
    
    
#%% Fresh Start
    
    wTrainer = ModelTransLearn(wModel, wOptimizer)
    wEncoderIdxList, wDecoderNameList=[-2, 142, 80], ['top_15', 'top_27', 'top_37']
    if wArgs.mCkpt is None:
        wTrainer.setTransferLearnLoadPath(wTransferLearnLoadPath)
        wTrainer.loadTransferLearn()
        wTrainer.removeClassificationLayers(wEncoderIdxList, wDecoderNameList)
        wTrainer.addTransferLearnLayers(wEncoderIdxList, wDecoderNameList, wDepthList, wKernelList)
        wTrainer.freezeBackBone()
    else:
        wTrainer.removeClassificationLayers(wEncoderIdxList, wDecoderNameList)
        wTrainer.addTransferLearnLayers(wEncoderIdxList, wDecoderNameList, wDepthList, wKernelList)
        wLoadDir = os.path.abspath(os.path.join(wArgs.mCkpt, os.pardir))
        wTrainer.setLoadDir(wLoadDir)
        wCkpt = PurePath(wArgs.mCkpt).parts[-1].split('.')[0]
        wTrainer.freezeBackBone()
        wTrainer.loadFromCkpt(wCkpt)
        wStart = int(wCkpt.split('_')[0])+1
        print("Automatically starting from Epoch: %s"%wStart)
        wTrainer.setLossLvlScheduleFromDict({})
        wTrainer.setLayerFreezeScheduleFromDict({})
        wTrainer.setLRSchedFromDict({})
        wTrainer.setBatchSizeScheduleFromDict({})
    
    wTrainer.setPlotFreq(iPlotFreq=wPlotFreq, iPlotAll=wPlotAll)        
    
    for wEp, wLR in zip(wEpList, wLRList):
        wTrainer.setLRSched(wEp, wLR)
    
    wTrainer.setDecoderResolutionsDict()
    wTrainer.setLossLvlScheduleFromFlagList(wLossLvlEpList, wLossLvlFlagList)    
    wTrainer.setLossDict(iLossDict = {'Pos': 5., 'Neg': 5.*5., 'Dice': 1.})
    # wTrainer.setLossDict(iLossDict = {'Pos': 5., 'Neg': 5.*5., 'Dice': 1., 'Cart':5.*5.})
    wBaseModelLayers = [wLayer.name for wLayer in wTrainer.getModel().layers[::-1][1:] if not('batch' in str(type(wLayer)).lower()  and 'normalization' in str(type(wLayer)).lower())]
    wTrainer.setGradualUnfreeze(iStart=wEpList[1], iRate=wUnfreezeRate, iLayerNames=wBaseModelLayers)    

    wTrainer.setSaveDir(iSaveDir=wSaveDir)
    wTrainer.setData(iTrainData=wDataObjectList, iValidData=wValidDataObjectList)#, iBatchSize=wBatchSize)
    wTrainer.setBatchSizeSchedFromLists(wBatchEpochList, wBatchSizeList)
    wTrainer.logDataNames(wWithPath)
    wTrainer.setNorm(iNorm = wNorm)

    wTrainer.setCkptFreq(wCheckPointFreq)
    if wModelFlag == 'resnet':
        wImageProcess = tf.keras.applications.resnet50.preprocess_input
    elif wModelFlag == 'vgg':
        wImageProcess = tf.keras.applications.vgg16.preprocess_input
    wTrainer.setImageProcess(iImageProcess = wImageProcess, iModelFlag = wModelFlag)
    wTrainer.setAugments(iAugments=chooseAugments(wAugments))
    wTrainer.setBatchGen(iBatchGen = generate_batch)
    wTrainer.setBreakEpochsTrain(np.inf)
    wTrainer.setBreakEpochsVal(np.inf)
    wTrainer.setFromShell(gFromShell)
    wTrainer.setLogFreq(wLogFreq)
    wTrainer.printSetupInfo()
    
#%%  
    # wTrainer.train(iEpochs= wEpochs, iSaveLastN = wLastN)
    try:    
        wTrainer.train(iEpochs= wEpochs, iSaveLastN=wLastN)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Attempting to log final information")
        wTrainer.logFile()
        wTrainer.clearLossLogBuffer()
        print("Done!")
    
#%% Load Start

#     wTrainer = ModelTrainer(wModel, wOptimizer)
#     wTrainer.getModel().trainable = False
#     wTrainer.setPlotFreq(iPlotFreq=25, iPlotAll=True)
#     wDecoderName = wModel.layers[-1].name
#     wTrainer.setDecoderName(wDecoderName)
#     #%%
#     wLoadDir = os.path.join(ROOT_DIR, 'project2024','resnet_test_13_real_8bit_res2_ep_(0, 1000)_lr_1e-04_test_01')
#     wTrainer.setLoadDir(wLoadDir)
#     wCkpt = '0149_ckpt'
#     wTrainer.loadFromCkpt(wCkpt)
#     wStart = int(wCkpt.split('_')[0])+1
#     print("Automatically starting from Epoch: %s"%wStart)

#     wTrainer.setLossLvlScheduleFromDict({'0': [1, 1, 1], str(wStart): [1, 1, 1]}) #to reset watchdog
#     wTrainer.setLRSchedFromDict({'0': 0.0001, str(wStart): 1e-09, '500': 5e-07, '549': 1e-07})
#     # # wTrainer.setLossLvlScheduleFromDict({'0': [1, 0, 0], '88': [0, 1, 0], '177': [0, 0, 1], '266': [1, 1, 1]})   
#     wTrainer.setLayerFreezeScheduleFromDict({'0': {'top_model': {'top_14': True,
#         'top_15': True,
#         'top_16': True,
#         'top_out_1': True,
#         'top_red_dim_1': True,
#         'top_20': True,
#         'top_22': True,
#         'top_23': True,
#         'top_28': True,
#         'top_29': True,
#         'top_out_2': True,
#         'top_red_dim_2': True,
#         'top_30': True,
#         'top_31': True,
#         'top_32': True,
#         'top_33': True,
#         'top_34': True,
#         'top_out_3': True}}})
    
#     wBaseModelLayers = [wLayer.name for wLayer in  wTrainer.getModel().layers[:-1]]
#     wBaseModelLayers.reverse()
#     wUnfreezeNo=3
#     wUnfreezePeriod=5
#     wTrainer.setGradualUnfreeze(iStart=wStart, iRate=wUnfreezeNo/wUnfreezePeriod, iLayerNames=wBaseModelLayers)
    
#     wTrainer.setSaveDir(iSaveDir = wSaveDir)
#     wTrainer.setData(iTrainData = wDataObjectList, iValidData = wValidDataObjectList, iBatchSize = wBatchSize)
#     wTrainer.logDataNames()
#     wTrainer.setNorm(iNorm = wNorm)
    
    
    
#     wTrainer.setCkptFreq(50)
#     if wModelFlag == 'resnet':
#         wImageProcess = tf.keras.applications.resnet50.preprocess_input
#     elif wModelFlag == 'vgg':
#         wImageProcess = tf.keras.applications.vgg16.preprocess_input
#     wTrainer.setImageProcess(iImageProcess = wImageProcess, iModelFlag = wModelFlag)
#     wTrainer.setAugments(iAugments = augments)
#     wTrainer.setBatchGen(iBatchGen = generate_batch)
#     wTrainer.printSetupInfo()
    
#     wTrainer.train(iEpochs = (0,1000), iSaveLastN = 3)
    
# """ 
# #%%          
#     # for wLayer in wTrainer.getModel().layers:
#     #     # print(wLayer.name)
#     #     wWeights = wLayer.trainable_weights
#     #     for wWeight in wWeights:
#     #         print(wWeight.name)
            
# #%%
#         # wKeyList = list(wTrainer.getLayerStateDict().keys())
#         # wCurrentEpoch = wTrainer.getEpoch()
#         # for wKey in wKeyList:
#         #     if int(wKey) < wCurrentEpoch:
#         #         wDict = wTrainer.getLayerStateDict()[wKey]
#         #         wModel = wTrainer.getModel()
#         #         wDecoderName = wTrainer.getDecoderName()
#         #         if wDecoderName in wDict.keys():
#         #             wDecoder = wModel.get_layer(wDecoderName)
#         #             wDecoderDict = wDict[wDecoderName]
#         #             for wDecoderLayer in wDecoderDict.keys():
#         #                 print("setting %s to %s"%(wDecoder.get_layer(wDecoderLayer).name, wDecoderDict[wDecoderLayer]))
#         #                 wDecoder.get_layer(wDecoderLayer).trainable = wDecoderDict[wDecoderLayer]
#         #         for wLayer in wDict.keys():
#         #             if wLayer != wDecoderName:
#         #                 wModel.get_layer(wLayer).trainable = wDict[wLayer]
#     #%%
# '''    
#     wLoadFile = '0252__min_val'
#     wTrainer.loadFromCkpt(wLoadFile)
#     wStartEpoch= int(wLoadFile.split('_')[0])
#     wMinValLossList, wMinValLossNameList = wTrainer.getMinValLossList()
#     wMinValLossCounter, wMinTrainLossCounter = wTrainer.getMinValLossCounter(), wTrainer.getMinTrainLossCounter()
#     wTrainLossTracker, wValLossTracker = wTrainer.getLossTracker(), wTrainer.getLossTracker(False)
#     wMinValLoss, wMinTrainLoss = wTrainer.getMinValLoss(), wTrainer.getMinTrainLoss()
#     wValDict = {'min': wMinValLoss,
#                 'list': wMinValLossList,
#                 'names': wMinValLossNameList,
#                 'counter': wMinValLossCounter,
#                 'tracker': wValLossTracker}
#     wTrainDict = {'min': wMinTrainLoss,
#                 'counter': wMinTrainLossCounter,
#                 'tracker': wTrainLossTracker} 
                
#     wDataDict = {'val': wValDict, 'train': wTrainDict}
 
#     dump(wDataDict, os.path.join(wTrainer.getSaveDir(), wTrainer.savePrint('list_data.dump', 0)))
#     wLoadDataDict = load(os.path.join(wTrainer.getLoadDir(), wTrainer.savePrint('list_data.dump', 0)))
    
#     wLRSched = wTrainer.getLRSched()
#     wLossLvlSched = wTrainer.getLossLvlSchedule()
#     wLayerStateDict = wTrainer.getLayerStateDict()
#     wInitDict = {'loss types': wTrainer.getLossDict(), 'break': {'train': wTrainer.getBreakEpochsTrain(), 'val': wTrainer.getBreakEpochsVal()}}
#     wSchedDict = {'lr': wLRSched,
#                   'loss level': wLossLvlSched,
#                   'layer states': wLayerStateDict}
#     wInitDict.update({'sched': wSchedDict})
#     dump(wSchedDict, os.path.join(wTrainer.getSaveDir(), 'sched.dump'))
#     wLoadSchedDict = load(os.path.join(wTrainer.getLoadDir(), 'sched.dump'))


#     wMinValLossList, wMinValLossNameList = wTrainer.getMinValLossList()
#     wMinValLossCounter, wMinTrainLossCounter = wTrainer.getMinValLossCounter(), wTrainer.getMinTrainLossCounter()
#     wTrainLossTracker, wValLossTracker = wTrainer.getLossTracker(), wTrainer.getLossTracker(False)
#     wMinValLoss, wMinTrainLoss = wTrainer.getMinValLoss(), wTrainer.getMinTrainLoss()
      
#     for wFile in os.listdir(wTrainer.getLoadDir()):
#         wExt = '.data-00000-of-00001'
#         if wExt in wFile:
#             print(wFile.split(wExt)[0])    
# '''                
# """
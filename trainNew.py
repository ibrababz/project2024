# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:09:29 2024

@author: i_bab
"""
import tensorflow as tf
import os
import numpy as np
import file
from dataLoad import loadDataFilesAsObjects, generate_batch, makeNewDir
from models import makeYoloType
from tensorflow.keras.utils import plot_model
from models import save_model_summary
from TrainerClass import ModelTrainer
from augment_utils import augments

#%%

if __name__ =='__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    
    ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))
    
#%%
    wNorm = 255.
    # iSrcDir, iRes = "data4k\\train_real", 1
    iSrcDir, iRes = "data4k\\train_real_448_res2", 2
    iSrcPath = os.path.join(ROOT_DIR, iSrcDir)         
    wDataObjectList = loadDataFilesAsObjects(iSrcPath)
#%%
    # for wDataObj in wDataObjectList[:15]:
    #     show_wait(wDataObj.getImage(), 2)
        
#%%
    # iValidSrcDir = "data4k\\valid_real"
    iValidSrcDir = "data4k\\valid_real_448_res2"
    iValidSrcPath = os.path.join(ROOT_DIR, iValidSrcDir)         
    wValidDataObjectList = loadDataFilesAsObjects(iValidSrcPath)        
        
#%%
    wShape = wDataObjectList[0].getShape()
    wModelFlag = 'resnet'
    wDecLR  = 0.0001
    wFineTuneLR, wFineTuneLR2 = 5*wDecLR/10, wDecLR/10
    wModel = makeYoloType(wShape, wModelFlag, iRes)
    wOptimizer = tf.keras.optimizers.Adam(learning_rate= wDecLR ) 
    
    wModelName = 'test_13_real_8bit_res2'
    wBatchSize = 4
    wEpochs = (0,1000)
    
#%%
    wSaveFolder = wModelFlag +'_'+ wModelName + '_ep_' +str(wEpochs)+"_lr_" + "{:.0e}".format(wDecLR)
    wSaveDir = makeNewDir(os.path.join(os.getcwd(), wSaveFolder), 0)
    os.mkdir(wSaveDir)

#%%
    plot_model(wModel.layers[-1], os.path.join(wSaveDir, 'top_model.png'), show_shapes = True)
    plot_model(wModel, os.path.join(wSaveDir, 'wModel.png'), show_shapes = True)
    save_model_summary(wSaveDir, wModel.layers[-1])
    save_model_summary(wSaveDir, wModel)
    
    
#%% Fresh Start
    # wTrainer = ModelTrainer(wModel, wOptimizer)
    # wTrainer.getModel().trainable = False
    # wTrainer.setPlotFreq(iPlotFreq=25, iPlotAll=True)
    # wDecoderName = wModel.layers[-1].name
    # wTrainer.setDecoderName(wDecoderName)
    # wDen = 4
    
    # wEpochRes0, wEpochRes1, wEpochRes2, wEpochFineTune, wEpochFineTune2 = wEpochs[0], wEpochs[1]//wDen, 2*wEpochs[1]//wDen, 2*wEpochs[1]//wDen, 4*wEpochs[1]//wDen
    
    # wEpList, wLRList = [0, 299, 500, 549], [0.0001, 5e-5, 1e-5, 5e-6]
    # for wEp, wLR in zip(wEpList, wLRList):
    #     wTrainer.setLRSched(wEp, wLR)

    
    # wRes0 = ['top_14', 'top_15', 'top_16', 'top_out_1']
    # wRes1 = ['top_red_dim_1', 'top_20', 'top_22', 'top_23', 'top_28', 'top_29', 'top_out_2']
    # wRes2 = ['top_red_dim_2', 'top_30', 'top_31', 'top_32', 'top_33', 'top_34', 'top_out_3']
    
    # # wTrainer.setLossDict(iLossDict = {'Pos': 5., 'Neg': 5.*5., 'Dice': 1.})
    # wTrainer.setLossDict(iLossDict = {'Pos': 5., 'Neg': 5.*5., 'Dice': 1., 'Cart':5.*5.})

    # wTrainer.setLossLvlSchedule(iEpoch=0, iLossLvl = [1,1,1])
    # # wTrainer.setLossLvlSchedule(iEpoch = wEpochRes0, iLossLvl = [1,0,0])
    # # wTrainer.setLossLvlSchedule(iEpoch = wEpochRes1, iLossLvl = [0,1,0])
    # # wTrainer.setLossLvlSchedule(iEpoch = wEpochRes2, iLossLvl = [0,0,1])
    # wTrainer.setLossLvlSchedule(iEpoch=499, iLossLvl = [1,1,1]) #to reset loss watch dog

    # wFreezeDict = {}
    # wDecoderDict = {}
    # wDecoderLayers = []
    # wDecoderLayers.extend(wRes0)
    # wDecoderLayers.extend(wRes1)
    # wDecoderLayers.extend(wRes2)
    # wDecoderDict.update(wTrainer.genLayerStateDict(wDecoderLayers, [True]*len(wDecoderLayers)))
    # wFreezeDict.update({wDecoderName: wDecoderDict})
    # wTrainer.setLayerFreezeScheduleByName(wEpochRes0, wFreezeDict)

    # wBaseModelLayers = [wLayer.name for wLayer in wModel.layers[:-1]]
    # wBaseModelLayers.reverse()
    # wUnfreezeNo = 3
    # wUnfreezePeriod = 5
    # wTrainer.setGradualUnfreeze(iStart=wEpochFineTune, iRate=wUnfreezeNo/wUnfreezePeriod, iLayerNames=wBaseModelLayers)    

    # wTrainer.setSaveDir(iSaveDir=wSaveDir)
    # wTrainer.setData(iTrainData=wDataObjectList, iValidData=wValidDataObjectList, iBatchSize=wBatchSize)
    # wTrainer.logDataNames()
    # wTrainer.setNorm(iNorm = wNorm)

    # wTrainer.setCkptFreq(50)
    # if wModelFlag == 'resnet':
    #     wImageProcess = tf.keras.applications.resnet50.preprocess_input
    # elif wModelFlag == 'vgg':
    #     wImageProcess = tf.keras.applications.vgg16.preprocess_input
    # wTrainer.setImageProcess(iImageProcess = wImageProcess, iModelFlag = wModelFlag)
    # wTrainer.setAugments(iAugments = augments)
    # wTrainer.setBatchGen(iBatchGen = generate_batch)
    # wTrainer.setBreakEpochsTrain(np.inf)
    # wTrainer.setBreakEpochsVal(np.inf)
    # wTrainer.printSetupInfo()
    
#%%        
    # wTrainer.train(iEpochs= wEpochs, iSaveLastN = 3)
#%%
    
# #%% Fresh Start
#     wTrainer = ModelTrainer(wModel, wOptimizer)
#     wTrainer.getModel().trainable = False
#     wTrainer.setPlotFreq(iPlotFreq=25, iPlotAll=True)
#     wDecoderName = wModel.layers[-1].name
#     wTrainer.setDecoderName(wDecoderName)
#     wDen = 6
    
#     wEpochRes0, wEpochRes1, wEpochRes2, wEpochFineTune, wEpochFineTune2 = wEpochs[0], wEpochs[1]//wDen, 2*wEpochs[1]//wDen, 3*wEpochs[1]//wDen, 4*wEpochs[1]//wDen
    
#     wTrainer.setLRSched(wEpochRes0, wDecLR)
#     # wTrainer.setLRSched(170, 0.0005)
#     wTrainer.setLRSched(wEpochFineTune, wFineTuneLR)
#     wTrainer.setLRSched(wEpochFineTune2, wFineTuneLR2)
    
#     wRes0 = ['top_14', 'top_15', 'top_16', 'top_out_1']
#     wRes1 = ['top_red_dim_1', 'top_20', 'top_22', 'top_23', 'top_28', 'top_29', 'top_out_2']
#     wRes2 = ['top_red_dim_2', 'top_30', 'top_31', 'top_32', 'top_33', 'top_34', 'top_out_3']
    
#     wTrainer.setLossDict(iLossDict = {'Pos': 5., 'Neg': 5.*5., 'Dice': 1.})
    
#     # wTrainer.setLossLvlSchedule(iEpoch = wEpochRes0, iLossLvl = [1,1,1])
#     wTrainer.setLossLvlSchedule(iEpoch = wEpochRes0, iLossLvl = [1,0,0])
#     wTrainer.setLossLvlSchedule(iEpoch = wEpochRes1, iLossLvl = [0,1,0])
#     wTrainer.setLossLvlSchedule(iEpoch = wEpochRes2, iLossLvl = [0,0,1])
#     wTrainer.setLossLvlSchedule(iEpoch = wEpochFineTune, iLossLvl = [1,1,1])
    
#     wFreezeDict = {}
#     wDecoderDict = {}
#     wDecoderDict.update(wTrainer.genLayerStateDict(wRes0, [True]*len(wRes0)))
#     wFreezeDict.update({wDecoderName: wDecoderDict})
#     wTrainer.setLayerFreezeScheduleByName(wEpochRes0, wFreezeDict)
#     wUnfreezePeriod = 1
#     wUnfreezeNo = len(wRes1)
#     wTrainer.setGradualUnfreeze(iStart=wEpochRes1, iRate=wUnfreezeNo/wUnfreezePeriod, iLayerNames=wRes1, iDecoderFlag=True)
#     wUnfreezePeriod = 1
#     wUnfreezeNo = len(wRes2)
#     wTrainer.setGradualUnfreeze(iStart=wEpochRes2, iRate=wUnfreezeNo/wUnfreezePeriod, iLayerNames=wRes2, iDecoderFlag=True)
    
    
#     # wBaseModelLayers = [wLayer.name for wLayer in wModel.layers[:-1]]
#     # wBaseModelLayers.reverse()
    
#     # wFreezeDict = {}
#     # wFreezeDict.update(wTrainer.genLayerStateDict(wBaseModelLayers, [True]*len(wBaseModelLayers)))
#     # wTrainer.setLayerFreezeScheduleByName(wEpochFineTune, wFreezeDict)
#     wBaseModelLayers = [wLayer.name for wLayer in wModel.layers[:-1]]
#     wBaseModelLayers.reverse()
#     wPart1, wPart2, wPart3 = [], [], []
#     wFlag = 0
#     for wLayer in wBaseModelLayers:
#         if wLayer in ['conv4_block4_3_bn', 'conv4_block4_3_conv', 'conv4_block4_2_relu']:
#             wPart2.append(wLayer)
#             wFlag = 1
#         else:
#             if wFlag:
#                 wPart3.append(wLayer)
#             else:
#                 wPart1.append(wLayer)
#     wUnfreezePeriod = 4
#     wUnfreezeNo = 3
#     wTrainer.setGradualUnfreeze(iStart = wEpochFineTune, iRate= wUnfreezeNo/wUnfreezePeriod, iLayerNames = wPart1)  
#     wLastEpoch =int(list(wTrainer.getLayerStateDict().keys())[-1])
#     wUnfreezeNo = 1
#     wTrainer.setGradualUnfreeze(iStart = wLastEpoch + wUnfreezePeriod, iRate= wUnfreezeNo/wUnfreezePeriod, iLayerNames = wPart2)  
#     wLastEpoch =int(list(wTrainer.getLayerStateDict().keys())[-1])
#     wUnfreezeNo = 3
#     wTrainer.setGradualUnfreeze(iStart = wLastEpoch + wUnfreezePeriod, iRate= wUnfreezeNo/wUnfreezePeriod, iLayerNames = wPart3) 
#     wTrainer.setSaveDir(iSaveDir = wSaveDir)
#     wTrainer.setData(iTrainData = wDataObjectList, iValidData = wValidDataObjectList, iBatchSize = wBatchSize)
#     wTrainer.setNorm(iNorm = wNorm)
#     wTrainer.logDataNames()
#     wTrainer.setCkptFreq(50)
#     if wModelFlag == 'resnet':
#         wImageProcess = tf.keras.applications.resnet50.preprocess_input
#     elif wModelFlag == 'vgg':
#         wImageProcess = tf.keras.applications.vgg16.preprocess_input
#     wTrainer.setImageProcess(iImageProcess = wImageProcess, iModelFlag = wModelFlag)
#     wTrainer.setAugments(iAugments = augments)
#     wTrainer.setBatchGen(iBatchGen = generate_batch)
#     wTrainer.setBreakEpochsTrain(np.inf)
#     wTrainer.setBreakEpochsVal(np.inf)
# #%%        
#     wTrainer.train(iEpochs= wEpochs, iSaveLastN = 3)
    
#%% Load Start

    wTrainer = ModelTrainer(wModel, wOptimizer)
    wTrainer.getModel().trainable = False
    wTrainer.setPlotFreq(iPlotFreq=25, iPlotAll=True)
    wDecoderName = wModel.layers[-1].name
    wTrainer.setDecoderName(wDecoderName)
    #%%
    wLoadDir = os.path.join(ROOT_DIR, 'project2024','resnet_test_13_real_8bit_res2_ep_(0, 1000)_lr_1e-04_test_01')
    wTrainer.setLoadDir(wLoadDir)
    wCkpt = '0149_ckpt'
    wTrainer.loadFromCkpt(wCkpt)
    wStart = int(wCkpt.split('_')[0])+1
    print("Automatically starting from Epoch: %s"%wStart)

    wTrainer.setLossLvlScheduleFromDict({'0': [1, 1, 1], str(wStart): [1, 1, 1]}) #to reset watchdog
    wTrainer.setLRSchedFromDict({'0': 0.0001, str(wStart): 1e-09, '500': 5e-07, '549': 1e-07})
    # # wTrainer.setLossLvlScheduleFromDict({'0': [1, 0, 0], '88': [0, 1, 0], '177': [0, 0, 1], '266': [1, 1, 1]})   
    wTrainer.setLayerFreezeScheduleFromDict({'0': {'top_model': {'top_14': True,
       'top_15': True,
       'top_16': True,
       'top_out_1': True,
       'top_red_dim_1': True,
       'top_20': True,
       'top_22': True,
       'top_23': True,
       'top_28': True,
       'top_29': True,
       'top_out_2': True,
       'top_red_dim_2': True,
       'top_30': True,
       'top_31': True,
       'top_32': True,
       'top_33': True,
       'top_34': True,
       'top_out_3': True}}})
    
    wBaseModelLayers = [wLayer.name for wLayer in  wTrainer.getModel().layers[:-1]]
    wBaseModelLayers.reverse()
    wUnfreezeNo=3
    wUnfreezePeriod=5
    wTrainer.setGradualUnfreeze(iStart=wStart, iRate=wUnfreezeNo/wUnfreezePeriod, iLayerNames=wBaseModelLayers)
    
    wTrainer.setSaveDir(iSaveDir = wSaveDir)
    wTrainer.setData(iTrainData = wDataObjectList, iValidData = wValidDataObjectList, iBatchSize = wBatchSize)
    wTrainer.logDataNames()
    wTrainer.setNorm(iNorm = wNorm)
    
    
    
    wTrainer.setCkptFreq(50)
    if wModelFlag == 'resnet':
        wImageProcess = tf.keras.applications.resnet50.preprocess_input
    elif wModelFlag == 'vgg':
        wImageProcess = tf.keras.applications.vgg16.preprocess_input
    wTrainer.setImageProcess(iImageProcess = wImageProcess, iModelFlag = wModelFlag)
    wTrainer.setAugments(iAugments = augments)
    wTrainer.setBatchGen(iBatchGen = generate_batch)
    wTrainer.printSetupInfo()
    
    wTrainer.train(iEpochs = (0,1000), iSaveLastN = 3)
    
""" 
#%%          
    # for wLayer in wTrainer.getModel().layers:
    #     # print(wLayer.name)
    #     wWeights = wLayer.trainable_weights
    #     for wWeight in wWeights:
    #         print(wWeight.name)
            
#%%
        # wKeyList = list(wTrainer.getLayerStateDict().keys())
        # wCurrentEpoch = wTrainer.getEpoch()
        # for wKey in wKeyList:
        #     if int(wKey) < wCurrentEpoch:
        #         wDict = wTrainer.getLayerStateDict()[wKey]
        #         wModel = wTrainer.getModel()
        #         wDecoderName = wTrainer.getDecoderName()
        #         if wDecoderName in wDict.keys():
        #             wDecoder = wModel.get_layer(wDecoderName)
        #             wDecoderDict = wDict[wDecoderName]
        #             for wDecoderLayer in wDecoderDict.keys():
        #                 print("setting %s to %s"%(wDecoder.get_layer(wDecoderLayer).name, wDecoderDict[wDecoderLayer]))
        #                 wDecoder.get_layer(wDecoderLayer).trainable = wDecoderDict[wDecoderLayer]
        #         for wLayer in wDict.keys():
        #             if wLayer != wDecoderName:
        #                 wModel.get_layer(wLayer).trainable = wDict[wLayer]
    #%%
'''    
    wLoadFile = '0252__min_val'
    wTrainer.loadFromCkpt(wLoadFile)
    wStartEpoch= int(wLoadFile.split('_')[0])
    wMinValLossList, wMinValLossNameList = wTrainer.getMinValLossList()
    wMinValLossCounter, wMinTrainLossCounter = wTrainer.getMinValLossCounter(), wTrainer.getMinTrainLossCounter()
    wTrainLossTracker, wValLossTracker = wTrainer.getLossTracker(), wTrainer.getLossTracker(False)
    wMinValLoss, wMinTrainLoss = wTrainer.getMinValLoss(), wTrainer.getMinTrainLoss()
    wValDict = {'min': wMinValLoss,
                'list': wMinValLossList,
                'names': wMinValLossNameList,
                'counter': wMinValLossCounter,
                'tracker': wValLossTracker}
    wTrainDict = {'min': wMinTrainLoss,
                'counter': wMinTrainLossCounter,
                'tracker': wTrainLossTracker} 
                
    wDataDict = {'val': wValDict, 'train': wTrainDict}
 
    dump(wDataDict, os.path.join(wTrainer.getSaveDir(), wTrainer.savePrint('list_data.dump', 0)))
    wLoadDataDict = load(os.path.join(wTrainer.getLoadDir(), wTrainer.savePrint('list_data.dump', 0)))
    
    wLRSched = wTrainer.getLRSched()
    wLossLvlSched = wTrainer.getLossLvlSchedule()
    wLayerStateDict = wTrainer.getLayerStateDict()
    wInitDict = {'loss types': wTrainer.getLossDict(), 'break': {'train': wTrainer.getBreakEpochsTrain(), 'val': wTrainer.getBreakEpochsVal()}}
    wSchedDict = {'lr': wLRSched,
                  'loss level': wLossLvlSched,
                  'layer states': wLayerStateDict}
    wInitDict.update({'sched': wSchedDict})
    dump(wSchedDict, os.path.join(wTrainer.getSaveDir(), 'sched.dump'))
    wLoadSchedDict = load(os.path.join(wTrainer.getLoadDir(), 'sched.dump'))


    wMinValLossList, wMinValLossNameList = wTrainer.getMinValLossList()
    wMinValLossCounter, wMinTrainLossCounter = wTrainer.getMinValLossCounter(), wTrainer.getMinTrainLossCounter()
    wTrainLossTracker, wValLossTracker = wTrainer.getLossTracker(), wTrainer.getLossTracker(False)
    wMinValLoss, wMinTrainLoss = wTrainer.getMinValLoss(), wTrainer.getMinTrainLoss()
      
    for wFile in os.listdir(wTrainer.getLoadDir()):
        wExt = '.data-00000-of-00001'
        if wExt in wFile:
            print(wFile.split(wExt)[0])    
'''                
"""
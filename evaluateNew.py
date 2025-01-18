# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:09:29 2024

@author: i_bab
"""
import tensorflow as tf
import os
import file
from dataLoad import loadDataFilesAsObjects, generate_batch
from models import makeYoloType
from TrainerClass import ModelEvaluator


#%%

if __name__ =='__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    
    ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))
    
#%%
    wNorm = 255.
    # iTestSrcDir = "data4k\\test_real"
    iTestSrcDir, iRes, iDepth = "data4k\\test_real_448_res2", 3, 1
    iTestSrcPath = os.path.join(ROOT_DIR, iTestSrcDir)         
    wTestDataObjectList = loadDataFilesAsObjects(iTestSrcPath)        
        
#%%
    wShape = wTestDataObjectList[0].getShape()
    wModelFlag = 'resnet'
    wDecLR  = 0.0001
    wModel = makeYoloType(wShape, wModelFlag, iRes, iDeeper=iDepth)
    wOptimizer = tf.keras.optimizers.Adam(learning_rate= wDecLR ) 
    wBatchSize = 4
    
#%% Load Start
    wEvaluator = ModelEvaluator(wModel, wOptimizer)
    # wDecoderName = wModel.layers[-1].name
    # wEvaluator.setDecoderName(wDecoderName)
    #%%
    # wLoadDir = os.path.join(ROOT_DIR, 'project2024','resnet_test_13_real_8bit_res2_ep_(0, 1000)_lr_1e-04')
    wLoadDir = os.path.join(ROOT_DIR, 'project2024','test_13_resnet_ep_0-1200_lr_1e-04')    
    wEvaluator.setLoadDir(wLoadDir)
    wEvaluator.loadFromCkpt('1199_ckpt')
    wEvaluator.setData(iTestData = wTestDataObjectList, iBatchSize = wBatchSize)
    wEvaluator.setNorm(iNorm = wNorm)
    wEvaluator.setSaveDir(iSaveDir = wLoadDir)
    wEvaluator.logDataNames()

    if wModelFlag == 'resnet':
        wImageProcess = tf.keras.applications.resnet50.preprocess_input
    elif wModelFlag == 'vgg':
        wImageProcess = tf.keras.applications.vgg16.preprocess_input
    wEvaluator.setImageProcess(iImageProcess = wImageProcess, iModelFlag = wModelFlag)
    wEvaluator.setAugments(iAugments = None)
    wEvaluator.setBatchGen(iBatchGen = generate_batch)
    wEvaluator.setShowPlots(False)
    wEvaluator.setSaveSize((1280,1280))
    wEvaluator.setSavePlotType('truth_acts')
    wEvaluator.setSavePlots(False)
#%%    
    for iResIdx in range(3):
        print("Res:%s"%(iResIdx+1))
        wStart, wEnd, wStep = 0.5, 1, 0.025
        wNoSteps = int((wEnd-wStart)//wStep)+1
        wThreshList = [round(wStart+i*wStep,3) for i in range(wNoSteps)]
        # wThreshList = [.85]
        for wThresh in wThreshList:
            wEvaluator.computeMetrics(wThresh, iResIdx)
            wEvaluator.printMetrics(iResIdx, wThresh)
        # print(wEvaluator.getPrecMetrics(iResIdx))    
#%% Save original color images to png for inputs to segmenation code 
    # import cv2 as cv

    # wTestNames = wEvaluator.genLogDataNames(False)
    # wTestNamesSorted = wTestNames.copy()
    # wTestNamesSorted.sort()
    
    # wTestNamesToCompare = []
    # for wName in wTestNamesSorted:
    #     if int(wName.split('_')[1].split('.bmp')[0]) > 130:
    #         wTestNamesToCompare.append(wName)
            
    # wSrcFolder = '1'
    # wDstFolder = '1_test_png_for_segmentation_compare'
   
    # wDataDir = 'data4k'
    
    # wSrcDir = os.path.join(ROOT_DIR, wDataDir, wSrcFolder)
    # wDstDir = os.path.join(ROOT_DIR,  wDataDir, wDstFolder)
    # os.makedirs(wDstDir, exist_ok=True)
    
    # for wFile in os.listdir(wSrcDir):
    #     if wFile in wTestNamesToCompare:
    #         wReadPath = os.path.join(wSrcDir, wFile)
    #         wIm = cv.imread(wReadPath)
    #         wWriteName = wFile.split('.bmp')[0]+'.png'
    #         # print(wWriteName)
    #         wWritePath = os.path.join(wDstDir, wWriteName)
    #         cv.imwrite(wWritePath, wIm)
    
#%% resize to square for easier plotting in report
    # import cv2 as cv
            
    # wSrcFolder = '1_test_png_for_segmenation_compare_outputs_sifted'
    # wDstFolder = '1_test_png_for_segmenation_compare_outputs_sifted_square'
   
    # wDataDir = 'data4k'
    
    # wSrcDir = os.path.join(ROOT_DIR, wDataDir, wSrcFolder)
    # wDstDir = os.path.join(ROOT_DIR,  wDataDir, wDstFolder)
    # os.makedirs(wDstDir, exist_ok=True)
    
    # for wFile in os.listdir(wSrcDir):
    #     wReadPath = os.path.join(wSrcDir, wFile)
    #     wIm = cv.imread(wReadPath)
    #     H, W, C = wIm.shape
    #     wWriteName = wFile
    #     # print(wWriteName)
    #     wWritePath = os.path.join(wDstDir, wWriteName)
    #     cv.imwrite(wWritePath, cv.resize(wIm, (H,H), interpolation=cv.INTER_NEAREST))
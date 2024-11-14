# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:09:29 2024

@author: i_bab
"""
import tensorflow as tf
import os
import file
from modelTemplate import makeYoloTypeWithResNet50
from LoadAndPredictClass import ModelLoad
from helper import show_wait
import cv2 as cv


#%%

if __name__ =='__main__':
    
    ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))
    
#%%
    wNorm = 255. #if data is type uint8 (8 bit/clr_channel)
    wInputShape = (448,448,3) #shape of expected input image (448,448,3) for this model       
        
#%%
    wModel = makeYoloTypeWithResNet50(wInputShape)
    
#%% Load Start
    wEvaluator = ModelLoad(wModel)
    #%%
    wLoadDir = os.path.join(ROOT_DIR, 'project2024','resnet_test_13_real_8bit_res2_ep_(0, 1000)_lr_1e-04')
    wEvaluator.setLoadDir(wLoadDir)
    wEvaluator.loadFromCkpt('0987_min_val')
    
    #%%
    wImageProcess = tf.keras.applications.resnet50.preprocess_input
    wEvaluator.setImageProcess(iImageProcess=wImageProcess)
    
    #%%
    wEvaluator.setNorm(iNorm=wNorm)

    #%%
    wDataFolder = os.path.join('data4k', 'test_images_448_bitmap')
    wDataDir= os.path.join(ROOT_DIR, wDataFolder)
    wEvaluator.setDataDir(wDataDir)
    wEvaluator.loadData()
    
    #%%
    wScale = 10.
    wFlag = cv.INTER_NEAREST
    wEvaluator.predict()
    wEvaluator.thresholdPreds(0.85) #define threshold
    wEvaluator.extractPredActivations()
    wEvaluator.computePredCentroids()
    for wIdx in range(len(wEvaluator.getNameList()[5:9])):
        for wRes in wEvaluator.getResolutions():
            show_wait(wEvaluator.getPredAtIndexAtRes(wIdx, wRes), scale=wScale, interpolation= wFlag)
            show_wait(wEvaluator.getPredAtIndexAtRes(wIdx, wRes, 'thresh'), scale=wScale, interpolation= wFlag)
            show_wait(wEvaluator.getPredAtIndexAtRes(wIdx, wRes, 'act'), scale=wScale, interpolation= wFlag)
            show_wait(wEvaluator.getPredAtIndexAtRes(wIdx, wRes, 'cent'), scale=wScale, interpolation= wFlag)

#%%
    wSaveFolder='outputs'
    wEvaluator.setSaveDir(iSaveDir=os.path.join(wLoadDir, wSaveFolder))    
    wEvaluator.savePlots(wInputShape[:2])
    
#%%
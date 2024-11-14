# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:25:43 2024

@author: i_bab
"""
#%%
import os
import file
import cv2 as cv
from helper import show_wait
#from imageUtils import CombinedWeed
from dataLoad import GenerateWeedDataList, WeedDataLoader, RandomWeedBatchGenerator, loadDataFilesAsObjects
from sklearn.model_selection import train_test_split
from dataLoad import genDataFiles
#from copy import deepcopy
#from RandomSampler import RandSampler

#%%
    
ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))

src_folder = 'data2019\\1\\train1_contrast_masks_clean'
src_dir = os.path.join(ROOT_DIR, src_folder)

src_folder2 = 'data2019\\1\\train1'
src_dir2 = os.path.join(ROOT_DIR, src_folder2)

pts_file = 'mask_centers.json'
pts_path = os.path.join(src_folder, pts_file)

src_folder3 = "data2019\\0\\train1_0_grass"
src_path3 = os.path.join(ROOT_DIR, src_folder3)

Wo, Ho = 640, 480
W, H = 640, 480 #224, 224
dim = (W,H)

dstDim = (224, 224) #(448,448)

W_grid1, H_grid1 = 7, 7 #14, 14
dim_grid1 = (W_grid1, H_grid1)
W_grid2, H_grid2 = 14, 14 #28, 28
dim_grid2 = (W_grid2, H_grid2)
# W_grid3, H_grid3 = 28, 28
# dim_grid3 = (W_grid3, H_grid3)
dim_grid_list = [dim_grid1, dim_grid2]#, dim_grid3]

scale_x, scale_y = W/Wo, H/Ho

weed_list = GenerateWeedDataList(src_dir, src_dir2, pts_file, scale_x, scale_y)
grass_list = [cv.resize(cv.imread(os.path.join(src_path3,x)), dim, interpolation = cv.INTER_AREA) for x in  os.listdir(src_path3)]


ValidLenPercent = 0.30
weed_train, weed_valid, _, _ =  train_test_split(weed_list, weed_list, test_size = ValidLenPercent, random_state = 42)
grass_train, grass_valid, _, _ =  train_test_split(grass_list, grass_list, test_size = ValidLenPercent, random_state = 42)
# grass_train = grass_list[34:35]#[12:13]#[34:35]#[:1]
# weed_train = weed_list[27:28]
#%%
batchSize = 1
weedSampleSize = 4
samplerSeed = 0

#%%
validSize = 900
ValidData = WeedDataLoader(weed_valid, dim)
ValidBatchGen = RandomWeedBatchGenerator(batchSize, ValidData, dim_grid_list, grass_valid, weedSampleSize, samplerSeed)
ValidBatchGen.setBatchDim(dstDim)
ValidBatchGen.setTranLimits((1/5, 1/2), 0.5, 15)
ValidBatchGen.setNoRepeat(False)
ValidBatchGen.setSize(validSize)

#%%
trainSize = 3000
TrainData = WeedDataLoader(weed_train, dim)
TrainBatchGen = RandomWeedBatchGenerator(batchSize, TrainData, dim_grid_list, grass_train, weedSampleSize, samplerSeed)
TrainBatchGen.setBatchDim(dstDim)
TrainBatchGen.setTranLimits((1/5, 1/2), 0.5, 15)
TrainBatchGen.setNoRepeat(False)
TrainBatchGen.setSize(trainSize)

#%%
iDestDir = "data2019\\1\\train_synth_8bit_"+str(dstDim).strip('(').strip(')').replace(', ', '_')+'_sample_' +str(weedSampleSize)
iDestPath = os.path.join(ROOT_DIR, iDestDir)
genDataFiles(TrainBatchGen, iDestPath, iUintFlag = 1)
#%%    
iValidDestDir ="data2019\\1\\valid_synth_8bit_"+str(dstDim).strip('(').strip(')').replace(', ', '_')+'_sample_' +str(weedSampleSize)
iValidDestPath = os.path.join(ROOT_DIR, iValidDestDir)
genDataFiles(ValidBatchGen, iValidDestPath, iUintFlag  = 1)

#%%
import numpy as np
from loss_functions import flatten_map_v2

def TrainEpoch(myWeedBatchGen, epochSize):
    for i, batch in zip(range(int(epochSize/myWeedBatchGen.getBatchSize())), myWeedBatchGen):
        batchNames, batchImages, batchMasks, batchMaps = batch
        multidimHmaps_list = []
        print("myWeedBatchGen.getCounter() = %s" %myWeedBatchGen.getCounter())
        print("batchImages[0].dtype %s" %batchImages[0].dtype)
        show_wait(np.hstack(batchImages), 2)
        show_wait(np.hstack(batchMasks), 2)
        for map_list in batchMaps:
            hmaps_list = []
            for hmap in map_list:
                hmaps_list.append(flatten_map_v2(hmap))
            multidimHmaps_list.append(hmaps_list)
        for hmaps_list in multidimHmaps_list[1:]:
            show_wait(np.hstack(hmaps_list), 20, interpolation = cv.INTER_NEAREST)

#%%
# TrainEpoch(ValidBatchGen, 50)


#%%
# TrainEpoch(TrainBatchGen, 50)


   
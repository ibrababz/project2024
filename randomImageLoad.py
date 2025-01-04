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
from dataLoad import flat_map_list_v2
#from copy import deepcopy
#from RandomSampler import RandSampler

#%%
    


ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))

#%%
iSrcDir = "data2019\\synth\\test_11_tr_10_dim_448_sample_4_frac_1e+00_res_3_blend_True_clr_True_uint"
iSrcPath = os.path.join(ROOT_DIR, iSrcDir)         
wDataObjectList = loadDataFilesAsObjects(iSrcPath)[:]
#%%
for wDataObj in wDataObjectList[:15]:
    show_wait(wDataObj.getImage(), 2)

    
#%%
iValidSrcDir = "data2019\\1\\valid_synth_8bit"
iValidSrcPath = os.path.join(ROOT_DIR, iValidSrcDir)         
wValidDataObjectList = loadDataFilesAsObjects(iValidSrcPath)[:]

#%%
for wValidDataObj in wValidDataObjectList[int(5*8):int(5*8)+8]:
    show_wait(wValidDataObj.getImage(), 2)

#%%
for wValidDataObj in wValidDataObjectList:
    print(round(wValidDataObj.getImage().min()*255, 0), round(wValidDataObj.getImage().max()*255,0))

#%%
import numpy as np
from loss_functions import flatten_map_v2, act_list_3D

def getAvgSpread(iDataObjList, iDimIdx = 0, iOneChan = False):
    n = len(iDataObjList)
    wMap3DList = []

    for i, wDataObj in zip(range(n),iDataObjList):
        wMap3D = wDataObj.getMapAtIndex(iDimIdx)
        if iOneChan:
            wMap3DList.append(wMap3D[...,0][...,None])
        else:
            wMap3DList.append(wMap3D)
    
    wAct3DList = act_list_3D(wMap3DList)
    wActList = flat_map_list_v2(wAct3DList)
    #show_wait(wActList[0])
    oActSumMap = np.sum(wActList, axis=0, keepdims=False)
    # print(wActList[0].shape)
    # print(oActSumMap.shape)
    return oActSumMap

    
#%%
iDimIdx = 0
wValidAvgSpreadMap = getAvgSpread(wValidDataObjectList, iDimIdx)[...,0]
wValidSampleCountMap = getAvgSpread(wValidDataObjectList, iDimIdx , True)[...,0]   
print("valid samples counted <=%s:"%np.sum(wValidSampleCountMap))
wValidSpreadIm = wValidAvgSpreadMap/np.max(wValidAvgSpreadMap)
wAvgSpreadMap = getAvgSpread(wDataObjectList, iDimIdx)[...,0]
wSampleCountMap = getAvgSpread(wDataObjectList, iDimIdx, True)[...,0]  
print("training samples counted <=%s:"%np.sum(wSampleCountMap))
wSpreadIm = wAvgSpreadMap/np.max(wAvgSpreadMap)

show_wait(wValidSpreadIm, 20, interpolation = cv.INTER_NEAREST)
show_wait(wSpreadIm, 20, interpolation = cv.INTER_NEAREST)   

#%%
iDimIdx = 1
wValidAvgSpreadMap = getAvgSpread(wValidDataObjectList, iDimIdx)[...,0]
wValidSampleCountMap = getAvgSpread(wValidDataObjectList, iDimIdx , True)[...,0]   
print("valid samples counted <=%s:"%np.sum(wValidSampleCountMap))
wValidSpreadIm = wValidAvgSpreadMap/np.max(wValidAvgSpreadMap)
wAvgSpreadMap = getAvgSpread(wDataObjectList, iDimIdx)[...,0]
wSampleCountMap = getAvgSpread(wDataObjectList, iDimIdx, True)[...,0]  
print("training samples counted <=%s:"%np.sum(wSampleCountMap))
wSpreadIm = wAvgSpreadMap/np.max(wAvgSpreadMap)

show_wait(wValidSpreadIm, 20, interpolation = cv.INTER_NEAREST)
show_wait(wSpreadIm, 20, interpolation = cv.INTER_NEAREST) 
  
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


   
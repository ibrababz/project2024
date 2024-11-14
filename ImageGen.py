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
from dataLoad import generate_data_list, GenerateWeedDataList, WeedDataLoader, RandomWeedBatchGenerator, loadDataFilesAsObjects
from sklearn.model_selection import train_test_split
from dataLoad import genDataFiles, genDataFilesFromOrgImBatchGen, generate_batch, loadDataFilesAsObjects
#from copy import deepcopy, 
#from RandomSampler import RandSampler

#%%
    
ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))
data_dir = 'data4k'
folder = '1_448x448'
data_folder_path = os.path.join(ROOT_DIR, data_dir, folder)
folder_points = '1'
file_points = "plant_centers_sifted_FINAL.json"
points_path = os.path.join(ROOT_DIR, data_dir, folder_points, file_points)

#%%
Wo, Ho = 4056, 3040
W, H = 448, 448
shape = (H,W,3)
dim = (W,H)
scale_x, scale_y = W/Wo, H/Ho
oimg_list = generate_data_list(data_folder_path, points_path, scale_x, scale_y)
X_train, X_test, _, _ =  train_test_split(oimg_list, oimg_list, test_size = 0.2, random_state = 42)
X_test, X_valid, _, _ = train_test_split(X_test, X_test, test_size = 0.5, random_state = 42)
W_grid1, H_grid1 = 14, 14
dim_grid1 = (W_grid1, H_grid1)
W_grid2, H_grid2 = 28, 28
dim_grid2 = (W_grid2, H_grid2)
W_grid3, H_grid3 = 56, 56
dim_grid3 = (W_grid3, H_grid3)
dim_grid_list = [dim_grid1, dim_grid2, dim_grid3]

#%%
wImBatchGen = generate_batch(X_train, 1)
iDestDir = "data4k\\train_real_448_res2"
iDestPath = os.path.join(ROOT_DIR, iDestDir)
genDataFilesFromOrgImBatchGen(wImBatchGen, dim, dim_grid_list, iDestPath)

#%%
wValidImBatchGen = generate_batch(X_valid, 1)
iValidDestDir = "data4k\\valid_real_448_res2"
iValidDestPath = os.path.join(ROOT_DIR, iValidDestDir)
genDataFilesFromOrgImBatchGen(wValidImBatchGen, dim, dim_grid_list, iValidDestPath, 'valid')

#%%
wTestImBatchGen = generate_batch(X_test, 1)
iTestDestDir = "data4k\\test_real_448_res2"
iTestDestPath = os.path.join(ROOT_DIR, iTestDestDir)
genDataFilesFromOrgImBatchGen(wTestImBatchGen, dim, dim_grid_list, iTestDestPath, 'test')

#%%
iSrcDir = "data4k\\train_real_448_res2"
iSrcPath = os.path.join(ROOT_DIR, iSrcDir)         
wDataObjectList = loadDataFilesAsObjects(iSrcPath)

#%%
iValidSrcDir = "data4k\\valid_real_448_res2"
iValidSrcPath = os.path.join(ROOT_DIR, iValidSrcDir)
wValidDataObjectList = loadDataFilesAsObjects(iValidSrcPath)

#%%
iTestSrcDir = "data4k\\test_real_448_res2"
iTestSrcPath = os.path.join(ROOT_DIR, iTestSrcDir)
wTestDataObjectList = loadDataFilesAsObjects(iTestSrcPath)  

#%%
import numpy as np
from loss_functions import flatten_map_v2

for wDataObj in wDataObjectList[:20]:
    h, w, d  = wDataObj.getMapAtIndex(0).shape
    if d > 1:
        print(wDataObj.getNamesList()[0])
        show_wait(wDataObj.getImage(), 2)
        for i in range(wDataObj.getNoDims()):
                print(wDataObj.getMapAtIndex(i).shape)
                show_wait(flatten_map_v2(wDataObj.getMapAtIndex(i)),15, interpolation = cv.INTER_NEAREST)

#%%
# import numpy as np
# from loss_functions import flatten_map_v2

# def TrainEpoch(myWeedBatchGen, epochSize):
#     for i, batch in zip(range(int(epochSize/myWeedBatchGen.getBatchSize())), myWeedBatchGen):
#         batchNames, batchImages, batchMasks, batchMaps = batch
#         multidimHmaps_list = []
#         print("myWeedBatchGen.getCounter() = %s" %myWeedBatchGen.getCounter())
#         print("batchImages[0].dtype %s" %batchImages[0].dtype)
#         show_wait(np.hstack(batchImages), 2)
#         show_wait(np.hstack(batchMasks), 2)
#         for map_list in batchMaps:
#             hmaps_list = []
#             for hmap in map_list:
#                 hmaps_list.append(flatten_map_v2(hmap))
#             multidimHmaps_list.append(hmaps_list)
#         for hmaps_list in multidimHmaps_list[1:]:
#             show_wait(np.hstack(hmaps_list), 20, interpolation = cv.INTER_NEAREST)

#%%
# TrainEpoch(ValidBatchGen, 50)


#%%
# TrainEpoch(TrainBatchGen, 50)


   
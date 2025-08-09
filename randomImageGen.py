# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:25:43 2024

@author: i_bab
"""
#%%
import os
import sys
import file
import cv2 as cv
from helper import show_wait
#from imageUtils import CombinedWeed
from dataLoad import GenerateWeedDataList, WeedDataLoader, RandomWeedBatchGenerator, makeNewDirV2, loadDataFilesAsObjects
from sklearn.model_selection import train_test_split
from dataLoad import genDataFiles
import argparse
from parsingFunctions import logArgs
# from pathlib import PurePath

    
#%%
ROOT_DIR = file.ROOT_DIR
gDefaultModelDir = os.path.join(file.ROOT_DIR,'model')
gDefaultModelFiles = gDefaultModelDir
gDefaultModelPath = os.path.join(gDefaultModelDir, gDefaultModelFiles.split('.')[0])
gDefaultDataDir =  os.path.abspath(os.path.join(ROOT_DIR, os.pardir))

def getArguments():
    parser = argparse.ArgumentParser(
                        prog='Synthetic Dandelion Image Generator',
                        description='Generate synthetic training and validation data sets',
                        epilog='Text at the bottom of help')
    
    parser.add_argument('-t', '--mTrainSize', default=3000, type=int, 
                        help='Amount of training data to generate')
    
    parser.add_argument('-v', '--mValidSize', default=0, type=int, 
                        help='Amount of validation data to generate')
    
    parser.add_argument('-d', '--mDstDim', nargs='+', default=[448]*2, type=int,
                        help='Height and Width dimensions of desired image shape i.e. H W')
    
    parser.add_argument('-s', '--mSrcDim', nargs='+', default=[480, 640], type = int,
                        help='Height and Width dimensions of source image shape i.e. H W')
    
    parser.add_argument('-w', '--mWeedPath', default=os.path.join(*[gDefaultDataDir, "data2019", "1", "train1"]),
                        help='Full path weed image folder')
     
    parser.add_argument('-m', '--mMaskPath', default=os.path.join(*[gDefaultDataDir, "data2019", "1", "train1_contrast_masks_clean"]),
                        help='Full path mask image folder')
    
    parser.add_argument('-g', '--mGrassPath', nargs='+', default=[os.path.join(*[gDefaultDataDir, "data2019", "0","train1_0_grass"])],
                        help='Full path grass image folder')
    
    parser.add_argument('-ex', '--mExtClrSrcPath', nargs='+',
                        help='Use another directory for color transfer')
    
    parser.add_argument('-n', '--mWeedSamples', default=4, type=int, 
                        help='Max number of weeds to sample per synthetic image')
    
    parser.add_argument('-o', '--mOutputDirPath', default=os.path.join(*[gDefaultDataDir, "data2019", "synth"]),
                        help='Full path of container folder for both training and validation subfolders')
    
    parser.add_argument('-f', '--mFloat', nargs ='?', default=False, const=True, 
                        help='If True, will output images as floats')
    
    parser.add_argument('-x', '--mDownFactor', default=32, type=int, 
                        help='Downsampling factor, by which input is divided to smallest tensor H, W resolution')
    
    parser.add_argument('-r', '--mNMaps', default=3, type=int, 
                        help='Number of output map resolutions to generate')
    
    parser.add_argument('-vf', '--mValidFraction', default=0.3, type=float, 
                        help='Fraction between 0 and 1 of source images to use for validation set generation')
    
    parser.add_argument('-b', '--mBlend', default=1, type=int, 
                        help='Use Gaussian pyramid blending 0 or 1')
    
    parser.add_argument('-c', '--mColorTrans', default=1, type=int, 
                        help='Use color transfer 0 or 1')
    return parser


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
if __name__ == '__main__':        
    # ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))
    parser = getArguments()
    wArgs = parser.parse_args()
    src_dir = wArgs.mMaskPath
    pts_file = 'mask_centers.json' 
    src_dir2 = wArgs.mWeedPath
    
    Ho, Wo = wArgs.mSrcDim
    H, W = Ho, Wo
    
    dim = W,H
    scale_x, scale_y = W/Wo, H/Ho
    
    dstDim = tuple(wArgs.mDstDim[::-1])
    
    wDownFactor = wArgs.mDownFactor
    wNMaps = wArgs.mNMaps
    
    dim_grid_list = [(2**i*dstDim[0]//wDownFactor, 2**i*dstDim[1]//wDownFactor) for i in range(wNMaps)]
        
    src_path3 = wArgs.mGrassPath
    
    src_path_ext_clr = wArgs.mExtClrSrcPath
    
    
    wColorTrans = bool(wArgs.mColorTrans)
    wBlend = bool(wArgs.mBlend)

    
    weedSampleSize = wArgs.mWeedSamples
    wUintFlag = not wArgs.mFloat
    wOutputDirPath = wArgs.mOutputDirPath  
    trainSize = wArgs.mTrainSize
    ValidLenPercent = wArgs.mValidFraction
    validSize = wArgs.mValidSize
    if not ValidLenPercent or not validSize:
        ValidLenPercent, validSize = 0., 0   
    #%%

    weed_list = GenerateWeedDataList(src_dir, src_dir2, pts_file, scale_x, scale_y)
    grass_list = []
    for wPath in src_path3:
        grass_list+=[cv.resize(cv.imread(os.path.join(wPath,x)), dim, interpolation = cv.INTER_AREA) for x in  os.listdir(wPath)]
    
    ext_clr_src_list = []
    if src_path_ext_clr is not None:
        for path in src_path_ext_clr:    
            file_name_list = os.listdir(path)
            ext_clr_src_list += [cv.imread(os.path.join(path, name)) for name in file_name_list]
    else:
        ext_clr_src_list = [None]

        
        
    print("\nGenerating synthetic weed-on-grass images from %s weeds and %s grass samples"%(len(weed_list), len(grass_list)))

    # ValidLenPercent = 0.30
    if ValidLenPercent and validSize:
        weed_train, weed_valid, _, _ =  train_test_split(weed_list, weed_list, test_size = ValidLenPercent, random_state = 42)
        grass_train, grass_valid, _, _ =  train_test_split(grass_list, grass_list, test_size = ValidLenPercent, random_state = 42)
    else:
        weed_train = weed_list
        grass_train = grass_list
    # grass_train = grass_list[34:35]#[12:13]#[34:35]#[:1]
    # weed_train = weed_list[27:28]
    # wColorTrans, wBlend = True, True
    #%%
    batchSize = 1
    # weedSampleSize = 4
    samplerSeed = 0
    
    #%%

    #%%
    # validSize = 900
    if ValidLenPercent and validSize:
        ValidData = WeedDataLoader(weed_valid, dim)
        ValidBatchGen = RandomWeedBatchGenerator(batchSize, ValidData, dim_grid_list, grass_valid, weedSampleSize, samplerSeed, wColorTrans, wBlend, ext_clr_src_list)
        ValidBatchGen.setBatchDim(dstDim)
        ValidBatchGen.setTranLimits((1/5, 1/2), 0.5, 15)
        ValidBatchGen.setNoRepeat(False)
        ValidBatchGen.setSize(validSize)
    
    #%%
    # trainSize = 3000
    TrainData = WeedDataLoader(weed_train, dim)
    TrainBatchGen = RandomWeedBatchGenerator(batchSize, TrainData, dim_grid_list, grass_train, weedSampleSize, samplerSeed, wColorTrans, wBlend, ext_clr_src_list)
    TrainBatchGen.setBatchDim(dstDim)
    TrainBatchGen.setTranLimits((1/5, 1/2), 0.5, 15)
    TrainBatchGen.setNoRepeat(False)
    TrainBatchGen.setSize(trainSize)
    
    #%%
    if src_path_ext_clr is not None:
        wExternal=True
    else:
        wExternal=False
        
    iDestFolder = "{}_dim_{}_sample_{}_frac_{:.0e}_res_{}_blend_{}_clr_{}_ext_{}".format(trainSize, dstDim[0], weedSampleSize, 
                                                                                  1-ValidLenPercent, wNMaps, wBlend, wColorTrans, wExternal)
    
    iValDestFolder = "{}_dim_{}_sample_{}_frac_{:.0e}_res_{}_blend_{}_clr_{}_ext_{}".format(validSize, dstDim[0], weedSampleSize, 
                                                                                  ValidLenPercent, wNMaps, wBlend, wColorTrans, wExternal)    
    if not wUintFlag:
        iDestFolder +='_float'  
        iValDestFolder +='_float'  
    else:
        iDestFolder +='_uint'
        iValDestFolder +='_uint'
        
    iDestPath = makeNewDirV2(wOutputDirPath, iDestFolder,'tr', 0)
    
    if ValidLenPercent and validSize:
        iValidDestPath = makeNewDirV2(wOutputDirPath, iValDestFolder, 'val', 0)
    
    # %%
    genDataFiles(TrainBatchGen, iDestPath, iUintFlag=wUintFlag)
    wScriptName=os.path.splitext(os.path.basename(__file__))[0]    
    wArgLogName=wScriptName +'_args.csv'
    logArgs(wArgs, iDestPath, wArgLogName)
    with open(os.path.join(iDestPath, wArgLogName), 'a') as wFile:
        wFile.write('\n\n'+' '.join(sys.argv))
        
    print('\nSaved argument Log to: %s'%wArgLogName)    
    if ValidLenPercent and validSize:
        genDataFiles(ValidBatchGen, iValidDestPath, iUintFlag =wUintFlag)
    
    
    #%%
    # TrainEpoch(TrainBatchGen, 2)
    
    #%%
    # TrainEpoch(ValidBatchGen, 2)
    
    
    
    
       
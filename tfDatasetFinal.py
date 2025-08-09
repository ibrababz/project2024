# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:10:20 2025

@author: User
"""

from dataLoad import loadDataFilesAsObjectsMP
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import sys

import keras_cv
import tensorflow as tf
# from customLayers import RandomRotationHMap
from dataLoad import getImageListFromBatch, getMapListsFromBatch
# from dataLoad import ProcessMapList3D
from loss_functions import act_list_3D
import multiprocessing as mp
from timeit import time

import numpy as np
import matplotlib.pyplot as plt
from dataLoad import flatten_map_v2
from pathlib import Path
import argparse
from parsingFunctions import logArgs
    
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd()))
gParentDir = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))

parser = argparse.ArgumentParser(
                    prog='Convert to TFData',
                    description='Converting my custom data-format to a TF Dataset',
                    epilog='Text at the bottom of help')

parser.add_argument('mDataDir', type=Path,
                    help='Path to directory for data source converted to TFData')
# example 1 wLoadSubPath = os.path.join('data2019','synth','test_01_tr_7000_dim_448_sample_4_frac_1e+00_res_3_blend_True_clr_True_uint')
# example 1 wLoadSubPath = os.path.join('data4K', 'valid_real_448_res2')
# example 1  wLoadSubPath = os.path.join('data4K', 'train_real_448_res2')
# example 1  wLoadSubPath = os.path.join('data4K', 'test_real_448_res2')

parser.add_argument('-p', '--mStandardizePath', type=Path, 
                    help='Path to directory from which to compute target mean and standard deviation')

parser.add_argument('-w', '--mSampleWeights', type=float, default = 0.0,
                    help='add sample weights')

parser.add_argument('-mn', '--mMean', nargs= '+', type=float,
                    help='add sample weights')

parser.add_argument('-std', '--mStdDev', nargs= '+', type=float,
                    help='add sample weights')

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
    
if sys.stdin and sys.stdin.isatty():
    gFromShell=True
    ROOT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
    gScriptName=os.path.splitext(os.path.basename(__file__))[0]
else:
    gFromShell=False
    ROOT_DIR = os.path.abspath(os.path.join(os.getcwd()))
    gScriptName='IDE'
    
    

if __name__ == '__main__':
    
    # wLoadSubPath = os.path.join('data2019','synth','test_01_tr_7000_dim_448_sample_4_frac_1e+00_res_3_blend_True_clr_True_uint')
    # wLoadSubPath = os.path.join('data4K', 'valid_real_448_res2')
    # wLoadSubPath = os.path.join('data4K', 'train_real_448_res2')
    # wLoadSubPath = os.path.join('data4K', 'test_real_448_res2')
    
    # wLoadSubPath = os.path.join('data2019','synth','test_02_tr_7000_dim_448_sample_4_frac_1e+00_res_3_blend_True_clr_True_uint')
    # wLoadSubPath = os.path.join('data2019','synth','test_04_tr_10000_dim_448_sample_4_frac_1e+00_res_3_blend_True_clr_True_uint')

    wArgs = parser.parse_args()
    wSrcPath = os.path.abspath(wArgs.mDataDir)
    wSubPath = wSrcPath.split(os.sep)
    if 'synth' in wSubPath:
        wSubPath = wSubPath[-3:]
    else:
        wSubPath = wSubPath[-2:]
    wSampleWeights = wArgs.mSampleWeights


    wStandardize = wArgs.mStandardizePath
    wTargetChannelMean = None
    wTargetChannelStd = None
    
    wMean = wArgs.mMean
    wStdDev = wArgs.mStdDev
    if wMean is not None and wStdDev is not None:
        wTargetChannelMean = np.array(wMean)
        wTargetChannelStd = np.array(wStdDev)
    
    elif wStandardize is not None:
        
        wStandardize = os.path.abspath(wStandardize)
        print(f"Standardizing image data to match data from:\n{wStandardize}\n")
        wDataToMatch= loadDataFilesAsObjectsMP(wStandardize, iProcesses=4)
        wImageToMatchList = np.array([wX.getImage() for wX in wDataToMatch])
        wMatchLen = len(wImageToMatchList)
        wChannelMeanPerImage = np.mean(wImageToMatchList, axis=(1,2))
        wTargetChannelMean = np.mean(wChannelMeanPerImage, axis = 0)
        
        wChannelStdPerImage = np.std(wImageToMatchList, axis= (1,2))
        wTargetChannelStd = np.mean(wChannelStdPerImage, axis = 0)
                
        
        # print(wChannelMeanPerImage)
        print("Target Mean:")
        print(wTargetChannelMean)
        
        # print(wChannelStdPerImage)
        print("Target Std:")
        print(wTargetChannelStd)
        
        del wImageToMatchList
        del wDataToMatch
        
        time.sleep(1)


        
    wData= loadDataFilesAsObjectsMP(wSrcPath, iProcesses=4)
    wDataLen = len(wData)
    print("Data Size: %s\n"%wDataLen)
    
    H,W,C = wData[0].getImage().shape
    
    if wTargetChannelMean is not None and wTargetChannelStd is not None:
        
        wBufferSize = 500
        wSteps = int(np.ceil(wDataLen/wBufferSize))
        wImageList = np.zeros((wDataLen, H, W, C), dtype=np.uint8)
        
        for i in range(wSteps):
            wStart, wEnd = i*wBufferSize, (i+1)*wBufferSize
            if wEnd >= wDataLen:
                wEnd = wDataLen-1
            wImageListSegment = np.array([wX.getImage() for wX in wData[wStart:wEnd]])
            
            wChannelMeanPerImage = np.mean(wImageListSegment, axis=(1,2))
            wSourceChannelMean = np.mean(wChannelMeanPerImage, axis = 0)
            
            wChannelStdPerImage = np.std(wImageListSegment, axis= (1,2))
            wSourceChannelStd = np.mean(wChannelStdPerImage, axis = 0)
                    
            
            # print(wChannelMeanPerImage)
            print("Source Mean:")
            print(wSourceChannelMean)
            
            # print(wChannelStdPerImage)
            print("Source Std:")
            print(wSourceChannelStd)
            
            wImageListSegment = (wImageListSegment-wSourceChannelMean)/wSourceChannelStd
            wImageListSegment = wImageListSegment*wTargetChannelStd + wTargetChannelMean
            wImageListSegment = np.uint8(np.round(np.clip(wImageListSegment, 0., 255.)))
            
            
            
            wChannelMeanPerImage = np.mean(wImageListSegment, axis=(1,2))
            wChannelMean = np.mean(wChannelMeanPerImage, axis = 0)
            
            wChannelStdPerImage = np.std(wImageListSegment, axis= (1,2))
            wChannelStd = np.mean(wChannelStdPerImage, axis = 0)
                    
            
            # print(wChannelMeanPerImage)
            print("\nNew Mean:")
            print(wChannelMean)
            
            # print(wChannelStdPerImage)
            print("New Std:")
            print(wChannelStd)
            
            wImageList[wStart:wEnd] = wImageListSegment
    else:
        wImageList = np.array([wX.getImage() for wX in wData])
        
    # wMaxDepth= max([wX.getMapList()[0].shape[-1] for wX in wData])
    wMaxDepth = 9
    wMapListList =[]
    for wX in wData:
        wMapList = wX.getMapList()
        wNo = wMaxDepth - wMapList[0].shape[-1]
        if wNo:
            wPaddedMapList =[]
            for wMap in wMapList:
                H, W, _ = wMap.shape
                wPaddedMapList.append(np.concatenate([wMap, np.zeros((H,W,wNo))], axis=2))
            wMapList = wPaddedMapList
        wMapListList.append(wMapList)
    wDim=wImageList[0].shape[:-1]
    wMapListList = [[wMapList[0] for wMapList in wMapListList], 
                    [wMapList[1] for wMapList in wMapListList], 
                    [wMapList[2] for wMapList in wMapListList]]
    

    
    # wTFData = tf.data.Dataset.from_tensor_slices((wImageList, wMapListList[2]))#, wMapListList[1], wMapListList[2]))

    wSaveDict = {"images":wImageList,  "segmentation_masks":wMapListList[2]}
    if wSampleWeights:
        wSaveDict.update({'weights': wSampleWeights*np.ones((wImageList.shape[0],))})
    wTFData = tf.data.Dataset.from_tensor_slices(wSaveDict)#, wMapListList[1], wMapListList[2]))
    wName = wSubPath[-1]
    if wSampleWeights:
        wName  = '_'.join([wName, f'weighted_{wSampleWeights:.3f}'.replace('.', 'p')])
    wDataSavePath = makeNewDirV2(os.path.join(gParentDir, 'tfdataset', *wSubPath[:-1]), wName, '')
    

    # wDataSavePath = os.path.join(gParentDir, 'tfdataset', *wSubPath)
    os.makedirs(wDataSavePath, exist_ok=True)

    wTFData.save(wDataSavePath)
    
    print('\nSave dir: %s'%wDataSavePath)
    wArgLogName=gScriptName +'_args.csv'
    logArgs(wArgs, wDataSavePath, wArgLogName)
    print('\nSaved argument Log to: %s'%wArgLogName)
    with open(os.path.join(wDataSavePath, wArgLogName), 'a') as wFile:
        wFile.write('\n\n'+' '.join(sys.argv))    
    
    
    
    # del wData
    # del wImageList
    # del wMapListList
    # del wMapList
    # del wMap
    # del wX
    # del wPaddedMapList
    # for wX in wTFData:
    #     break       
    
    # #Affine Pipeline
        
    # from customRotLayer import RandomRotationHMap
    # from customFourierLayer import FourierMixHMap
    
    # tf.random.set_seed(0)
    # wRotMap =RandomRotationHMap(0.13, fill_mode='constant', seed=0)
    # wRot =keras_cv.layers.RandomRotation(0.13, fill_mode='constant', seed=0)
    # wFlip = keras_cv.layers.RandomFlip(mode='horizontal_and_vertical', rate=0.5, seed=0)
    # wTrans = keras_cv.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="constant", seed=0)
    # wShear = keras_cv.layers.RandomShear(x_factor=0.3, y_factor=0.3, fill_mode="constant", interpolation="bilinear", seed =0)
    # wZoom = keras_cv.layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode="constant", seed=0)
    
    # def affine(ioX, iOperations):
    #     for wOp in iOperations:
    #         ioX = wOp(ioX)
    #     return ioX

    # wIm, wMap = wX['images'], wX['segmentation_masks']
    # plt.imshow(np.uint8(wIm.numpy()[...,::-1]))
    # plt.show()
    
    # plt.imshow(flatten_map_v2(wMap, inv=1))
    # plt.show()
    # for wOp in [wRot, wRotMap]:
    #     wAugX = wOp(wX)
    #     wAugIm, wAugMap = wAugX['images'], wAugX['segmentation_masks']
    #     plt.imshow(np.uint8(wAugIm.numpy()[...,::-1]))
    #     plt.show()
        
    #     plt.imshow(flatten_map_v2(wAugMap, inv=1))
    #     plt.show()    
    # wLoadSubPath = os.path.join('data4K', 'valid_real_448_res2')
    # iSrcPath = os.path.join(gParentDir,  wLoadSubPath)# wArgs.mTrainDir
    
    # wAffineAugments = keras_cv.layers.Augmenter([wFlip, wTrans, wShear, wZoom, wRotMap])
    # wAugX = wAffineAugments(wX)
    # wAugIm, wAugMap = wAugX['images'], wAugX['segmentation_masks']
    # plt.imshow(np.uint8(wAugIm.numpy()[..., ::-1]))
    # plt.show()

    # plt.imshow(flatten_map_v2(wAugMap, inv=1))
    # plt.show()
   
    # wRandGrayIm = tf.keras.layers.RandomGrayscale(factor=0.5, seed=0)
    # #vectorizeable layers
    # wGrayIm = keras_cv.layers.Grayscale(output_channels=3, seed=0)
    
    # #non-vectorizeable layers        
    
    # wChannelShuffle = keras_cv.layers.ChannelShuffle(groups=3, seed=0)
    # wSolar = keras_cv.layers.Solarization(value_range=(0, 255), seed=0)
    # wGrid = keras_cv.layers.GridMask(rotation_factor=0.2, seed=0)
    
    # wPost1 = keras_cv.layers.Posterization(value_range=(0, 255), bits=1, seed=0)
    # wPost2 = keras_cv.layers.Posterization(value_range=(0, 255), bits=2, seed=0)
    # wPost3 = keras_cv.layers.Posterization(value_range=(0, 255), bits=3, seed=0)
    # wPost4 = keras_cv.layers.Posterization(value_range=(0, 255), bits=4, seed=0)
    
    # wEq = keras_cv.layers.Equalization(value_range=(0, 255), seed=0) 
    # wClrJitter = keras_cv.layers.RandomColorJitter(value_range=(0, 255), brightness_factor=0.2, contrast_factor=0.2, saturation_factor=(0.5,0.8), hue_factor=(0.3,0.6), seed=0)
    # wClrDegen = keras_cv.layers.RandomColorDegeneration(factor=(0.3, 0.6), seed=0)
    # wHue = keras_cv.layers.RandomHue(factor=(0.3, 0.6), value_range=(0, 255), seed=0)
    # wSat = keras_cv.layers.RandomSaturation(factor=(0.5,0.8), seed=0)
    # wContrast = keras_cv.layers.RandomContrast(value_range=(0, 255), factor=0.2, seed=0)
    
    # wCutout = keras_cv.layers.RandomCutout(height_factor=0.05, width_factor=0.05, fill_mode="constant",)
    
    # wFourier = keras_cv.layers.FourierMix(seed=0)
    # wFourierHMap = FourierMixHMap(seed=0)

    
    # wOneOfPost = keras_cv.layers.RandomChoice(layers=[wPost1, wPost2, wPost3, wPost4], seed=0)
    # wTwoOf = keras_cv.layers.RandomAugmentationPipeline(layers=[wGrayIm, wChannelShuffle, wSolar, wGrid, wEq, wOneOfPost, wClrJitter, wClrDegen], augmentations_per_image = 3, rate = 0.5, seed=0)
    # # wSomeTimes = keras_cv.layers.RandomApply(layer=wOneOf, rate=0.5, seed=0)
    
    # wIm, wMap = wX['images'], wX['segmentation_masks']
    # plt.imshow(np.uint8(wIm.numpy()[...,::-1]))
    # plt.show()
    
    # plt.imshow(flatten_map_v2(wMap, inv=1))
    # plt.show()
    # for i in range(4):
    #     for wOp in [wRotMap, wRot]:
    #         wAugX = wOp(wX)
    #         wAugIm, wAugMap = wAugX['images'], wAugX['segmentation_masks']
    #         plt.imshow(np.uint8(wAugIm.numpy()[...,::-1]))
    #         plt.show()
            
    #         plt.imshow(flatten_map_v2(wAugMap, inv=1))
    #         plt.show()
        
    
    # wAugments = keras_cv.layers.Augmenter([wTwoOf, wFlip, wTrans, wShear, wZoom, wRotMap])#, wFourierHMap])
    # i=0
    # for wX in wTFData:
    #     wAugX = wAugments(wX)
    
    #     wAugIm, wAugMap = wAugX['images'], wAugX['segmentation_masks']
    #     plt.imshow(np.uint8(wAugIm.numpy()[...,::-1]))
    #     plt.show()
        
    #     plt.imshow(flatten_map_v2(wAugMap, inv=1))
    #     plt.show()
    #     i+=1
    #     if i > 10:
    #         break
        
    # for wBatchSize in [8,16]:#, 32, 64]*3:
    #     print(f"Batch size: {wBatchSize}")
    #     wAugTFData = wTFData.batch(wBatchSize).map(wAugments, num_parallel_calls=tf.data.AUTOTUNE)
    #     wT0 = time.perf_counter()
    #     for wAugX in wAugTFData:
    #         pass
    #     print(f"augment time: {time.perf_counter() - wT0:.2f}")        
    # wAugIm, wAugMap = wAugX.values()
    # for i in range(len(wAugIm)):
    #     plt.imshow(np.uint8(wAugIm.numpy()[i,..., ::-1]))
    #     plt.show()
    
    #     plt.imshow(flatten_map_v2(wAugMap[i], inv=1))
    #     plt.show()

    # for i in range(50):
    #     wImAug=np.uint8(wTwoOf(wIm[None,...]).numpy())
    #     plt.imshow(wImAug[0,...,::-1])
    #     plt.show()
    
    # import pydoc
    
    # wText=  pydoc.render_doc(keras_cv.layers.RandomGaussianBlur, r"Help on %s")
    # with open('help.txt', 'w') as wFile:
    #     wFile.write(keras_cv.layers.RandomGaussianBlur.__doc__)
        
        
    # # wRandImPipeLine = tf.keras.layers.Pipeline([wRandRotIm, wRandShrIm, wRandTrnsIm, wRandGrayIm])
    # # wRandMaskPipeLine = tf.keras.layers.Pipeline([wRandRotMask, wRandShrMask, wRandTrnsMask])
    
    # # wImAug = wRandRotIm(wIm)
    # # wMaskAug = wRandRotMask(wMask)

    # # for i in range(100):
    # #     wData=tf.constant(0)
    # #     print(someOf(wData, (0,3), wOpList, i).numpy())

        
    # # for i in range(100):
    # #     wData=tf.constant(0)
    # #     print(oneOf(wData, wOpList, i))
    
    # # wSeed = 0
    # # wLen=3
    # # wList = tf.range(0, wLen)
    # # for j in range(5):
    # #     print('\nResetting Global')
        
    # #     for i in range(wLen):
    # #         tf.random.set_seed(j)
    # #         wStatelessRandomSeed = i+j*wLen
    # #         wShuffle = tf.random.shuffle(wList,wStatelessRandomSeed)
    # #         wN = tf.random.stateless_uniform([1], (wStatelessRandomSeed,0), minval=0, maxval=wLen+1, dtype=tf.int32)
    # #         print(wShuffle.numpy(), wN.numpy(), wStatelessRandomSeed)
            
    
    

    # # #             wAugmentsArgs = {wKey: wNumpy for wKey, wNumpy in zip(wKeyList, wNumpyList)}
    # # #             oList = augSeqWrapperV2(wSeq, wAugmentsArgs)
    # # #             return oList
        
    # # #         wAugment = wTFData.batch(batch).map(lambda x,y: tf.py_function(augmentImage2, [x,y], [tf.uint8] +[tf.float32]*1), num_parallel_calls=tf.data.AUTOTUNE)#, tf.float32, tf.float32, tf.float32]
            
    # # #         wT0 = time.perf_counter()
    # # #         for wX in wAugment:
    # # #             pass
    # # #         print(f"augment time: {time.perf_counter() - wT0:.2f}")
    
    # # from dataLoad import flatten_map_v2
    # # import matplotlib.pyplot as plt
        
    # # for i in range(10):
    # #     wImAugBatch = wRandImPipeLine([wIm]*8)
    # #     wMaskAugBatch = wRandMaskPipeLine([1-wMask]*8)
    # #     wImAug=wImAugBatch[6]
    # #     wMaskAug = wMaskAugBatch[6]
    # #     plt.imshow(np.uint8(wImAug[...,::-1]))
    # #     plt.show()
    # #     plt.imshow(flatten_map_v2(wMaskAug, inv=1))
    # #     plt.show()
    # #     wResize =tf.image.resize(wMaskAug, tuple(wD//2 for wD in wMaskAug.shape[:-1]), method= tf.image.ResizeMethod.AREA)
    # #     plt.imshow(flatten_map_v2(wResize, inv=1))
    # #     plt.show()
        
    # #     wResize =tf.image.resize(wMaskAug, tuple(wD//4 for wD in wMaskAug.shape[:-1]), method= tf.image.ResizeMethod.AREA)
    # #     plt.imshow(flatten_map_v2(wResize, inv=1))
    # #     plt.show()
   


    
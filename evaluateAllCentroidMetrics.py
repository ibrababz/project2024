# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:34:54 2024

@author: i_bab
"""


import numpy as np
import os
import file
from helper import show_wait
from BinaryMetricsClass import BinaryMetricsManager
from imageUtils import putTitleOnImage

import matplotlib.pyplot as plt


#%%
if __name__ =='__main__':
    ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir)) 
    
#%%Segmentation Metrics
    wLoadFolder, wLoadSubFolder = 'data4k', '1_test_png_for_segmenation_compare_outputs_sifted_square'
    wLoadDir= os.path.join(ROOT_DIR, wLoadFolder, wLoadSubFolder)
    wScale = 0.5
    wBnMetrics = BinaryMetricsManager(wLoadDir)
    wBnMetrics.loadData()

    H, W = wBnMetrics.getShape()
    wDataDir = 'data4k'
    wFolder = '1'
    wDataPath = os.path.join(ROOT_DIR, wDataDir, wFolder)

    
    wPointsFolder = '1'
    wPointsFile = "plant_centers_sifted_FINAL.json"
    wPointsPath = os.path.join(ROOT_DIR, wDataDir, wPointsFolder, wPointsFile) 
    wPointsDir = os.path.join(ROOT_DIR, wDataDir, wPointsFolder)
    Wo, Ho = 4056, 3040
    wShape = (H,W,3)
    wDim = (W,H)
    wScaleX, wScaleY = W/Wo, H/Ho

    wBnMetrics.setTruthDir(wDataPath)    
    wBnMetrics.setTruthLabelsDir(wPointsDir)
    wBnMetrics.setTruthLabelsFile(wPointsFile)
    wBnMetrics.setTruthScale(wScaleX, wScaleY)
    wBnMetrics.loadTruthSet()
    
    wHypotenuse = np.sqrt(W*W + H*H)
    wMaxDist = wHypotenuse
    wMinDist = 0 #wHypotenuse/8
    wStep = wHypotenuse/32


    for wThresh in np.arange(wMinDist, wMaxDist, wStep)[:]:
        wThresh = np.round(wThresh,3)
        wBnMetrics.computeMetrics(wThresh, iMinPercentArea=0.003)
        # show_wait(wBnMetrics.drawPredictionAtIdxAtThresh(0, wThresh),0.75)
    

 

        
    wBnMetrics.evaluateMetrics(iNorm=True)
    wBnMetrics.printMetrics()
    
    wMetrics, _ = wBnMetrics.getMetrics()
    wMetricsPredicted, wNoUnpredict= wBnMetrics.getMetricsPredicted()
    # print(wMetrics)
    # for wThresh in wBnMetrics.getClusterThresholds()[:5]:
    #     show_wait(wBnMetrics.drawPredictionAtIdxAtThresh(3, wThresh),0.5)
    
    
    

    print(len(wMetrics))
    wValues = [wMetrics[wKey] for wKey in wMetrics]
    

    
#%%    

    wSaveFolder = 'project2024'
    wSaveSubFolder = 'resnet_test_13_real_8bit_res2_ep_(0, 1000)_lr_1e-04'
    wBnMetrics.setSaveDir(os.path.join(ROOT_DIR,wSaveFolder, wSaveSubFolder ))    

    wThresholds = wBnMetrics.getClusterThresholds()    
    for wThresh in wThresholds:
        wBnMetrics.saveImagesForThresh(wThresh, iFolderPrefix='seg_thresh_')
    
#%% HeatMap Metrics    
    wLoadFolder2 = 'project2024'
    wLoadSubFolder2 = os.path.join('resnet_test_13_real_8bit_res2_ep_(0, 1000)_lr_1e-04', 'test_output_act')
    wLoadDir2= os.path.join(ROOT_DIR, wLoadFolder2, wLoadSubFolder2)
    wBnMetrics2 = BinaryMetricsManager(wLoadDir2)
    wBnMetrics2.loadData()
   
    H2, W2 = wBnMetrics2.getShape()
    wDataDir2 = 'data4k'
    wFolder2 = '1'
    wDataPath2 = os.path.join(ROOT_DIR, wDataDir2, wFolder2)

    
    wPointsFolder = '1'
    wPointsFile = "plant_centers_sifted_FINAL.json"
    wPointsPath = os.path.join(ROOT_DIR, wDataDir2, wPointsFolder, wPointsFile) 
    wPointsDir = os.path.join(ROOT_DIR, wDataDir2, wPointsFolder)
    Wo, Ho = 4056, 3040
    wShape2 = (H2,W2,3)
    wDim2 = (W2,H2)
    wScaleX2, wScaleY2 = W2/Wo, H2/Ho

    wBnMetrics2.setTruthDir(wDataPath2)    
    wBnMetrics2.setTruthLabelsDir(wPointsDir)
    wBnMetrics2.setTruthLabelsFile(wPointsFile)
    wBnMetrics2.setTruthScale(wScaleX2, wScaleY2)
    wBnMetrics2.loadTruthSet()
  
    wHypotenuse2 = np.sqrt(W2*W2 + H2*H2)
    wMaxDist2 = wHypotenuse2
    wMinDist2 = 0 #wHypotenuse2/8
    wStep2 = wHypotenuse2/32
    

    for wThresh in np.arange(wMinDist2, wMaxDist2, wStep2)[0:1]:
        wThresh = np.round(wThresh,3)
        wBnMetrics2.computeMetrics(wThresh, iMinPercentArea = 0.0)
        # show_wait(wBnMetrics2.drawPredictionAtIdxAtThresh(0, wThresh),0.75)
 

    wBnMetrics2.evaluateMetrics(iNorm=True)
    
    
    wBnMetrics2.printMetrics()    
    wMetrics2, _ = wBnMetrics2.getMetrics()
    wMetricsPredicted2, wNoUnpredict2= wBnMetrics2.getMetricsPredicted()
    wBnMetrics2.getClusterThresholds()
    

    # print(wMetrics2)
    print(len(wMetrics2))
    wValues2 = [wMetrics2[wKey] for wKey in wMetrics2]

#%%
    
    wSaveFolder2 = 'project2024'
    wSaveSubFolder2 = 'resnet_test_13_real_8bit_res2_ep_(0, 1000)_lr_1e-04'
    wBnMetrics2.setSaveDir(os.path.join(ROOT_DIR,wSaveFolder2, wSaveSubFolder2 ))    
    
    
    wThresholds2 = wBnMetrics2.getClusterThresholds()    
    for wThresh2 in wThresholds2:
        wBnMetrics2.saveImagesForThresh(wThresh2, iFolderPrefix='act_')
#%%    
    wLoadFolder3 = 'project2024'
    wLoadSubFolder3 = os.path.join('resnet_test_13_real_8bit_res2_ep_(0, 1000)_lr_1e-04', 'test_output_cent')
    wLoadDir3= os.path.join(ROOT_DIR, wLoadFolder3, wLoadSubFolder3)
    wBnMetrics3 = BinaryMetricsManager(wLoadDir3)
    wBnMetrics3.loadData()
   
    H3, W3 = wBnMetrics3.getShape()
    wDataDir3 = 'data4k'
    wFolder3 = '1'
    wDataPath3 = os.path.join(ROOT_DIR, wDataDir3, wFolder3)

    
    wPointsFolder = '1'
    wPointsFile = "plant_centers_sifted_FINAL.json"
    wPointsPath = os.path.join(ROOT_DIR, wDataDir3, wPointsFolder, wPointsFile) 
    wPointsDir = os.path.join(ROOT_DIR, wDataDir3, wPointsFolder)
    Wo, Ho = 4056, 3040
    wShape3 = (H3,W3,3)
    wDim3 = (W3,H3)
    wScaleX3, wScaleY3 = W3/Wo, H3/Ho

    wBnMetrics3.setTruthDir(wDataPath3)    
    wBnMetrics3.setTruthLabelsDir(wPointsDir)
    wBnMetrics3.setTruthLabelsFile(wPointsFile)
    wBnMetrics3.setTruthScale(wScaleX3, wScaleY3)
    wBnMetrics3.loadTruthSet()
  
    wHypotenuse3 = np.sqrt(W3*W3 + H3*H3)
    wMaxDist3 = wHypotenuse3
    wMinDist3 = 0 #wHypotenuse3/8
    wStep3 = wHypotenuse3/32
    

    for wThresh in np.arange(wMinDist3, wMaxDist3, wStep3)[0:1]:
        wThresh = np.round(wThresh,3)
        wBnMetrics3.computeMetrics(wThresh, iMinPercentArea = 0.0)
        # show_wait(wBnMetrics3.getDataAtIdx(i).drawPredictions(wThresh),0.5)
    
    wBnMetrics3.evaluateMetrics(iNorm=True)
    wBnMetrics3.printMetrics()
    wMetrics3, _ = wBnMetrics3.getMetrics()
    wMetricsPredicted3, wNoUnpredict3= wBnMetrics3.getMetricsPredicted()
    # print(wMetrics3)
    print(len(wMetrics3))
    wValues3 = [wMetrics3[wKey] for wKey in wMetrics3]
    

    #%%

    wSaveFolder3 = 'project2024'
    wSaveSubFolder3 = 'resnet_test_13_real_8bit_res2_ep_(0, 1000)_lr_1e-04'
    wBnMetrics3.setSaveDir(os.path.join(ROOT_DIR,wSaveFolder3, wSaveSubFolder3 ))    
    
    
    wThresholds3 = wBnMetrics3.getClusterThresholds()    
    for wThresh3 in wThresholds3:
        wBnMetrics3.saveImagesForThresh(wThresh3, iFolderPrefix='cent_')
        
#%%
    wLoadFolder4 = 'project2024'
    wLoadSubFolder4 = os.path.join('resnet_test_13_real_8bit_res2_ep_(0, 1000)_lr_1e-04', 'test_output_truth_acts')
    wLoadDir4= os.path.join(ROOT_DIR, wLoadFolder4, wLoadSubFolder4)
    wBnMetrics4 = BinaryMetricsManager(wLoadDir4)
    wBnMetrics4.loadData()
    
    H4, W4 = wBnMetrics4.getShape()
    wDataDir4 = 'data4k'
    wFolder4 = '1'
    wDataPath4 = os.path.join(ROOT_DIR, wDataDir4, wFolder4)
    
    
    wPointsFolder = '1'
    wPointsFile = "plant_centers_sifted_FINAL.json"
    wPointsPath = os.path.join(ROOT_DIR, wDataDir4, wPointsFolder, wPointsFile) 
    wPointsDir = os.path.join(ROOT_DIR, wDataDir4, wPointsFolder)
    Wo, Ho = 4056, 3040
    wShape4 = (H4,W4,3)
    wDim4 = (W4,H4)
    wScaleX4, wScaleY4 = W4/Wo, H4/Ho
    
    wBnMetrics4.setTruthDir(wDataPath4)    
    wBnMetrics4.setTruthLabelsDir(wPointsDir)
    wBnMetrics4.setTruthLabelsFile(wPointsFile)
    wBnMetrics4.setTruthScale(wScaleX4, wScaleY4)
    wBnMetrics4.loadTruthSet()

#%% Combine output images for segmentation, activation, centroid methods and Truth
    wMetricsPlotsFolder = 'metrics_plots'
    wMetricsPlotsDir = os.path.join(ROOT_DIR, wSaveFolder, wSaveSubFolder, wMetricsPlotsFolder)
    os.makedirs(wMetricsPlotsDir, exist_ok=True)
    
    wSampleIdxList=[0,10,14]    
    
    wTruthIm = wBnMetrics4.drawRawImageRange(wSampleIdxList, iPadLast=True)
    # show_wait(wTruthIm,0.25)
    
    wThreshList = wBnMetrics.getClusterThresholds()
    wMinThreshIdx = wBnMetrics.getMinIdx()  
    wLen = len(wThreshList)
    wThreshIdx1, wThreshIdx2 = int(5*wMinThreshIdx/12), int(wMinThreshIdx/2)#int(wMinThreshIdx + (wLen - wMinThreshIdx)/2)
    wThreshIdx = [wThreshIdx1, wThreshIdx2, wMinThreshIdx]
    wThreshListShow = [wThreshList[wIdx] for wIdx in wThreshIdx]

    wSegIm = wBnMetrics.drawPredRangeForThreshRange(wSampleIdxList, wThreshListShow, iPadLast=True)
    # show_wait(wSegIm,0.25)

    
    wActIm = wBnMetrics2.drawPredRangeForThreshRange(wSampleIdxList, wBnMetrics2.getClusterThresholds(), iPadLast=False)
    # show_wait(wActIm, 0.25)
    
    wCentIm = wBnMetrics3.drawPredRangeForThreshRange(wSampleIdxList, wBnMetrics3.getClusterThresholds(), iPadLast=False)
    # show_wait(wCentIm, 0.25)

    wTruthIm = putTitleOnImage(wTruthIm, iTitle='Ground Truth')
    wSegIm = putTitleOnImage(wSegIm, iTitle='Hierarchical Clustering')
    wActIm = putTitleOnImage(wActIm, iTitle='Max-Act. (our method)', iFontScale=3)
    wCombineIm = np.hstack([wTruthIm, wSegIm, wActIm])#, wCentIm])
    # show_wait(wCombineIm, 0.2)
    plt.imsave(os.path.join(wMetricsPlotsDir, 'comparison.png'), wCombineIm[:,:,::-1])
#%%    
# #%%
#     plt.figure(1)
#     wXValues = wBnMetrics.getClusterThresholds(iNorm=True)
#     wValues, _, _ = wBnMetrics.getMetricsAsList()
#     plt.scatter(wXValues, wValues, s=10, marker='s')
#     wValues2, _, _ = wBnMetrics2.getMetricsAsList()
#     plt.plot(wXValues, wValues2*len(wValues), ':r')
#     wValues3, _ , _= wBnMetrics3.getMetricsAsList()
#     plt.scatter(wXValues, wValues3*len(wValues), c= 'green', marker='x', linewidths=1)
#     plt.title('Prediction-Truth Distance per Threshold')
#     plt.xlabel("Cluster Threshold")
#     plt.ylabel("Cost")
#     plt.axis([-0.05*wXValues[-1],1.05*wXValues[-1], 0,1.1*np.max(wValues)])
#     plt.legend(['Segmentation Cluster', 'H.map Acts. (fixed 0-threshold)', 'H.map Cents. (fixed 0-threshold)'])
#     plt.tight_layout()
#     plt.savefig(os.path.join(wMetricsPlotsDir, 'cost.png'), bbox_inches='tight')
#     # plt.xticks(np.linspace(0,1,len(wValues)))
    
# #%%
#     plt.figure(2)
#     wXValues = wBnMetrics.getClusterThresholds(iNorm=True)
#     wValues, _, _ = wBnMetrics.getMetricsAsList(True)
#     plt.scatter(wXValues, wValues, s=10, marker='s')
#     wValues2, _, _ = wBnMetrics2.getMetricsAsList(True)
#     plt.plot(wXValues, wValues2*len(wValues), ':r')
#     wValues3, _, _ = wBnMetrics3.getMetricsAsList(True)
#     plt.scatter(wXValues, wValues3*len(wValues), c= 'green', marker='x', linewidths=1)
#     plt.title('Prediction-Truth Distance per Threshold (Predicted Only)')
#     plt.xlabel("Cluster Threshold")
#     plt.ylabel("Cost")
#     plt.axis([-0.05*wXValues[-1],1.05*wXValues[-1], 0,1.1*np.max(wValues)])
#     plt.legend(['Segmentation Cluster', 'H.map Acts. (fixed 0-threshold)', 'H.map Cents. (fixed 0-threshold)'])
#     plt.tight_layout()
#     plt.savefig(os.path.join(wMetricsPlotsDir, 'cost-pred.png'), bbox_inches='tight')

#     # plt.xticks(np.linspace(0,1,len(wValues)))
    
#     #%%
#     plt.figure(3)
#     wXValues = wBnMetrics.getClusterThresholds(iNorm=True)
#     _, wValues, _ = wBnMetrics.getMetricsAsList(True)
#     plt.scatter(wXValues, wValues, s=10, marker='s')
#     _, wValues2, _ = wBnMetrics2.getMetricsAsList(True)
#     plt.plot(wXValues, wValues2*len(wValues), ':r')
#     _, wValues3, _ = wBnMetrics3.getMetricsAsList(True)
#     plt.scatter(wXValues, wValues3*len(wValues), c= 'green', marker='x', linewidths=1)
#     plt.title('Number of Unpredicted Samples per Threshold')
#     plt.xlabel("Cluster Threshold")
#     plt.ylabel("Number")
#     plt.axis([-0.05*wXValues[-1],1.05*wXValues[-1], 1.1*np.min(wValues) ,2.*np.max(wValues)])
#     plt.legend(['Segmentation Cluster', 'H.map Acts. (fixed 0-threshold)', 'H.map Cents. (fixed 0-threshold)'])
#     plt.tight_layout()
#     plt.savefig(os.path.join(wMetricsPlotsDir, 'missed.png'), bbox_inches='tight')
#     # plt.xticks(np.linspace(0,1,len(wValues)))

#%%
# Create two subplots and unpack the output array immediately
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14*.8,3*0.8), dpi=300, sharex=True)
    
    wXValues = wBnMetrics.getClusterThresholds(iNorm=True)
    wValues, _, _ = wBnMetrics.getMetricsAsList()
    ax1.scatter(wXValues, wValues, s=10, marker='s')
    wValues2, _, _ = wBnMetrics2.getMetricsAsList()
    ax1.plot(wXValues, wValues2*len(wValues), ':r')
    wValues3, _ , _= wBnMetrics3.getMetricsAsList()
    ax1.scatter(wXValues, wValues3*len(wValues), c= 'green', marker='x', linewidths=1)
    ax1.set_title('Pred-Truth Cost')
    ax1.set_xlabel("Cluster Threshold")
    ax1.set_ylabel("Cost")
    ax1.set_xlim(-0.05*wXValues[-1],1.05*wXValues[-1])
    ax1.set_ylim(0,1.1*np.max(wValues))
    # ax1.legend(['Segmentation Cluster', 'H.map Acts. (fixed 0-threshold)', 'H.map Cents. (fixed 0-threshold)'])
    # fig.tight_layout()
    # plt.savefig(os.path.join(wMetricsPlotsDir, 'cost.png'), bbox_inches='tight')
    # plt.xticks(np.linspace(0,1,len(wValues)))

    wXValues = wBnMetrics.getClusterThresholds(iNorm=True)
    wValues, _, _ = wBnMetrics.getMetricsAsList(True)
    ax2.scatter(wXValues, wValues, s=10, marker='s')
    wValues2, _, _ = wBnMetrics2.getMetricsAsList(True)
    ax2.plot(wXValues, wValues2*len(wValues), ':r')
    wValues3, _, _ = wBnMetrics3.getMetricsAsList(True)
    ax2.scatter(wXValues, wValues3*len(wValues), c= 'green', marker='x', linewidths=1)
    ax2.set_title('Pred-Truth Cost (Predicted Only)')
    ax2.set_xlabel("Cluster Threshold")
    ax2.set_ylabel("Cost")
    ax2.set_xlim(-0.05*wXValues[-1],1.05*wXValues[-1])
    ax2.set_ylim(0,1.1*np.max(wValues))
    # ax2.axis([-0.05*wXValues[-1],1.05*wXValues[-1], 0,1.1*np.max(wValues)])
    # ax2.legend(['Segmentation Cluster', 'H.map Acts. (fixed 0-threshold)', 'H.map Cents. (fixed 0-threshold)'])
    # fig.tight_layout()
    # plt.savefig(os.path.join(wMetricsPlotsDir, 'cost-pred.png'), bbox_inches='tight')

    # plt.xticks(np.linspace(0,1,len(wValues)))
    
    # plt.figure(3)
    wXValues = wBnMetrics.getClusterThresholds(iNorm=True)
    _, wValues, _ = wBnMetrics.getMetricsAsList(True)
    ax3.scatter(wXValues, wValues, s=10, marker='s')
    _, wValues2, _ = wBnMetrics2.getMetricsAsList(True)
    ax3.plot(wXValues, wValues2*len(wValues), ':r')
    _, wValues3, _ = wBnMetrics3.getMetricsAsList(True)
    ax3.scatter(wXValues, wValues3*len(wValues), c= 'green', marker='x', linewidths=1)
    ax3.set_title('Number of Missed Preds')
    ax3.set_xlabel("Cluster Threshold")
    ax3.set_ylabel("Number")
    ax3.set_xlim(-0.05*wXValues[-1],1.05*wXValues[-1])
    ax3.set_ylim(1.1*np.min(wValues) ,2.*np.max(wValues))
    # ax3.axis([-0.05*wXValues[-1],1.05*wXValues[-1], 1.1*np.min(wValues) ,2.*np.max(wValues)])
    fig.legend(['Segment. Cluster', 'ACT (our method)', 'CENT (our method)'],
               loc='center right')
    fig.tight_layout()
    fig.savefig(os.path.join(wMetricsPlotsDir, 'costMetricsAll.png'), bbox_inches='tight')
    # plt.xticks(np.linspace(0,1,len(wValues)))
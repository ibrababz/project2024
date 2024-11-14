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
# from hungarian_algorithm import algorithm as hungAlg

            
#%%
if __name__ =='__main__':
    ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))
    wLoadFolder, wLoadSubFolder = 'data4k', '1_test_png_for_segmenation_compare_outputs_sifted'
    wLoadDir= os.path.join(ROOT_DIR, wLoadFolder, wLoadSubFolder)
    wScale = 0.5
    wBnMetrics = BinaryMetricsManager(wLoadDir)
    wBnMetrics.loadData()

    # wIm = wBnMetrics.getImage()
    # show_wait(wIm, wScale)
    # wCentList = wBnMetrics.computeCentroidsOnIm(wIm)
    # wImClr = wBnMetrics.getClrImage()
    # wBnMetrics.drawCentroidsOnIm(wImClr, wCentList)
    # show_wait(wImClr, wScale)
    # show_wait(wBnMetrics.getImage(), wScale)
    # wLinkMatrix = linkage(np.array(wCentList))
    # wClusterList = fcluster(wLinkMatrix, 0.99)
    
    # wClrPerCluster = wBnMetrics.getColorPerCluster(wClusterList)
    # wKeys = list(wClrPerCluster.keys())

    # for wThresh in np.arange(475, 501, 25):
    #     wImClusters = wBnMetrics.drawClusters(wIm, iThresh=wThresh, iRadius=10)
    #     # print(wThresh)
    #     # show_wait(wImClusters,0.5)
    
    
#%%
    # wBnMetrics = BinaryMetricsManager(wLoadDir)
    # wBnMetrics.loadData()
    # for wData in wBnMetrics.getData()[-1:]:            
    #     wData.initiateMetrics()
    #     for wThresh in np.arange(400, 501, 50):
    #         wPredictions = wData.predict(iThresh=wThresh)
    #         wIm = wData.drawPredictions(iDrawPts=True)
    #         # show_wait(wIm, 0.5)
    
#%%
    
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
    # for wTruthName, wPredName in zip(wBnMetrics.getTruthNames(), wBnMetrics.getNames()):
    #     print(wTruthName,' ', wPredName)
    
    wHypotenuse = np.sqrt(W*W + H*H)
    wMaxDist = wHypotenuse
    wMinDist = 0 #wHypotenuse/8
    wStep = wHypotenuse/32
    for wThresh in np.arange(wMinDist, wMaxDist, wStep):
        wBnMetrics.computeMetrics(wThresh, iMinPercentArea=0.0)
        # show_wait(wBnMetrics.getDataAtIdx(0).drawPredictions(),0.5)
    wBnMetrics.evaluateMetrics(1./W, 1./H)
    wMetrics = wBnMetrics.getMetrics()
    print(wMetrics)
    print(len(wMetrics))
    
#%%    

    # wOriginalImList = generate_data_list(wDataPath, wPointsPath, wScaleX, wScaleY)
    
    # wOIm = wOriginalImList[0]
    
    # wTestOim =[]
    # for wOIm in wOriginalImList:
    #     wName = wOIm.get_name().split('.')[0] +'.'+ wData.getName().split('.')[1]
    #     if wName in wBnMetrics.getNames():
    #         wTestOim.append(wOIm)

    # for wOIm in wTestOim[-1:]:
    #     print(wOIm.get_center_dict())
    
    # wTruthDict = wOIm.get_center_dict()
    # wPredDict = wData.getClusterCentroids()
    # G = {}
    # wTruthKeyList = list(wTruthDict.keys())
    # wPredKeyList = list(wPredDict.keys())
    # wNoTruth = len(wTruthKeyList)
    # wNoPred = len(wPredKeyList)
    # wDelta = wNoTruth-wNoPred
    # if wDelta > 0:
    #     for i in range(wDelta):
    #         wKey = wPredKeyList[-1]
    #         wNewKey = wKey + str(i)
    #         wPredDict.update({wNewKey: wPredKeyList[wKey]})
    #         wPredKeyList.append(wNewKey)
    # elif wDelta < 0:
    #     for i in range(abs(wDelta)):
    #         wKey = wTruthKeyList[-1]
    #         wNewKey = wKey + str(i)
    #         wTruthDict.update({wNewKey: wTruthDict[wKey]})
    #         wTruthKeyList.append(wNewKey)
#%%        
    # for wKey in wTruthDict:
    #     wTruthCent = wTruthDict[wKey]
    #     G.update({wKey: {}})
    #     for wKey2 in wPredDict:
    #         wPredCent = wPredDict[wKey2]
    #         G[wKey].update({wKey2: dist(list(wTruthCent), list(wPredCent))})
           
    # wMatch = hungAlg.find_matching(G, 'min')
    # print(wMatch)
            
            


    
    

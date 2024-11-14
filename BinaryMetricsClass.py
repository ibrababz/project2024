# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:34:54 2024

@author: i_bab
"""

import cv2 as cv
import numpy as np
import os
import file
from helper import show_wait
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
# from hungarian_algorithm import algorithm as hungAlg
from dataLoad import generate_data_list
from math import dist
from copy import deepcopy
#%%

class BinaryMetricsManager:
    def __init__(self, iLoadDir):
        self.mLoadDir = iLoadDir
        self.setData([])
        
    def __repr__(self):
        return "Binary Metrics Manager"
        
    def setLoadDir(self, iLoadDir):
        self.mLoadDir = iLoadDir
        
    def getLoadDir(self):
        return self.mLoadDir
    
    def setData(self, iData):
        self.mData = iData
        self.setDataSize()
    
    def resetData(self):
        self.setData([])
    
    def getData(self):
        return self.mData
    
    def getNames(self):
        return [wData.getName() for wData in self.getData()]
    def setDataSize(self):
        self.mDataSize = len(self.mData)
    
    def getDataSize(self):
        return self.mDataSize
    
    def getFilesInDir(self):
        return os.listdir(self.getLoadDir())
    
    def getFilePath(self, iFile):
        return os.path.join(self.getLoadDir(), iFile)
    
    def loadData(self):
        if self.getDataSize() > 0:
            self.resetData()
        wData = []     
        for wFile in self.getFilesInDir():
            wImPath = self.getFilePath(wFile)
            wIm = cv.imread(wImPath, flags=cv.IMREAD_GRAYSCALE)
            wData.append(BinaryMetricsImage(wIm, wImPath))
        self.setData(wData)
    
    def getImage(self,iIdx=0):
        return self.getDataAtIdx(iIdx).getImage()
        
    
    def getDataAtIdx(self,iIdx=0):
        if iIdx <self.getDataSize():
            oData = self.getData()[iIdx]
        else:
            print('out of data bounds')
            oData = None
        return oData
    
    def getClrImage(self,iIdx=0, iCode=cv.COLOR_GRAY2BGR ):
        return self.getDataAtIdx(iIdx).getClr(iCode)
    
    def getShape(self, iIdx=0):
        return self.getDataAtIdx(iIdx).getShape()
    
    def findContours(self, iIm, iMode=cv.RETR_EXTERNAL, iMethod=2):
        return cv.findContours(iIm, iMode, iMethod)
    
    def findMoments(self, iCnt):
        return cv.moments(iCnt)
    def findArea(self, iCnt):
        return cv.contourArea(iCnt)
    
    def findCentroid(self, iCnt):
        wMoments =self.findMoments(iCnt)
        oCx = int(wMoments['m10']/(wMoments['m00']+1e-7))
        oCy = int(wMoments['m01']/(wMoments['m00']+1e-7))
        return oCx, oCy
        
    
    def computeCentroidsOnIm(self, iIm, iMinPercentArea = 0.003):
        wContourList, wHierarchy = self.findContours(iIm)
        wCentroidList = []
        for wCnt in wContourList:
            wArea = self.findArea(wCnt)
            H, W = iIm.shape
            wImArea = H*W
            if wArea/wImArea >= iMinPercentArea:
                wCx, wCy = self.findCentroid(wCnt)
                if not (wCx==0 and wCy ==0):
                    wCentroidList.append([wCx, wCy])
        return wCentroidList
        
    def drawCentroidsOnIm(self, iIm, iCentroidList, iRadius=20, iClr=(0, 0, 127), iThickness = -1):
        if len(iIm.shape) == 2:
            wClr = list(iClr)
            wClr = int(max(iClr))
        elif len(iIm.shape) == 3:
            wB, wG, wR = iClr
            wClr = int(wB), int(wG), int(wR)  
        for wCent in iCentroidList:
            cv.circle(iIm, tuple(wCent), radius=iRadius, color=wClr, thickness=iThickness)
    
    def computeCentroids(self, iMinPercentArea= 0.003):
        for wIm in self.getData():
            self.computeCentroidsOnIm(wIm, iMinPercentArea)
            # break
        
    def drawCentroids(self, iMinPercentArea= 0.003):
        for wIm in self.getData():
            wCentList = self.computeCentroidsOnIm(wIm, iMinPercentArea)
            self.drawCentroidsOnIm(wIm, wCentList)
            
    def getColorPerCluster(self, iClusterList):
        oClrPerCluster ={}
        wHue = np.uint8(0)
        for wCluster in iClusterList:
            if str(wCluster) not in oClrPerCluster.keys():
                wHSV = np.array([[[wHue,255,255]]], dtype=np.uint8)
                wClr = cv.cvtColor(wHSV, cv.COLOR_HSV2BGR)
                oClrPerCluster.update({str(wCluster):tuple(wClr[0,0])})
                wHue = np.uint8(wHue+15)
        return oClrPerCluster
    
    def getClusterList(self, iCentList, iThresh=0.99, iCriterion = 'distance', iDepth=2, iMethod='centroid', iMetric='euclidean', iOptOrder=True):
        wLinkMatrix = linkage(np.array(iCentList), method=iMethod, metric=iMetric, optimal_ordering=iOptOrder)
        # wLinkMatrix = ward(pdist(np.array(iCentList)))
        return fcluster(wLinkMatrix, iThresh, criterion=iCriterion, depth=iDepth)
        
    def getCentroidPerCluster(self, iCentList, iClusterList):
        wCentPerCluster = {}
        for wCluster, wCent in zip(iClusterList, iCentList):
            wKey = str(wCluster)
            wCent = tuple(wCent)
            if wKey in wCentPerCluster.keys():
                wCentPerCluster[wKey].append(wCent)
            else:
                wCentPerCluster.update({wKey:[wCent]})
        return wCentPerCluster
    
    def drawClusters(self, iIm, iThresh=0.99, iRadius=20, iThickness=-1):
        wCentList = self.computeCentroidsOnIm(iIm)
        wClusterList = self.getClusterList(wCentList, iThresh)
        wCentPerCluster = self.getCentroidPerCluster(wCentList, wClusterList)
        wClrPerCluster = self.getColorPerCluster(wClusterList)
        oImClr = cv.cvtColor(iIm, cv.COLOR_GRAY2BGR)
        for wKey in wCentPerCluster:
            wClusterCentList = wCentPerCluster[wKey]
            wClusterClr = wClrPerCluster[wKey]
            self.drawCentroidsOnIm(oImClr, wClusterCentList, iRadius=iRadius, iClr=wClusterClr , iThickness=iThickness)
        
        return oImClr
    
    def setTruthDir(self, iDir):
        self.mTruthDir = iDir

    def setTruthLabelsDir(self, iLabelsDir):
        self.mLabelsDir = iLabelsDir
        
    def setTruthLabelsFile(self, iLabelsFile):
        self.mLabelsFile = iLabelsFile
    
    def getTruthDir(self):
        return self.mTruthDir
    
    def getTruthLabelsDir(self):
        return self.mLabelsDir
    
    def getTruthLabelsFile(self):
        return self.mLabelsFile
    
    def setTruthScale(self, iScaleX, iScaleY):
        self.mScaleX, self.mScaleY = iScaleX, iScaleY
        
    def getTruthScale(self):
        return self.mScaleX, self.mScaleY
    
    def setTruthData(self, iTruthData):
        self.mTruthData = iTruthData
    
    def getTruthData(self):
        return self.mTruthData
    
    def loadTruthSet(self):
        wDataPath = self.getTruthDir()
        wLabelsPath = os.path.join(self.getTruthLabelsDir(), self.getTruthLabelsFile())
    
        wScaleX, wScaleY = self.getTruthScale()
        
        wOriginalImList = generate_data_list(wDataPath, wLabelsPath, wScaleX, wScaleY)
        wTestOim =[]
        wPairingDict = {}
        for wData in self.getData():
            for wOIm in wOriginalImList:
                wTruthName = wOIm.get_name()
                wPredName = wData.getName()
                if wPredName.split('.')[0] == wTruthName.split('.')[0]:
                    wKey = wPredName.split('.')[0]
                    # print(wPredName, wTruthName)
                    wPairingDict.update({wKey:{'truth':{'data':wOIm, 'values':wOIm.get_center_dict()}, 'pred': {'data':wData}}})
                    wTestOim.append(wOIm)

        self.setTruthData(wTestOim)
        self.setTruthPredPair(wPairingDict)
        
        
    def getTruthDataAtIdx(self, iIdx):
        return self.getTruthData()[iIdx]
    
    def getTruthNameAtIdx(self, iIdx):
        return self.getTruthData()[iIdx].get_name()
    
    def getTruthRegionsAtIdx(self, iIdx):
        return self.getTruthDataAtIdx(iIdx).get_regions()
    
    def getTruthCentroidsAtIdx(self, iIdx):
        return self.getTruthDataAtIdx(iIdx).get_center_dict()
    
    def getTruthNames(self):
        return [wData.get_name() for wData in self.getTruthData()]
    
    def setTruthPredPair(self, iPairingDict):
        self.mPairingDict = iPairingDict
    
    def getTruthPredPair(self):
        return self.mPairingDict

    def computeMetricsOnSample(self, iKey, iThresh, iMinPercentArea = 0.003):
        wTruthPredPair = self.getTruthPredPair()
        wPredDict = wTruthPredPair[iKey]['pred'] 
        wPredDict['data'].initiateMetrics(iMinPercentArea)
        # wPredDict.update({str(iThresh): {'values':wPredDict['data'].predict(iThresh)}})
        wPredDict.update({str(iThresh): {'values':wPredDict['data'].predict(iThresh), 
                                         'pts': wPredDict['data'].getCentroidsPerCluster(), 
                                         'colors':  wPredDict['data'].getColorPerCluster()}})

                
    def computeMetrics(self, iThresh, iMinPercentArea = 0.003):
        for wKey in self.getTruthPredPair():
            self.computeMetricsOnSample(wKey, iThresh, iMinPercentArea)
            
    def evaluateMetricsOnSample(self, iKey, iNorm=True):
        wTruthPredPair = self.getTruthPredPair()
        # print(wTruthPredPair[iKey])
        wTruth = wTruthPredPair[iKey]['truth']['values']
        
        wPredDicts =  wTruthPredPair[iKey]['pred']
        for wKey in wPredDicts:
            if wKey != 'data':
                wPred = wPredDicts[wKey]['values']
                wAdjTruth, wAdjPred, wNoUncount = self.adjustLengths(wTruth, wPred)   
                wMatchCost = self.match(wAdjTruth, wAdjPred, iNorm)
                wMatchPredictedOnly = self.matchPredictedOnly(wAdjTruth, wAdjPred, iNorm)
                # print(wMatchCost)
                wPredDicts[wKey].update({'cost': wMatchCost})
                wPredDicts[wKey].update({'cost_pred': wMatchPredictedOnly})
                wPredDicts[wKey].update({'#missed': wNoUncount})
                
                
                
                
    def evaluateMetrics(self, iNorm=True):
        for wKey in self.getTruthPredPair():
            # print(wKey)
            self.evaluateMetricsOnSample(wKey, iNorm)


    def adjustLengths(self, iTruthValuesDict, iPredValuesDict):
        wTruthValuesDict = iTruthValuesDict.copy()
        wPredValuesDict = iPredValuesDict.copy()
        wTruthKeyList = list(wTruthValuesDict.keys())
        wPredKeyList = list(wPredValuesDict.keys())
        wNoTruth = len(wTruthKeyList)
        wNoPred = len(wPredKeyList)
        wDelta = wNoTruth-wNoPred
        H,W = self.getShape()
        if wDelta > 0:
            wIdx = 0            
            # wIdx = np.random.randint(wNoPred)
            for i in range(wDelta):
                wKey = wPredKeyList[wIdx]
                wNewKey = wKey+'_' + str(i)
                wPredValuesDict.update({wNewKey: (2*W, 2*H)})
                # wPredKeyList.append(wNewKey)
        elif wDelta < 0:
            wIdx=0
            # wIdx = np.random.randint(wNoTruth)
            for i in range(abs(wDelta)):
                wKey = wTruthKeyList[wIdx]
                wNewKey = wKey + '_' +str(i)
                wTruthValuesDict.update({wNewKey: (2*W,2*H)})
                # wTruthKeyList.append(wNewKey)
                
        return wTruthValuesDict, wPredValuesDict, wDelta
    
    def match(self, iTruthValuesDict, iPredValuesDict, iNorm=True):
        wBipartMat = {}
        
        H, W = self.getShape()
        
        wScaleX, wScaleY = 1., 1.
        wHypotenuse = np.sqrt(H*H + W*W)
            
        for wKey in iTruthValuesDict:
            wTruthCent = list(deepcopy(iTruthValuesDict[wKey]))
            wBipartMat.update({wKey: {}})
            wTruthX, wTruthY = wTruthCent
            wTruthX, wTruthY = wScaleX*wTruthX, wScaleY*wTruthY
            wScaledTruth = [wTruthX, wTruthY] 
            for wKey2 in iPredValuesDict:
                wPredCent = list(deepcopy(iPredValuesDict[wKey2]))
                wPredX, wPredY = wPredCent
                wPredX, wPredY = wScaleX*wPredX, wScaleY*wPredY
                wScaledPred = [wPredX, wPredY]
                wDist =  dist(wScaledTruth, wScaledPred)
                if wDist > wHypotenuse:
                    wDist = np.clip(wDist, 0 , wHypotenuse)
                
                if iNorm:
                    wDist = wDist/wHypotenuse

                wBipartMat[wKey].update({wKey2: wDist})
        # print(wBipartMat)       
        wBipartMatArray = self.bipartMatDictToArray(wBipartMat)
        wRowIx, wColIdx = linear_sum_assignment(wBipartMatArray)
        oMatchCost = wBipartMatArray[wRowIx, wColIdx].sum()
        return oMatchCost
    
    def matchPredictedOnly(self, iTruthValuesDict, iPredValuesDict, iNorm=True):
        wBipartMat = {}
        H, W = self.getShape()
        
        wScaleX, wScaleY = 1., 1.
        wHypotenuse = np.sqrt(H*H + W*W)
        
        for wKey in iTruthValuesDict:
            wTruthCent = list(deepcopy(iTruthValuesDict[wKey]))
            wBipartMat.update({wKey: {}})
            wTruthX, wTruthY = wTruthCent
            wTruthX, wTruthY = wScaleX*wTruthX, wScaleY*wTruthY
            wScaledTruth = [wTruthX, wTruthY] 
            wUnpredictedCounter = 0
            for wKey2 in iPredValuesDict:
                wPredCent = list(deepcopy(iPredValuesDict[wKey2]))
                wPredX, wPredY = wPredCent
                wPredX, wPredY = wScaleX*wPredX, wScaleY*wPredY
                wScaledPred = [wPredX, wPredY]
                wDist =  dist(wScaledTruth, wScaledPred)
                if wDist > wHypotenuse: #this happens when a sample is not predicted
                    wDist = 0
                    wUnpredictedCounter+=1    
                if iNorm:
                    wDist = wDist/wHypotenuse
                # print("wDist: ", wDist)
                wBipartMat[wKey].update({wKey2: wDist})
        # print(wBipartMat)       
        wBipartMatArray = self.bipartMatDictToArray(wBipartMat)
        wRowIx, wColIdx = linear_sum_assignment(wBipartMatArray)
        oMatchCost = wBipartMatArray[wRowIx, wColIdx].sum()
        return oMatchCost
    
    def bipartMatDictToArray(self, iBipartMatDict):
        oBipartMatArray =[]
        for wKey in iBipartMatDict:
            wRow = []
            for wKey2 in iBipartMatDict[wKey]:
                wRow.append(iBipartMatDict[wKey][wKey2])
            oBipartMatArray.append(wRow)
        return np.array(oBipartMatArray)
        
    def getMetricsLists(self, iPredictedOnly=False):
        oMetricsLists = {}
        oNoUnpredicted = {}
        wTruthPredPair = self.getTruthPredPair()
        for wKey in wTruthPredPair:
            wPredDict = wTruthPredPair[wKey]['pred']
            for wKey2 in wPredDict:
                if wKey2 !='data':
                    if wKey2 not in oMetricsLists.keys():
                        oMetricsLists.update({wKey2: []})   
                    if iPredictedOnly:
                        if wKey2 not in oNoUnpredicted.keys(): #update keys in unpredicted only if required
                            oNoUnpredicted.update({wKey2: []})                        
                        oMetricsLists[wKey2].append(wPredDict[wKey2]['cost_pred'])  
                        oNoUnpredicted[wKey2].append(wPredDict[wKey2]['#missed'])  
                    else:
                        oMetricsLists[wKey2].append(wPredDict[wKey2]['cost'])

                    
        return oMetricsLists, oNoUnpredicted
    
    def getMetrics(self, iPredictedOnly=False, iNorm=True):
        oMetrics = {}
        oNoUnpredicted ={}
        wMetricsLists, wNoUnpredicted = self.getMetricsLists(iPredictedOnly)
        if iNorm:
            H,W = self.getShape()     
            wNorm = np.sqrt(H*H + W*W)
        else:
            wNorm = 1
        for wKey in wMetricsLists:
            wNrmKey = str(np.round(float(wKey)/wNorm,3))
            oMetrics.update({wNrmKey: np.mean(wMetricsLists[wKey])})
            
        for wKey in wNoUnpredicted:
            wNrmKey = str(np.round(float(wKey)/wNorm,3))
            oNoUnpredicted.update({wNrmKey: np.mean(wNoUnpredicted[wKey])})
            
        return oMetrics, oNoUnpredicted
    
    def getMinIdx(self, iPredictedOnly=False, iNorm = False):
        wMetricsAsList, wNoUnpredictAsList, wThresholds = self.getMetricsAsList(iPredictedOnly, iNorm)
        return np.argmin(wMetricsAsList)
        
    def getMinCost(self, iPredictedOnly=False, iNorm = False):
        wMetricsAsList, wNoUnpredictAsList, wThresholds = self.getMetricsAsList(iPredictedOnly, iNorm)
        wIdx = self.getMinIdx(iPredictedOnly, iNorm)
        return wMetricsAsList[wIdx], wThresholds[wIdx]
    
    def getMetricsAsList(self, iPredictedOnly=False, iNorm=True):
        wMetrics, wNoUnpredict = self.getMetrics(iPredictedOnly, iNorm)
        oThresholds = self.getClusterThresholds(iNorm)
        
        return [wMetrics[wMtrc] for wMtrc in wMetrics], [wNoUnpredict[wNo] for wNo in wNoUnpredict], oThresholds 
    
    def getMetricsPredicted(self, iNorm=True):
        return self.getMetrics(iPredictedOnly=True, iNorm=iNorm)
    
    def getMetricsPredictedAsList(self, iNorm=True):
        return self.getMetricsAsList(iPredictedOnly=True, iNorm=iNorm)

    
    def drawPredictionAtIdxAtThresh(self, iIdx, iThresh):
        wTruthPredPair = self.getTruthPredPair()
        wKey = list(wTruthPredPair.keys())[iIdx]
        wPredDictAtIdxAtThresh = wTruthPredPair[wKey]['pred'][str(iThresh)]
        wClusterCentroids = wPredDictAtIdxAtThresh['values']
        wCentersPerCluster = wPredDictAtIdxAtThresh['pts']
        wClrsPerCluster = wPredDictAtIdxAtThresh['colors']
        wData = wTruthPredPair[wKey]['pred']['data']
        oIm = wData.drawPredictions(iThresh, wClusterCentroids,wCentersPerCluster, wClrsPerCluster)
        H, W = wData.getShape()
        wFirstH = int(H-10*H/30)
        
        oIm, wTextW, wTextH = self.textOnPredIm(oIm, wPredDictAtIdxAtThresh, iValKey='cost', iPos=((W)//35 ,wFirstH))
        wPadding = wTextH
        oIm, wTextW, wTextH = self.textOnPredIm(oIm, wPredDictAtIdxAtThresh, iValKey='cost_pred', iPos=((W)//35, wFirstH + wTextH + wPadding))
        oIm, _, _ = self.textOnPredIm(oIm, wPredDictAtIdxAtThresh, iValKey='#missed', iPos=((W)//35, wFirstH + 2*(wTextH + wPadding)))

        return oIm
    
    def textOnPredIm(self, ioIm, iDict, iValKey, iPos, iFont=cv.FONT_HERSHEY_SIMPLEX, iFontScale=3.5, iTextColor=(0, 0, 0), iThick=10):
        if iValKey in list(iDict.keys()):
            H, W = ioIm.shape[:2]
            wVal = iDict[iValKey]
            wText = iValKey+ ': '+str(np.round(wVal, 2))
            wX, wY = iPos
            wTextSize, _ = cv.getTextSize(wText, iFont, iFontScale, iThick)
            wTextW, wTextH = wTextSize
            
            
            wShape = cv.rectangle(np.zeros_like(ioIm, np.uint8), (wX-10, wY-15), (wX + wTextW+10, wY + wTextH+15), (175, 175, 175), -1)
            wMask = wShape.astype(bool)
            wAlpha = 0.3
            ioIm[wMask] = cv.addWeighted(ioIm, wAlpha, wShape, 1 - wAlpha, 0)[wMask]
            ioIm = cv.putText(ioIm, wText ,(wX, wY+ wTextH + int(iFontScale) - 1) ,iFont , iFontScale, iTextColor, iThick, lineType=cv.LINE_AA)
        return ioIm, wTextW, wTextH
    
    def setSaveDir(self, iSaveDir):
        self.mSaveDir = iSaveDir
        
    def getSaveDir(self):
        return self.mSaveDir
    
    def saveImagesForThresh(self, iThresh, iFolderPrefix, iExt = '.png'):
        wSaveDir = self.getSaveDir()
        wTruthPredPair = self.getTruthPredPair()
        wKeyList = list(wTruthPredPair.keys())
        for wIdx, wKey in zip(range(len(wKeyList)), wKeyList):
            wIm = self.drawPredictionAtIdxAtThresh(wIdx, iThresh)
            wName = wKey+iExt
            H, W = self.getShape(0)
            wThreshNrm = np.round(iThresh/(np.sqrt(H*H + W*W)),3)
            wSavePath = os.path.join(wSaveDir, iFolderPrefix +str(wThreshNrm).replace('.','p'))
            os.makedirs(wSavePath, exist_ok=True)
            # print(wSavePath)
            cv.imwrite(os.path.join(wSavePath, wName), wIm)
            
    
    def getClusterThresholds(self, iNorm=False):
        wTruthPredPair = self.getTruthPredPair()
        wKey = list(wTruthPredPair.keys())[0]
        wPredDict = wTruthPredPair[wKey]['pred']
        oThreshList = []
        
        if iNorm:
            H,W =self.getShape()
            wNrm = np.sqrt(H*H + W*W)
        else:
            wNrm = 1.
            
        for wKey2 in wPredDict:
            if wKey2 != 'data':
                wThresh = float(wKey2)/wNrm
                oThreshList.append(np.round(wThresh,3))
        return oThreshList
    
    def printMetrics(self):
        wMetrics, _ = self.getMetrics()
        wMetricsPredicted, wNoUnpredict= self.getMetricsPredicted()

        for wKey in wMetrics:
            wMtrc = wMetrics[wKey]
            wMtrcPred = wMetricsPredicted[wKey]
            wNo = wNoUnpredict[wKey]
            print("Tresh: %s, cost: %s, cost_predicted: %s, unpredicted: %s"%(wKey, wMtrc, wMtrcPred, wNo))
    
    def drawPredIdxForThreshRange(self, iIdx, iThreshList, iPad=0.01, iPadLast=False):
        wImList = []
        wLen = len(iThreshList)
        for i, wThresh in zip(range(wLen), iThreshList):
            wIm = self.drawPredictionAtIdxAtThresh(iIdx, wThresh)
            wImList.append(wIm)
            if iPad:
                wShape = list(wIm.shape)
                wShape[1] = int(iPad*wShape[1])
                wPad = np.ones(tuple(wShape), dtype=np.uint8)*255
                if iPadLast:
                    wImList.append(wPad)
                elif i < wLen-1:
                    wImList.append(wPad)
                
        return np.hstack(wImList)
    
    def drawPredRangeForThreshRange(self, iIdxList, iThreshList, iPad=0.01, iPadLast=False):
        wRowList = []
        wLen = len(iIdxList)
        for i, wIdx in zip(range(wLen), iIdxList):
            wIm = self.drawPredIdxForThreshRange(wIdx, iThreshList, iPad, iPadLast)
            wRowList.append(wIm)
            if iPad:
                wShape = list(wIm.shape)
                wShape[0] = int(iPad*wShape[0])
                wPad = np.ones(tuple(wShape), dtype=np.uint8)*255
                if i < wLen-1:
                    wRowList.append(wPad)
        return np.vstack(wRowList)
    
    def drawRawImage(self, iIdx, iPad=0.01, iPadLast=False):
        wImList = []
        wIm = self.getClrImage(iIdx)
        wImList.append(wIm)
        if iPad and iPadLast:
            wShape = list(wIm.shape)
            wShape[1] = int(iPad*wShape[1])
            wPad = np.ones(tuple(wShape), dtype=np.uint8)*255
            wImList.append(wPad)      
        return np.hstack(wImList)
    
    def drawRawImageRange(self, iIdxList, iPad=0.01, iPadLast=False):
        wRowList = []
        wLen = len(iIdxList)
        for i, wIdx in zip(range(wLen), iIdxList):
            wIm = self.drawRawImage(wIdx, iPad, iPadLast)
            wRowList.append(wIm)
            if iPad:
                wShape = list(wIm.shape)
                wShape[0] = int(iPad*wShape[0])
                wPad = np.ones(tuple(wShape), dtype=np.uint8)*255
                if i < wLen-1:
                    wRowList.append(wPad)
        return np.vstack(wRowList)
    
                
#%%            
class BinaryMetricsImage:
    def __init__(self, iImage, iFilePath):
        self.mImage = iImage
        self.mFilePath = iFilePath
        self.mLoadDir = os.path.dirname(iFilePath)
        self.mName = iFilePath.split('\\')[-1]
        self.setInitFlag(False)
 
    def getLoadDir(self):
        return self.mLoadDir
    
    def getPath(self):
        return self.mFilePath
    
    def getName(self):
        return self.mName
        
    def getImage(self):
        return self.mImage.copy()
    
    def getShape(self):
        return self.mImage.shape
    
    def getClr(self, iCode=cv.COLOR_GRAY2BGR):
        return cv.cvtColor(self.getImage(), code=iCode)
    
    def setContours(self, iMode=cv.RETR_EXTERNAL, iMethod=2):
        self.mContours, self.mHierarchy = cv.findContours(self.getImage(), iMode, iMethod)
    
    def getContours(self):
        return self.mContours
    
    def setMomentsList(self):
        self.mMomentsList = []
        for wCnt in self.getContours():
            self.mMomentsList.append(cv.moments(wCnt))
            
    def getMomentsList(self):
        return self.mMomentsList
    
    def getMoments(self, iIdx):
        return self.getMomentsList[iIdx]
     
    def setAreaList(self):
        self.mAreaList = []
        for wCnt in self.getContours():
            self.mAreaList.append(cv.contourArea(wCnt))
    
    def getAreaList(self):
        return self.mAreaList
   
    def getArea(self,iIdx):
        return self.getAreaList[iIdx]
            
    
    def findCentroid(self, iCnt):
        wMoments = cv.moments(iCnt)
        oCx = int(wMoments['m10']/(wMoments['m00']+1e-7))
        oCy = int(wMoments['m01']/(wMoments['m00']+1e-7))
        return oCx, oCy
    
    def setCentroidList(self, iMinPercentArea = 0.003):
        wCentroidList = []
        for wCnt, wArea in zip(self.getContours(), self.getAreaList()):
            H, W = self.getShape()
            wImArea = H*W
            if wArea/wImArea >= iMinPercentArea:
                wCx, wCy = self.findCentroid(wCnt)
                if not (wCx==0 and wCy ==0):
                    wCentroidList.append([wCx, wCy])
        if len(wCentroidList) ==0:
            H,W = self.getShape()
            wCentroidList.append([2*W, 2*H])
        self.mCentroidList = wCentroidList
                
    def getCentroidList(self):
        return self.mCentroidList
    
    def drawCentroidsOnIm(self, iIm, iRadius=20, iClr=(0, 0, 127), iThickness = -1):
        if len(iIm.shape) == 2:
            wClr = list(iClr)
            wClr = int(max(iClr))
        elif len(iIm.shape) == 3:
            wB, wG, wR = iClr
            wClr = int(wB), int(wG), int(wR)  
        for wCent in self.getCentroidList():
            cv.circle(iIm, tuple(wCent), radius=iRadius, color=wClr, thickness=iThickness)
            
    def setLinkageMatrix(self,iMethod='centroid', iMetric='euclidean', iOptOrder=True):
        if len(self.getCentroidList()) > 1:
            self.mLinkMatrix = linkage(np.array(self.getCentroidList()), method=iMethod, metric=iMetric, optimal_ordering=iOptOrder)
        else:
            self.mLinkMatrix = None# wLinkMatrix = ward(pdist(np.array(iCentList)))
            
    def getLinkageMatrix(self):
        return self.mLinkMatrix
    
    def setClusterList(self, iThresh, iCriterion = 'distance', iDepth=2):
        if self.getLinkageMatrix() is not None:
            self.mClusterList = fcluster(self.getLinkageMatrix(), iThresh, criterion=iCriterion, depth=iDepth)
        else:
            self.mClusterList = np.array([1])
        
    def setClusterListToCentroidList(self):
        pass
        
    def getClusterList(self):
        return self.mClusterList
    
    def setColorPerCluster(self):
        wClrPerCluster ={}
        wHue = 0
        for wCluster in self.getClusterList():
            if str(wCluster) not in wClrPerCluster.keys():
                wHSV = np.array([[[wHue,255,255]]], dtype=np.uint8)
                wClr = tuple(cv.cvtColor(wHSV, cv.COLOR_HSV2BGR)[0,0])
                wB, wG, wR = wClr
                wClr = int(wB), int(wG), int(wR)
                wClrPerCluster.update({str(wCluster):wClr})
                wHue = np.uint8(wHue+12)
        self.mClrPerCluster = wClrPerCluster
        
    def getColorPerCluster(self):
        return self.mClrPerCluster
    
    def setCentroidPerCluster(self):
        wCentPerCluster = {}
        for wCluster, wCent in zip(self.getClusterList(), self.getCentroidList()):
            wKey = str(wCluster)
            wCent = tuple(wCent)
            if wKey in wCentPerCluster.keys():
                wCentPerCluster[wKey].append(wCent)
            else:
                wCentPerCluster.update({wKey:[wCent]})
        self.mCentsPerCluster= wCentPerCluster   
    
    def getCentroidsPerCluster(self):
        return self.mCentsPerCluster
    
    def drawClustersOnIm(self, iCentPerCluster= None, iClrPerCluster= None,  iRadius=20, iThickness=-1):
        if iCentPerCluster is None:
            wCentPerCluster = self.getCentroidsPerCluster()
        else:
            wCentPerCluster= iCentPerCluster
        if iClrPerCluster is None:    
            wClrPerCluster = self.getColorPerCluster()
        else:
            wClrPerCluster = iClrPerCluster
            
        oImClr = self.getClr()
        
        for wKey in wCentPerCluster:
            wClusterCentList = wCentPerCluster[wKey]
            wClusterClr =wClrPerCluster[wKey]        
            for wCent in wClusterCentList:
                cv.circle(oImClr, wCent, radius=iRadius, color=wClusterClr , thickness=iThickness)      
        return oImClr
    
    def setClusterCentroids(self):
        wClusterCentroids = {}
        wCentsPerCluster = self.getCentroidsPerCluster()
        for wKey in wCentsPerCluster:
            wCentList = wCentsPerCluster[wKey]
            wCx, wCy = np.mean(wCentList, axis=0)                
            wClusterCentroids.update({wKey:(int(wCx), int(wCy))})
        self.mClusterCentroids = wClusterCentroids
        
    def getClusterCentroids(self):
        return self.mClusterCentroids
    
    def drawClusterCentroids(self, iClusterCentroids=None, iCentPerCluster=None, iClrPerCluster=None, iRadius=50, iThickness=15, iDrawPts=True):
        
        if iClusterCentroids is None:
            wClusterCentroids = self.getClusterCentroids()
        else:
            wClusterCentroids= iClusterCentroids
            
        if iCentPerCluster is None:
            wCentPerCluster = self.getCentroidsPerCluster()
        else:
            wCentPerCluster= iCentPerCluster
        
        if iClrPerCluster is None:    
            wClrPerCluster = self.getColorPerCluster()
        else:
            wClrPerCluster = iClrPerCluster
            
        if iDrawPts:
            oImClr = self.drawClustersOnIm(wCentPerCluster,wClrPerCluster)
        else:
            oImClr = self.getClr()
        
        for wKey in wClusterCentroids:
            wCent = wClusterCentroids[wKey]
            wClusterClr = wClrPerCluster[wKey]
            cv.circle(oImClr, wCent, radius=iRadius, color=wClusterClr , thickness=iThickness)         
        return oImClr
    
    def setInitFlag(self, iBool=True):
        self.mInitFlag=iBool
    
    def getInitFlag(self):
        return self.mInitFlag
    
    def initiateMetrics(self, iMinPercentArea=0.003, iMethod='centroid', iMetric='euclidean', iOptOrder=True):
        self.setContours()
        self.setAreaList()
        self.setMomentsList()
        self.setCentroidList(iMinPercentArea)
        self.setLinkageMatrix(iMethod, iMetric, iOptOrder)
        self.setInitFlag(True)
        
    def predict(self, iThresh, iMinPercentArea=0.003):
        if not self.getInitFlag():
            print("initiatiing metrics with default values")
            self.initiateMetrics(iMinPercentArea=iMinPercentArea)
        # print(iThresh)
        self.setClusterList(iThresh)
        self.setColorPerCluster()
        self.setCentroidPerCluster()
        # wIm = self.drawClustersOnIm()
        self.setClusterCentroids()
        # wIm = self.drawClusterCentroids()
        return self.getClusterCentroids()
    
    def drawPredictions(self, iThresh, iClusterCentroids=None, iCentersPerCluster=None, iColorsPerCluster=None, iRadius=50, iThickness=15, iDrawPts=True):
        H, W = self.getShape()
        wThresh = iThresh/(np.sqrt(H*H +W*W))
        wText = 'Dist.: '+str(np.round(wThresh, 2))
        wFont = cv.FONT_HERSHEY_SIMPLEX
        wFontScale = 3.5
        wTextColor = (0, 0, 0)
        wThick = 10
        wPos = ((W)//35 ,H//35)
        wX, wY = wPos
        wTextSize, _ = cv.getTextSize(wText, wFont, wFontScale, wThick)
        wTextW, wTextH = wTextSize
        wIm = self.drawClusterCentroids(iClusterCentroids, iCentersPerCluster, iColorsPerCluster, iRadius, iThickness, iDrawPts)        
        
        
        wAlpha = 0.3
        wShape = cv.rectangle(np.zeros_like(wIm, np.uint8), (wX-10, wY-15), (wX + wTextW+10, wY + wTextH+15), (175, 175, 175), -1)
        wMask = wShape.astype(bool)
        wIm[wMask] = cv.addWeighted(wIm, wAlpha, wShape, 1 - wAlpha, 0)[wMask]
        
        return cv.putText(wIm, wText ,(wX, wY+ wTextH + int(wFontScale) - 1) ,wFont , wFontScale, wTextColor, wThick)
    
    def getNumberOfClusters(self):
        return len(list(self.getClusterCentroids().keys()))
    
    def __repr__(self):
        return "Binary Metrics Image"
    
            
#%%
if __name__ =='__main__':
    ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))
    wLoadFolder, wLoadSubFolder = 'data4k', '1_test_png_for_segmenation_compare_outputs'
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
    wMinDist = wHypotenuse/8
    wStep = wHypotenuse/32
    for wThresh in np.arange(wMinDist, wMaxDist+wStep, 50):
        wBnMetrics.computeMetrics(wThresh)
        show_wait(wBnMetrics.getDataAtIdx(0).drawPredictions(),0.5)
    wBnMetrics.evaluateMetrics(1., 1.)
    print(wBnMetrics.getMetrics())
    
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
            
            


    
    

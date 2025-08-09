#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 13:11:02 2025

@author: ibabi
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def loadHistory(iLoadPath, iKeyList=None):
    
    with open(iLoadPath) as wFile:
        wHeader = wFile.readline().lower().replace('\n', '').split(',')
        wContents = [[float(wVal) for wVal in wLine.replace('\n', '').split(',')] for wLine in wFile]
    wContentsArray = np.array(wContents).T
    oDict ={wKey: wContentsArray[i] for i, wKey in enumerate(wHeader)}
    if iKeyList is not None:
        wNewKeyList = ['epoch'] + iKeyList
        wTemp = {wKey: oDict[wKey] for wKey in wNewKeyList}
        oDict = wTemp
    
    if 'epoch' in oDict.keys():
        oDict['epoch'] = np.int32(oDict['epoch'])
    

    return oDict
        
if __name__ =='__main__':
    wFolderName = 'testing_6'
    wDictList = []
    for wFileName in os.listdir(os.path.join('.', wFolderName)):
        if 'history' in wFileName and '.csv' in wFileName:
            wDict = {'name': wFileName}
            wDict.update(loadHistory(os.path.join('.', wFolderName, wFileName)))
            wDictList.append(wDict)
            
    wFigure = plt.figure()
    wPlotKey = 'loss'
    for wDict in wDictList:
        wName = '_'.join([wDict['name'].split('.')[0], wPlotKey])
        plt.plot(wDict['epoch'], wDict[wPlotKey], label=wName)
    plt.legend()
    plt.savefig(os.path.join('.', wFolderName, 'history_comparison.png'))
    plt.show()
    plt.close()
        
            
            
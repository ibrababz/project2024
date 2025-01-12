# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:57:49 2025

@author: User
"""
import os

def decodeParserSched(iList, iEpKey=int, iValKey= int):
    oEpList, oValList = [], []
    for i in range(len(iList)//2):
        oEpList.append(iEpKey(iList[2*i]))
        oValList.append(iValKey(iList[2*i+1]))
    return oEpList, oValList

def logArgs(iArgs, iSaveDir, iFileName='trainNewArgs.csv', iAttrMarker ='m'):
    wFilePath = os.path.join(iSaveDir, iFileName)
    with open(wFilePath, 'w') as wFile:
        for wAttr in dir(iArgs):
            if wAttr[0] ==iAttrMarker:
                wFile.write(wAttr)
                wFile.write(',')
                wVal = eval(".".join(['iArgs', wAttr]))
                if type(wVal) in [int, float, str]:
                    wFile.write(str(wVal))
                elif type(wVal) == list:
                    wLine = ','.join([str(wEle) for wEle in wVal])
                    wFile.write(wLine)
                elif wVal is None:
                    wFile.write('None')
                wFile.write('\n')

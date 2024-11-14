# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:25:43 2024

@author: i_bab
"""
import random
from copy import deepcopy


class RandSampler():
    def __init__(self, iList, iMax, iSeed = None):
        self.mList = iList
        self.mMax = iMax
        self.mModList = deepcopy(self.mList)
        self.lenModList = len(self.mModList)
        self.mSeed = iSeed
        self.random = random
        self.mCounter = 0
       
        if not self.lenModList:
            raise Exception('Input List has zero length')
        
    def __next__(self):
        if self.lenModList>0:
            wSeed = self.mCounter
            if self.mSeed is not None:
                wSeed += self.mSeed
            self.seed(wSeed)
            wIndices = list(range(len(self.mModList)))
            if self.lenModList == 1:
                nSample = 1
            else:
                nSample = self.random.randint(1,self.mMax)
                while nSample > self.lenModList:
                    nSample = self.random.randint(1,self.mMax)
            wSampled = self.random.sample(wIndices, nSample)
            #print(wSampled)
            wSampled_values = [self.mModList[i] for i in wSampled]
            for i in sorted(wSampled, reverse = True):
                self.mModList.pop(i)
            wIndices = list(range(len(self.mModList)))
            self.lenModList = len(self.mModList)
            #print(len(wSampled_values))
            self.mCounter +=1
            return wSampled_values
        else:
            self.resample()
            return self.__next__()
        
    def resample(self):
        # print('resampling')
        self.mModList = deepcopy(self.mList)
        self.lenModList = len(self.mModList)
        return self.lenModList
        
    def __iter__(self):
        return self
    
    def seed(self, iSeed):
        self.random.seed(iSeed)
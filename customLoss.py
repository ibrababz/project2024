#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:14:07 2025

@author: ibabi
"""
import os
import tensorflow as tf
import keras

from customRotLayer import RandomRotationHMap
from models import makeYoloType
import timeit
import matplotlib.pyplot as plt

from tfMapUtils import tfFlatMapBatch, tfAct3DBatch, tfFlatMap, tfAct3D
from tfMapUtils import tfFlatMapBatchEager, tfAct3DBatchEager
from tfMapUtils import getVectors, getVectorsEager




@tf.function
def tensorPosNegLoss(vect_pred, vect_map1, vect_act1):
    vect_map1_inv = 1 - vect_map1
    vect_act1_inv = 1 - vect_act1
    neg_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1_inv*vect_map1_inv*tf.math.log(vect_act1_inv*(1-vect_pred) + 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1_inv)
    pos_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1*tf.math.log(vect_act1*vect_pred+ 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1)
    return pos_loss, neg_loss

@keras.saving.register_keras_serializable()
@tf.function
def tensorPosNegLossV2(vect_map1, vect_act1, vect_pred):
    vect_map1_inv = 1. - vect_map1
    vect_act1_inv = 1. - vect_act1
    neg_loss = tf.math.divide_no_nan(tf.reduce_sum(tf.clip_by_value((-vect_act1_inv*vect_map1_inv*tf.math.log(vect_act1_inv*(1-vect_pred) + 1e-15)), 0.0, 15.), keepdims=True),tf.reduce_sum(vect_act1_inv))
    pos_loss = tf.math.divide_no_nan(tf.reduce_sum(tf.clip_by_value((-vect_act1*tf.math.log(vect_act1*vect_pred+ 1e-15)), 0.0, 15.), keepdims=True),tf.reduce_sum(vect_act1))
    batch_size = tf.cast(tf.shape(vect_map1)[0], dtype=tf.float32)
    return (pos_loss + 5.*neg_loss)/ batch_size
    


@tf.function
def tensorDiceLoss(vect_pred, vect_map1, vect_act1):
    vect_act1_inv = 1 - vect_act1
    dice_pos = 1 - 2.*tf.reduce_sum(vect_act1*vect_pred)/(tf.reduce_sum(vect_act1*vect_act1) + tf.reduce_sum(vect_pred*vect_pred)+1.e-15)
    dice_neg = 1 - 2.*tf.reduce_sum(vect_act1_inv*(1.-vect_pred))/(tf.reduce_sum(vect_act1_inv* vect_act1_inv) + tf.reduce_sum((1.-vect_pred)*(1.-vect_pred))+1.e-15)
    dice_loss = (dice_pos+dice_neg)/2
    return dice_loss

@tf.function
def tensorDiceLossV2(vect_act1, vect_pred):
    vect_act1_inv = 1 - vect_act1
    dice_pos = 1. - 2.*tf.math.divide_no_nan(tf.reduce_sum(vect_act1*vect_pred, keepdims=True),(tf.reduce_sum(vect_act1*vect_act1, keepdims=True) + tf.reduce_sum(vect_pred*vect_pred, keepdims=True)+1.e-15))
    dice_neg = 1. - 2.*tf.math.divide_no_nan(tf.reduce_sum(vect_act1_inv*(1.-vect_pred), keepdims=True), (tf.reduce_sum(vect_act1_inv* vect_act1_inv, keepdims=True) + tf.reduce_sum((1.-vect_pred)*(1.-vect_pred), keepdims=True)+1.e-15))
    dice_loss = (dice_pos+dice_neg)/2.
    batch_size = tf.cast(tf.shape(vect_act1)[0], dtype=tf.float32)
    return dice_loss/batch_size
    
def tensorPosNegLossEager(vect_pred, vect_map1, vect_act1):
    vect_map1_inv = 1 - vect_map1
    vect_act1_inv = 1 - vect_act1
    neg_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1_inv*vect_map1_inv*tf.math.log(vect_act1_inv*(1-vect_pred) + 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1_inv)
    pos_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1*tf.math.log(vect_act1*vect_pred+ 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1)
    return pos_loss, neg_loss

def tensorDiceLossEager(vect_pred, vect_map1, vect_act1):
    vect_act1_inv = 1 - vect_act1
    dice_pos = 1 - 2.*tf.reduce_sum(vect_act1*vect_pred)/(tf.reduce_sum(vect_act1*vect_act1) + tf.reduce_sum(vect_pred*vect_pred)+1.e-15)
    dice_neg = 1 - 2.*tf.reduce_sum(vect_act1_inv*(1.-vect_pred))/(tf.reduce_sum(vect_act1_inv* vect_act1_inv) + tf.reduce_sum((1.-vect_pred)*(1.-vect_pred))+1.e-15)
    dice_loss = (dice_pos+dice_neg)/2
    return dice_loss    

def tfResizeFunc(iShape):
    wResize = tf.keras.layers.Resizing(*iShape[:2], interpolation="nearest")
    @tf.function
    def resize(x):
        return wResize(x)
    return resize

def resizeFunction(iLayerOutputDim, iMapInputShape):
    
    # if iLayerOutputDim==iMapInputShape:
    #     return lambda x: x
    # else:
    #     return tfResizeFunc(iLayerOutputDim)
    return tfResizeFunc(iLayerOutputDim)


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, iLossWeightDict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mLossWeightDict = iLossWeightDict
    
    def call(self, y_true, y_pred):
        vect_pred, vect_map1, vect_act1 = getVectors(y_true, y_pred[...,0,None])
        wLossWeightDict=self.mLossWeightDict
        wKeys = wLossWeightDict.keys()
        oLoss = tf.constant([0.])
        if 'Pos' in wKeys and 'Neg' in wKeys:
            wPosWeight, wNegWeight = wLossWeightDict['Pos'], wLossWeightDict['Neg']
            if wPosWeight and wNegWeight :
                wPos, wNeg = tensorPosNegLoss(vect_pred, vect_map1, vect_act1)
                oLoss += wPosWeight*wPos + wNegWeight*wNeg
        if 'Dice' in wKeys:
            wDiceWeight = wLossWeightDict['Dice']
            if wDiceWeight:
                oLoss+=tensorDiceLoss(vect_pred, vect_map1, vect_act1)
        return oLoss
    
@keras.saving.register_keras_serializable()
class CustomLoss2(tf.keras.losses.Loss):
    def __init__(self, iWeights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mWeights = iWeights
        self.mDice=tf.keras.losses.Dice()
        self.mWeightedCE = tf.nn.weighted_cross_entropy_with_logits

    
    def call(self, y_true, y_pred):
        return self.mWeights[0]*self.mDice(y_true, y_pred)# + self.mWeights[1]*self.mWeightedCE(y_true, y_pred, pos_weight=0.25)

@keras.saving.register_keras_serializable()
class CustomLoss3(tf.keras.losses.Loss):
    def __init__(self, iWeights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mWeights = iWeights
        self.mDice=tf.keras.losses.Dice()
        self.mWeightedCE = tensorPosNegLossV2
    
    def call(self, y_true, y_pred):
        y_true_act = tf.where(y_true<1., 0., y_true)
        return self.mWeights[0]*self.mDice(y_true_act, y_pred) + self.mWeights[1]*self.mWeightedCE(y_true, y_true_act, y_pred)
   
    def get_config(self):
        base_config = super().get_config()
        config = {
            "iWeights": self.mWeights,
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        wWeightList = [tf.constant(wWeight['config']['value']) for wWeight in config['iWeights']]
        return cls(wWeightList)
    
    
@keras.saving.register_keras_serializable()
class CustomLoss4(tf.keras.losses.Loss):
    def __init__(self, iWeights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mWeights = iWeights
        self.mDice=tensorDiceLossV2
        self.mWeightedCE = tensorPosNegLossV2
    
    def call(self, y_true, y_pred):
        y_true_act = tf.where(y_true<1., 0., y_true)
        return self.mWeights[0]*self.mDice(y_true_act, y_pred) + self.mWeights[1]*self.mWeightedCE(y_true, y_true_act, y_pred)
   
    def get_config(self):
        base_config = super().get_config()
        config = {
            "iWeights": self.mWeights,
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        wWeightList = [tf.constant(wWeight['config']['value']) for wWeight in config['iWeights']]
        return cls(wWeightList)
     
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd()))
gParentDir = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))

if __name__ =='__main__':
    
    wRotMap =RandomRotationHMap(0.13, fill_mode='constant', seed=0)
    wLoadSubDir = os.path.join('data2019', 'synth', 'test_01_tr_7000_dim_448_sample_4_frac_1e+00_res_3_blend_True_clr_True_uint',)
    wTFData = tf.data.Dataset.load(os.path.join(gParentDir, 'tfdataset', wLoadSubDir))
    wTFDataBatch = wTFData.batch(8)
    
    wX = next(iter(wTFDataBatch))
    for wOp in [wRotMap]:
        wAugX = wOp(wX)
        wAugIm, wAugMap = wAugX.values()
        plt.imshow(tf.cast(wAugIm[0,...,::-1], tf.uint8))
        plt.show()
        
        plt.imshow(tfFlatMap(wAugMap[0], inv=0))
        plt.show()
        
        # wEqAugMap = tf.vectorized_map(tfAct3D, wAugMap, fallback_to_while_loop=True, warn=True)[0]
        
        # plt.imshow(tfFlatMap(wEqAugMap, inv=0))
        # plt.show()
        
    tfAct3DBatchEager(wAugMap)
    tfAct3DBatch(wAugMap)
    
    wEagerTime=timeit.timeit(lambda: tfAct3DBatchEager(wAugMap), number=100)
    wFuncTime =timeit.timeit(lambda: tfAct3DBatch(wAugMap), number=100)
    print("Eager conv:", wEagerTime)
    print("Function conv:", wFuncTime)
    print(f"tf function was {wEagerTime/wFuncTime} faster!")
        
    
    tfFlatMapBatchEager(wAugMap)
    tfFlatMapBatch(wAugMap)
    
    wEagerTime=timeit.timeit(lambda: tfFlatMapBatchEager(wAugMap), number=100)
    wFuncTime =timeit.timeit(lambda: tfFlatMapBatch(wAugMap), number=100)
    print("Eager conv:", wEagerTime)
    print("Function conv:", wFuncTime)
    print(f"tf function was {wEagerTime/wFuncTime} faster!")
    
    
    getVectorsEager(tfFlatMapBatchEager(wAugMap), wAugMap)
    getVectors(tfFlatMapBatch(wAugMap), wAugMap)
    
    wEagerTime=timeit.timeit(lambda: getVectorsEager(tfFlatMapBatchEager(wAugMap), wAugMap), number=100)
    wFuncTime =timeit.timeit(lambda: getVectors(tfFlatMapBatch(wAugMap), wAugMap), number=100)
    print("Eager conv:", wEagerTime)
    print("Function conv:", wFuncTime)
    print(f"tf function was {wEagerTime/wFuncTime} faster!")
        
    
    
    tensorPosNegLossEager(*getVectorsEager(tfFlatMapBatchEager(wAugMap), wAugMap))
    tensorPosNegLoss(*getVectors(tfFlatMapBatch(wAugMap), wAugMap))
    
    wEagerTime=timeit.timeit(lambda: tensorPosNegLossEager(*getVectorsEager(tfFlatMapBatchEager(wAugMap), wAugMap)), number=100)
    wFuncTime =timeit.timeit(lambda: tensorPosNegLoss(*getVectors(tfFlatMapBatch(wAugMap), wAugMap)), number=100)
    print("Eager conv:", wEagerTime)
    print("Function conv:", wFuncTime)
    print(f"tf function was {wEagerTime/wFuncTime} faster!")
    
    tensorDiceLossEager(*getVectorsEager(tfFlatMapBatchEager(wAugMap), wAugMap))
    tensorDiceLoss(*getVectors(tfFlatMapBatch(wAugMap), wAugMap))
    
    wEagerTime=timeit.timeit(lambda: tensorDiceLossEager(*getVectorsEager(tfFlatMapBatchEager(wAugMap), wAugMap)), number=100)
    wFuncTime =timeit.timeit(lambda: tensorDiceLoss(*getVectors(tfFlatMapBatch(wAugMap), wAugMap)), number=100)
    print("Eager conv:", wEagerTime)
    print("Function conv:", wFuncTime)
    print(f"tf function was {wEagerTime/wFuncTime} faster!")
    
    y_pred, y_true_map, y_true = getVectors(tfFlatMapBatch(wAugMap), wAugMap)
    
    wModel = makeYoloType(iShape=(448,448,3), iRes=3, iDeeper=2)
      
    wWeightDict = {'Pos':tf.constant([5.]), 'Neg':tf.constant([25.]), 'Dice':tf.constant([1.])}
    myLoss1 = CustomLoss(None, wWeightDict)
    myLoss1(wAugMap, tfFlatMapBatch(wAugMap))
    wShapeList = [wOutput.shape[1:3] for wOutput in wModel.output[:-1]] + [None]
    
    wLossList = [CustomLoss(wShape, wWeightDict) for wShape in wShapeList]
    
    for wShape, wLoss in zip(wShapeList, wLossList):
        if wShape is None:
            wResize = lambda x: x
        else:
            wResize = tf.keras.layers.Resizing(*wShape[:2], interpolation="nearest")
        y_true = tfAct3DBatch(wResize(wAugMap))
        y_pred = tfFlatMapBatch(tfAct3DBatch(wResize(wAugMap)))
        print(y_true.shape, y_pred.shape)
        
        plt.imshow(y_pred[0])
        plt.show()
        print(wLoss(y_true, y_pred))    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
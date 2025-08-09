#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 11:08:49 2025

@author: ibabi
"""

import tensorflow as tf
import keras

from keras.layers import RandomElasticTransform
from keras.layers import RandomErasing
# from keras.layers import MixUp
class LocalElasticTransform(keras.layers.Layer):
    def __init__(self, seed = 0, **kwargs):
        super(LocalElasticTransform, self).__init__(**kwargs)
        self.mErase1 = RandomErasing(factor=(1., 1.),fill_value=0, seed=seed)
        self.mErase2 = RandomErasing(factor=(1., 1.),fill_value=0, seed=seed)
        self.mElastic = RandomElasticTransform(factor=(1., 1.), scale = .4)
        

    def call(self, inputs):
        if isinstance(inputs, dict):
            images = inputs['images']
        else:
            images=inputs
        wElasticPart = self.mElastic(images)
        wElasticPart = wElasticPart - self.mErase1(wElasticPart)
        wOtherPart = self.mErase2(images)
        images = wElasticPart + wOtherPart
        
        if isinstance(inputs, dict):
            inputs['images'] = images
        else:
            inputs = images
        return inputs
    
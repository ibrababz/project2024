#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 10:26:08 2025

@author: ibabi
"""
import tensorflow as tf
import keras

class Slice(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Slice, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs[...,0,None]
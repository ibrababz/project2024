#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:14:07 2025

@author: ibabi
"""

import tensorflow as tf
from functools import partial


@tf.function(input_signature=(tf.TensorSpec(shape=[None,None,None], dtype=tf.float32),))
def tfAct3D(iMap):
    return tf.math.multiply(tf.where(iMap < tf.math.reduce_max(iMap, axis = (0,1), keepdims=True), 0., 1.), iMap)

@tf.function(input_signature=(tf.TensorSpec(shape=[None,None,None,None], dtype=tf.float32),))
def tfAct3DBatch(iMapBatch):
    return tf.vectorized_map(tfAct3D, iMapBatch, fallback_to_while_loop=True, warn=True)


def tfAct3DEager(iMap):
    return tf.where(iMap < tf.math.reduce_max(iMap, axis = (0,1), keepdims=True), 0.,1.)

def tfAct3DBatchEager(iMapBatch):
    return tf.vectorized_map(tfAct3DEager, iMapBatch, fallback_to_while_loop=True, warn=True)

@tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),))
def tfScale3D(iMap):
    return tf.math.divide_no_nan(iMap, tf.math.reduce_max(iMap, axis = (0,1), keepdims=True), name=None)

@tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),))
def tfScale3DBatch(iMapBatch):
    return tf.vectorized_map(tfScale3D, iMapBatch, fallback_to_while_loop=True, warn=True)


@tf.function(input_signature=(tf.TensorSpec(shape=[None,None,None], dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32),))
def tfFlatMap(im_map, inv = 0):
    max_map = tf.math.reduce_max(im_map, axis = (0,1), keepdims=True) #get maximum of each channel
    min_map = tf.math.reduce_min(im_map, axis = (0,1), keepdims=True)
    im_map = tf.math.divide_no_nan((im_map-min_map),(max_map - min_map))
    if inv == 0:
        im_map = tf.math.reduce_max(im_map, axis=2, keepdims = True)
    elif inv == 1:
        im_map = tf.math.reduce_min(im_map, axis =2, keepdims = True)
    return im_map

@tf.function(input_signature=(tf.TensorSpec(shape=[None,None,None,None], dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32),))
def tfFlatMapBatch(iMapBatch, inv = 0):
    wPartialFlatMap = partial(tfFlatMap, inv=inv)
    return tf.vectorized_map(wPartialFlatMap, iMapBatch, fallback_to_while_loop=True, warn=True)


def tfFlatMapEager(im_map, inv = 0):
    max_map = tf.math.reduce_max(im_map, axis = (0,1), keepdims=True) #get maximum of each channel
    min_map = tf.math.reduce_min(im_map, axis = (0,1), keepdims=True)
    im_map = tf.math.divide_no_nan((im_map-min_map),(max_map - min_map))
    if not inv:
        im_map = tf.math.reduce_max(im_map, axis=2, keepdims = True)
    else:
        im_map = tf.math.reduce_min(im_map, axis =2, keepdims = True)
    return im_map

def tfFlatMapBatchEager(iMapBatch, inv=0):
    wPartialFlatMap = partial(tfFlatMapEager, inv=inv)
    return tf.vectorized_map(wPartialFlatMap, iMapBatch, fallback_to_while_loop=True, warn=True)


@tf.function#(input_signature=[tf.TensorSpec(shape=[8,56,56,1], dtype=tf.float32), tf.TensorSpec(shape=[8,56,56,4], dtype=tf.float32)])
def getVectors(iMapBatch, pred):
    act_list = tfAct3DBatch(iMapBatch)
    shape_pred = [tf.shape(pred)[k] for k in range(4)] #dynamic shape
    # shape_pred = pred.shape
    
    batch_size = shape_pred[0]
    vect_pred = tf.reshape(pred, (batch_size*shape_pred[1]*shape_pred[2],1))
    flat_map_arr1 = tfFlatMapBatch(iMapBatch)
    flat_act_arr1 = tfFlatMapBatch(act_list)
    
    # shape_map = flat_map_arr1.shape
    shape_map = [tf.shape(flat_map_arr1)[k] for k in range(4)] #dynamic shape
    # shape_act = flat_act_arr1.shape
    shape_act = [tf.shape(flat_act_arr1)[k] for k in range(4)] #dynamic shape
    
    vect_map1 = tf.reshape(flat_map_arr1, (shape_map[0]*shape_map[1]*shape_map[2],shape_map[3]))
    vect_act1 = tf.reshape(flat_act_arr1, (shape_act[0]*shape_act[1]*shape_act[2],shape_act[3]))

    return vect_pred, vect_map1, vect_act1

def getVectorsEager(pred, iMapBatch):
    act_list = tfAct3DBatchEager(iMapBatch)
    shape_pred = pred.shape
    batch_size = shape_pred[0]
    vect_pred = tf.reshape(pred, (batch_size*shape_pred[1]*shape_pred[2],1))
    flat_map_arr1 = tfFlatMapBatchEager(iMapBatch)
    flat_act_arr1 = tfFlatMapBatchEager(act_list)
    
    shape_map = flat_map_arr1.shape
    shape_act = flat_act_arr1.shape

    vect_map1 = tf.reshape(flat_map_arr1, (shape_map[0]*shape_map[1]*shape_map[2],shape_map[3]))
    vect_act1 = tf.reshape(flat_act_arr1, (shape_act[0]*shape_act[1]*shape_act[2],shape_act[3]))
    
    return vect_pred, vect_map1, vect_act1

    
    
    
    
    
    
    
    
    
    
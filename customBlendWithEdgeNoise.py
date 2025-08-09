#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 11:08:49 2025

@author: ibabi
"""

import tensorflow as tf
import keras

from keras.layers import GaussianNoise, GaussianDropout
from keras.layers import RandomErasing

from tensorflow.image import rgb_to_grayscale, sobel_edges 

from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator

@tf.function
def uniformThresholdBlendSobel(iIm, iDim= (14, 14), iThresh=0.75, iDirection=0, iSeed=0):
    
    # if tf.random.uniform(shape=()) < 0.5:
    if iIm.dtype == tf.uint8:
        iIm = tf.cast(iIm, dtype=tf.float32)
    # iIm = tf.expand_dims(iIm, 0)
    wGray = rgb_to_grayscale(iIm)
    wSobel = tf.math.abs(sobel_edges(wGray))/4.
    
    wEdges = tf.cond(
        pred = iDirection == 0,
        true_fn=lambda: tf.math.reduce_euclidean_norm(wSobel, axis=-1)/tf.math.sqrt(2.), 
        false_fn=lambda: tf.cond(pred= iDirection == 1, 
                                 true_fn=lambda:  wSobel[...,0],
                                 false_fn=lambda: wSobel[...,1],
                                 ),
    )
    
    wImShape= tf.shape(iIm)
    if len(wImShape) == 3:
        wNoiseShape = (iDim[0],iDim[1],1)
    elif len(wImShape) == 4:
        wNoiseShape = (wImShape[0], iDim[0],iDim[1],1)
    wNoise = tf.random.uniform(shape=wNoiseShape, seed=iSeed)
    wNoise = tf.where(wNoise>iThresh, 1., 0.)
    # wNoise = tf.where(wNoise>0., 1., 0.) #for testing different directions
    
    wNoise =tf.image.resize(wNoise, (iIm.shape[-3], iIm.shape[-2]))
    
    return tf.cast(iIm*(1-wNoise) + wEdges*wNoise, dtype=tf.uint8)
    # return iIm
# 
# from keras.layers import MixUp
class RandomBlendEdgeNoise(BaseImagePreprocessingLayer):
    def __init__(
        self,
        dim=None,
        threshold=None,
        direction=None,
        seed=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self.dim=dim
        self.threshold = threshold
        self.direction = direction
        self.seed = seed
        self.generator = SeedGenerator(seed)
        
        
    def transform_images(self, images, transformation=None, training=True): 
        images_shape = self.backend.shape(images)
        rank = len(images_shape)
        
        if rank == 3:
            batch_size = 1
        elif rank == 4:
            batch_size = images_shape[0]
        else:
            raise ValueError(
                "Expected the input image to be rank 3 or 4. Received "
                f"inputs.shape={images_shape}"
            )

        if training:
            
            transformation_probability = tf.ones(
                shape=(batch_size,),
            )*0.5
            
            random_threshold = self.backend.random.uniform(
                shape=(batch_size,),
                minval=0.0,
                maxval=1.0,
                # seed=self.seed,
            )#*0. #uncomment if you want this to always apply
            apply_transform = random_threshold <= transformation_probability
            
            if rank == 4:
                apply_transform = apply_transform[:, None, None, None]
            else:
                apply_transform = apply_transform[:, None, None]
            
            transformed_images = uniformThresholdBlendSobel(images, self.dim, self.threshold, self.direction, self.seed) 
            images = self.backend.numpy.where(
                    apply_transform,
                    transformed_images,
                    images,
                )

            return self.backend.cast(images, self.compute_dtype)
        return images
   
    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        return bounding_boxes
    
    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return segmentation_masks

class RandomDirBlendEdgeNoise(BaseImagePreprocessingLayer):
    def __init__(
        self,
        dim=None,
        threshold=None,
        seed=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self.dim=dim
        self.threshold = threshold
        self.seed = seed
        self.generator = SeedGenerator(seed)
        
        
    def transform_images(self, images, transformation=None, training=True): 
        images_shape = self.backend.shape(images)
        rank = len(images_shape)
        
        if rank == 3:
            batch_size = 1
        elif rank == 4:
            batch_size = images_shape[0]
        else:
            raise ValueError(
                "Expected the input image to be rank 3 or 4. Received "
                f"inputs.shape={images_shape}"
            )

        if training:
            
            transformation_probability = tf.ones(
                shape=(batch_size,),
            )*0.5
            
            random_threshold = self.backend.random.uniform(
                shape=(batch_size,),
                minval=0.0,
                maxval=1.0,
                # seed=self.seed,
            )#*0. #uncomment if you want this to always apply
            apply_transform = random_threshold <= transformation_probability
            
            if rank == 4:
                apply_transform = apply_transform[:, None, None, None]
            else:
                apply_transform = apply_transform[:, None, None]
                
            direction = self.backend.random.randint(shape=(), minval=0, maxval=3)
            transformed_images = uniformThresholdBlendSobel(images, self.dim, self.threshold, direction, self.seed) 
            images = self.backend.numpy.where(
                    apply_transform,
                    transformed_images,
                    images,
                )

            return self.backend.cast(images, self.compute_dtype)
        return images
   
    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self,
        bounding_boxes,
        transformation,
        training=True,
    ):
        return bounding_boxes
    
    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return segmentation_masks




class BlendWithEdgeNoise(keras.layers.Layer):
    def __init__(self, seed = 0, **kwargs):
        super(BlendWithEdgeNoise, self).__init__(**kwargs)
        self.probability = 1./3.

    def call(self, inputs, training=True):
        
        if isinstance(inputs, dict):
            images = inputs['images']
        else:
            images=inputs
            
        wRandom = tf.random.uniform([])
        
        apply_layer1 = wRandom < self.probability
        apply_layer2 = tf.logical_and(self.probability <= wRandom, wRandom < 1 - self.probability)
        images = tf.cond(
            pred=tf.logical_and(apply_layer1, training),
            true_fn=lambda: uniformThresholdBlendSobel(images, iDirection=0),
            false_fn=lambda: tf.cond(pred=tf.logical_and(apply_layer2, training), 
                                     true_fn=lambda: uniformThresholdBlendSobel(images, iDirection=1), 
                                     false_fn=lambda: uniformThresholdBlendSobel(images, iDirection=2),
                                     ),
        )
        
        if isinstance(inputs, dict):
            inputs['images'] = images
        else:
            inputs = images
       
        return inputs
    

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt    
    noiser = GaussianNoise(stddev = 0.1, seed=0)
    noiser2= GaussianDropout(rate=0.3, seed=0)
    ##testing stuff
    image = wX['images'][None,...]
    image = tf.cast(image, dtype=tf.float32)
    gray = rgb_to_grayscale(image) 
    sobel = tf.math.abs(sobel_edges(gray))/4.
    edges = tf.math.reduce_euclidean_norm(sobel, axis=-1)/tf.math.sqrt(2.)
    
    
    # noise = tf.random.uniform(shape = edges.shape)
    noise = noiser2(noiser(tf.zeros(shape=edges.shape), training=True))
    
    wBlend = BlendWithEdgeNoise()
    

    for i in range(3):
        wBlend = BlendEdgeNoise(dim=(14,14), threshold=0.75, direction=i%3, seed=0)
        wAug = wBlend(wX, training=False)
        wPlot = wAug['images'][...,::-1]
        if len(wPlot.shape) == 4:
            wPlot = wPlot[0]
        plt.imshow(wPlot)
        plt.show()
        plt.close()

        
        # noise = noiser(tf.zeros(shape=edges.shape), training=True)
        noise = tf.random.uniform(shape = (14,14,1))
        plt.imshow(noise)
        plt.show()
        noise = tf.where(noise>0.75, 1., 0.)
        noise = tf.image.resize(noise, (edges.shape[-3], edges.shape[-2]))
        
        plt.imshow(noise)
        plt.show()
        
        # aug = image*(1-noise) + sobel[...,1]*noise
        # aug = image*(1-noise) + sobel[...,1]*noise
        aug = image*(1-noise) + edges*noise
        aug = tf.cast(aug, dtype=tf.uint8)
        plt.imshow(aug[0,...,::-1])
        plt.show()
        plt.close()
        
        

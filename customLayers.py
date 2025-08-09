#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:39:52 2025

@author: ibabi
"""
import tensorflow as tf

from keras_cv.layers import RandomRotation

class RandomRotationHMap(RandomRotation):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def augment_segmentation_masks(
            self, segmentation_masks, transformations, **kwargs
        ):
            # If segmentation_classes is specified, we have a dense segmentation
            # mask. We therefore one-hot encode before rotation to avoid bad
            # interpolation during the rotation transformation. We then make the
            # mask sparse again using tf.argmax.
            if self.segmentation_classes:
                one_hot_mask = tf.one_hot(
                    tf.squeeze(tf.cast(segmentation_masks, tf.int32), axis=-1),
                    self.segmentation_classes,
                )
                rotated_one_hot_mask = self._rotate_images(
                    one_hot_mask, transformations
                )
                rotated_mask = tf.argmax(rotated_one_hot_mask, axis=-1)
                return tf.expand_dims(rotated_mask, axis=-1)
            else:
                if segmentation_masks.shape[-1] == 1:
                    raise ValueError(
                        "Segmentation masks must be one-hot encoded, or "
                        "RandomRotate must be initialized with "
                        "`segmentation_classes`. `segmentation_classes` was not "
                        f"specified, and mask has shape {segmentation_masks.shape}"
                    )
                rotated_mask = self._rotate_images(
                    segmentation_masks, transformations
                )
                # Round because we are in one-hot encoding, and we may have
                # pixels with ambiguous value due to floating point math for
                # rotation.
                return rotated_mask
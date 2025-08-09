# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_cv
import tensorflow as tf
from tensorflow.keras.layers import GaussianDropout, GaussianNoise, AveragePooling2D
from tensorflow.keras.layers import Pipeline
from customRotLayer import RandomRotationHMap
from customGaussianDropoutLayer import GaussianDropoutCustom
from customGaussianNoiseLayer import GaussianNoiseCustom
from customFourierLayer import FourierMixHMap
from customLocalElasticDeform import LocalElasticTransform
from customLocalElasticDeformV2 import RandomLocalElasticTransform
from customBlendWithEdgeNoise import RandomBlendEdgeNoise, RandomDirBlendEdgeNoise
from dataLoad import show_batch
from tfMapUtils import tfFlatMap, tfFlatMapBatch, tfScale3D, tfScale3DBatch
import matplotlib.pyplot as plt
# from customRotationHmap2 import RandomRotationHmap2

# from tensorflow.keras.layers import RandomFlip, RandomShear, RandomTranslation, RandomZoom

class RandomChance(tf.keras.layers.Layer):
    def __init__(self, layer, probability, **kwargs):
        super(RandomChance, self).__init__(**kwargs)
        self.layer = layer
        self.probability = probability

    def call(self, inputs, training=True):
        apply_layer = tf.random.uniform([]) < self.probability
        outputs = tf.cond(
            pred=tf.logical_and(apply_layer, training),
            true_fn=lambda: self.layer(inputs),
            false_fn=lambda: inputs,
        )
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer": tf.keras.layers.serialize(self.layer),
                "probability": self.probability,
            }
        )
        return config
    
class OneOfTwo(tf.keras.layers.Layer):
    def __init__(self, layer1, layer2, probability, **kwargs):
        super(OneOfTwo, self).__init__(**kwargs)
        self.layer1 = layer1
        self.layer2 = layer2
        self.probability = probability

    def call(self, inputs, training=True):
        apply_layer = tf.random.uniform([]) < self.probability
        outputs = tf.cond(
            pred=tf.logical_and(apply_layer, training),
            true_fn=lambda: self.layer1(inputs),
            false_fn=lambda: self.layer2(inputs),
        )
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer1": tf.keras.layers.serialize(self.layer1),
                "layer2": tf.keras.layers.serialize(self.layer2),
                "probability": self.probability,
            }
        )
        return config


class OneOfThree(tf.keras.layers.Layer):
    def __init__(self, layer1, layer2, layer3, probability=1./3., **kwargs):
        super(OneOfThree, self).__init__(**kwargs)
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.probability1 = probability
        self.probability2 = 1.-probability

    def call(self, inputs, training=True):
        wRandom = tf.random.uniform([])
        apply_layer1 = wRandom < self.probability1
        apply_layer2 = tf.logical_and(self.probability1 <= wRandom, wRandom < self.probability2)
        outputs = tf.cond(
            pred=tf.logical_and(apply_layer1, training),
            true_fn=lambda: self.layer1(inputs),
            false_fn=lambda: tf.cond(pred=tf.logical_and(apply_layer2, training), 
                                     true_fn=lambda: self.layer2(inputs), 
                                     false_fn=lambda: self.layer3(inputs),
                                     ),
        )
        
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer1": tf.keras.layers.serialize(self.layer1),
                "layer2": tf.keras.layers.serialize(self.layer2),
                "layer3": tf.keras.layers.serialize(self.layer3),
                "probability": self.probability,
            }
        )
        return config
    
class ReturnInput(tf.keras.layers.Layer):

  def __init__(self):
      super().__init__()

  def call(self, inputs):
      return inputs
  
class ReturnFirstOf2Input(tf.keras.layers.Layer):

  def __init__(self):
      super().__init__()

  def call(self, inputs, training=True):
      return inputs




# tf.random.set_seed(0)
wEq = keras_cv.layers.Equalization(value_range=(0, 255), seed=(0,0))


wSolar = keras_cv.layers.Solarization(value_range=(0, 255), seed=(0,0)) #V
wContrast = keras_cv.layers.RandomContrast(value_range=(0, 255), factor=0.2, seed=(0,0)) #V
wBright = keras_cv.layers.RandomBrightness(factor=(-0.1, 0.4), value_range=(0., 255.), seed=(0,0)) #V

wOneOfSolarBright = keras_cv.layers.RandomChoice(layers=[wSolar, wContrast, wBright, wEq], seed=(0,0))

wGrid = keras_cv.layers.GridMask(ratio_factor=(0.,0.2), rotation_factor=0.2, seed=(0,0))
wCutout = keras_cv.layers.RandomCutout(height_factor=0.15, width_factor=0.15, fill_mode="constant", seed=0)
wOneOfDropout = keras_cv.layers.RandomChoice(layers=[wGrid, wCutout], seed=(0,0))


wPost6 = keras_cv.layers.Posterization(value_range=(0, 255), bits=6, seed=(0,0)) 
wPost2 = keras_cv.layers.Posterization(value_range=(0, 255), bits=2, seed=(0,0))
wPost3 = keras_cv.layers.Posterization(value_range=(0, 255), bits=3, seed=(0,0))
wPost4 = keras_cv.layers.Posterization(value_range=(0, 255), bits=4, seed=(0,0))
wOneOfPost = keras_cv.layers.RandomChoice(layers=[wPost2, wPost3, wPost4, wPost6], seed=(0,0)) #already vectorized

wGaussBlur3 = keras_cv.layers.RandomGaussianBlur(kernel_size=3, factor=0.5, seed=(0,0))
wGaussBlur5 = keras_cv.layers.RandomGaussianBlur(kernel_size=5, factor=0.5, seed=(0,0))
wGaussBlur7 = keras_cv.layers.RandomGaussianBlur(kernel_size=7, factor=0.5, seed=(0,0))
wGaussBlur9 = keras_cv.layers.RandomGaussianBlur(kernel_size=9, factor=0.5, seed=(0,0))
wOneGauss = keras_cv.layers.RandomChoice(layers=[wGaussBlur3, wGaussBlur5, wGaussBlur7, wGaussBlur9], seed=(0,0)) #can't auto_vectorize



wChannelShuffle = keras_cv.layers.ChannelShuffle(groups=3, seed=(0,0)) #V
wClrJitter = keras_cv.layers.RandomColorJitter(value_range=(0, 255), brightness_factor=0.2, contrast_factor=0.2, saturation_factor=(0.5,0.8), hue_factor=(0.3,0.6), seed=(0,0)) #V
wClrDegen = keras_cv.layers.RandomColorDegeneration(factor=(0.3, 0.6), seed=(0,0)) #V
wHue = keras_cv.layers.RandomHue(factor=(0.3, 0.6), value_range=(0, 255), seed=(0,0)) #V
wSat = keras_cv.layers.RandomSaturation(factor=(0.5,0.8), seed=(0,0)) #V
wGrayIm = keras_cv.layers.Grayscale(output_channels=3, seed=(0,0)) #V

wOneOfClr = keras_cv.layers.RandomChoice(layers=[wClrDegen, wClrJitter, wHue, wSat, wGrayIm, wChannelShuffle], auto_vectorize=False, seed=(0,0))


wSharp = keras_cv.layers.RandomSharpness(factor=0.5, value_range=(0, 255), seed=(0,0)) #V


# wFourier = keras_cv.layers.FourierMix(seed=0)
wFourierHMap = FourierMixHMap(seed=0)

wRandGauss= keras_cv.layers.RandomApply(layer=wOneGauss, rate = 0.5, seed=0)

wThreeOf = keras_cv.layers.RandomAugmentationPipeline(layers=[wOneOfPost, wOneOfClr, wOneOfSolarBright, wSharp, wCutout], augmentations_per_image = 3, rate = 0.5, seed=(0,0))
wFourOf = keras_cv.layers.RandomAugmentationPipeline(layers=[wOneOfPost, wOneOfClr, wOneOfSolarBright, wSharp, wCutout], augmentations_per_image = 4, rate = 0.5, seed=(0,0))
wFiveOf = keras_cv.layers.RandomAugmentationPipeline(layers=[wOneOfPost, wOneOfClr, wOneOfSolarBright, wSharp, wCutout], augmentations_per_image = 5, rate = 0.5, seed=(0,0))
wOneOf = keras_cv.layers.RandomAugmentationPipeline(layers=[wOneGauss, wGrid], augmentations_per_image = 1, rate = 0.5, seed=(0,0))


wRotMap =RandomRotationHMap(0.25, fill_mode='constant', seed=(0,0))
# wRot =keras_cv.layers.RandomRotation(0.15, fill_mode='constant', seed=(0,0))
wFlip = keras_cv.layers.RandomFlip(mode='horizontal_and_vertical', rate=0.5, seed=(0,0))
wTrans = keras_cv.layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="constant", seed=(0,0))
wShear = keras_cv.layers.RandomShear(x_factor=0.4, y_factor=0.4, fill_mode="constant", interpolation="bilinear", seed =0)
wZoom = keras_cv.layers.RandomZoom(height_factor=0.25, width_factor=0.25, fill_mode="constant", seed=(0,0))

Augments = keras_cv.layers.RandomAugmentationPipeline(layers=[wOneGauss, wGrid], augmentations_per_image = 2, rate = 0.5, seed=(0,0))


# AffineAugments = keras_cv.layers.Augmenter([wThreeOf, wFlip, wRotMap, wShear, wZoom, wTrans])
# AffineAugments = keras_cv.layers.Augmenter([wThreeOf])#, wFlip, wShear, wZoom, wTrans])
# AffineAugments = keras_cv.layers.Augmenter([wFlip, wRotMap,  wShear, wZoom, wTrans])
#

    
wOneOfThreeGaussianDropout = OneOfThree(GaussianDropout(rate=0.05, seed=0), GaussianDropout(rate=0.075, seed=0), GaussianDropout(rate=0.1, seed=0))
wOneOfThreeGuassianNoise = OneOfThree(GaussianNoise(stddev=0.05, seed=0), GaussianNoise(stddev=0.075, seed=0), GaussianNoise(stddev=0.1, seed=0))
wOneOfTwoGaussian = OneOfTwo(wOneOfThreeGaussianDropout, wOneOfThreeGuassianNoise, probability=0.5)

Augments = keras_cv.layers.RandomAugmentationPipeline(layers=[wOneGauss, wGrid], augmentations_per_image = 2, rate = 0.5, seed=(0,0))


# AffineAugments = keras_cv.layers.Augmenter([wThreeOf, wFlip, wRotMap, wShear, wZoom, wTrans])
# AffineAugments = keras_cv.layers.Augmenter([wThreeOf])#, wFlip, wShear, wZoom, wTrans])
# AffineAugments = keras_cv.layers.Augmenter([wFlip, wRotMap,  wShear, wZoom, wTrans])
#

    
wOneOfThreeGaussianDropout = OneOfThree(GaussianDropout(rate=0.05, seed=0), GaussianDropout(rate=0.075, seed=0), GaussianDropout(rate=0.1, seed=0))
wOneOfThreeGuassianNoise = OneOfThree(GaussianNoise(stddev=0.05, seed=0), GaussianNoise(stddev=0.075, seed=0), GaussianNoise(stddev=0.1, seed=0))
wOneOfTwoGaussian = OneOfTwo(wOneOfThreeGaussianDropout, wOneOfThreeGuassianNoise, probability=0.5)
wElastic = tf.keras.layers.RandomElasticTransform(factor=1., scale = .4)#, fill_mode="constant")
wErase =  tf.keras.layers.RandomErasing(factor=(1., 1.), fill_value=0, seed=0)
wErase2 =  tf.keras.layers.RandomErasing(factor=(1., 1.),fill_value=0, seed=0)
wLocalElast = LocalElasticTransform(seed = 0)
wElastic = RandomLocalElasticTransform()
wBlend = RandomDirBlendEdgeNoise(dim=(14,14), threshold=0.75, seed=0)

from customRotationHmap2 import RandomRotationHmap2
from customTranslationHmap2 import RandomTranslationHmap2
from tensorflow.keras.layers import RandomFlip, RandomShear, RandomTranslation, RandomZoom, RandomPerspective
# wRotMap2 = tf.keras.layers.RandomRotation(factor=1., fill_mode='constant', seed=0)
wRotMap2 = RandomRotationHmap2(factor=1., fill_mode='constant', seed=0)
wFlip2 = RandomFlip()
wShear2 = RandomShear(x_factor=0.3, y_factor=0.3, fill_mode="constant")
wTrans2 = RandomTranslationHmap2(height_factor=0.3, width_factor=0.3, fill_mode="constant")
wZoom2 = RandomZoom(height_factor=(-0.3,0.3), width_factor=(-0.3,0.3), fill_mode="constant")
wPerspective = RandomPerspective()
wAverage = AveragePooling2D(3)
wAffine = tf.keras.models.Sequential()
wAffine.add(wBlend)
wAffine.add(wElastic)
wAffine.add(wPerspective)
wAffine.add(wFlip2)
wAffine.add(wRotMap2)
wAffine.add(wShear2)
wAffine.add(wZoom2)
wAffine.add(wTrans2)


def plotData(iData, iKey = 'images'):
    if isinstance(iData, dict):
        wPlot = iData[iKey]
        if iKey=='images':
            wPlot = tf.cast(wPlot[...,::-1], tf.uint8)
    else:
        wPlot = iData[...,::-1]

    if len(wPlot.shape) == 4:
        if iKey =='segmentation_masks':
            wPlot = tfFlatMapBatch(tfScale3DBatch(wPlot), inv=0)
        show_batch(wPlot)
    else:
        if iKey =='segmentation_masks':
            wPlot = tfFlatMap(tfScale3D(wPlot), inv=0)
        plt.imshow(wPlot)
        plt.show()
        plt.close()
    

# %%
# wAffine = keras_cv.layers.RandomAugmentationPipeline(layers=[wOneOfClr, wGrid], augmentations_per_image = 1, rate = 1., seed=(0,0))  
# for i in range(5):
#     # wAffine = tf.keras.models.Sequential()
#     # wBlend = RandomDirBlendEdgeNoise(dim=(14,14), threshold=0.75, seed=0)
#     # wAffine.add(wBlend)
#     # wAug = wRotMap2(wX)
#     # wAug = wFlip2(wX)
#     # wAug= wTrans2(wX)
    
#     wAug= wAverage(tf.cast(wX['images'], tf.float32)[None,...,::-1])
#     # plotData(wX)
#     plotData(tf.cast(wAug, tf.uint8))
#     # plotData(wX, 'segmentation_masks')
#     # plotData(wAug, 'segmentation_masks')
#     # print(wAffine.layers[0].apply_transform.numpy()[:,0,0,0])
# %%
    
#     wAug2 = wErase2(wX)
#     plt.imshow(np.uint8(wAug2['images'].numpy()[...,::-1]))
#     plt.show()
    
#     wAug3= wLocalElast(wX)
#     plt.imshow(np.uint8(wAug3['images'].numpy()[...,::-1]))
#     plt.show()
    
#     # plt.imshow(wAug['segmentation_masks'].numpy())
#     # plt.show()
#     plt.close()
    
# wNoise = RandomChance(wOneOfTwoGaussian, probability=0.5)


def ChooseAugment(iInt):
    #only Affine
    if iInt==0:
        AffineAugments = tf.keras.models.Sequential()
        AffineAugments.add(wFlip2)
        AffineAugments.add(wRotMap2)
        AffineAugments.add(wShear2)
        AffineAugments.add(wZoom2)
        AffineAugments.add(wTrans2)
        wNoise = ReturnFirstOf2Input()  
        Augments = ReturnInput()
    #only Noise
    elif iInt==1:
        AffineAugments = ReturnInput()  
        Augments = ReturnInput()  
        wNoise = RandomChance(wOneOfTwoGaussian, probability=0.5)
    #only Colour stuff
    elif iInt ==2:
        AffineAugments = keras_cv.layers.Augmenter([wThreeOf])#
        wNoise = ReturnFirstOf2Input()  
        Augments = keras_cv.layers.RandomAugmentationPipeline(layers=[wOneGauss, wGrid], augmentations_per_image = 2, rate = 0.5, seed=(0,0))  
    #more colour stuff
    elif iInt==3:
        AffineAugments = keras_cv.layers.Augmenter([wFourOf])#
        wNoise = ReturnFirstOf2Input()  
        Augments = keras_cv.layers.RandomAugmentationPipeline(layers=[wOneGauss, wGrid], augmentations_per_image = 2, rate = 0.5, seed=(0,0))  
    elif iInt==4:
        AffineAugments = keras_cv.layers.Augmenter([wFiveOf])#
        wNoise = ReturnFirstOf2Input()  
        Augments = keras_cv.layers.RandomAugmentationPipeline(layers=[wOneGauss, wGrid], augmentations_per_image = 2, rate = 0.5, seed=(0,0))
    elif iInt==5:
        AffineAugments = keras_cv.layers.Augmenter([wFiveOf])#
        wNoise = ReturnFirstOf2Input()  
        Augments = RandomChance(wFourierHMap, probability=0.5)
    elif iInt==6:
        AffineAugments = keras_cv.layers.Augmenter([wFiveOf, wFlip, wRotMap,  wShear, wZoom, wTrans])
        wNoise = ReturnFirstOf2Input()  
        Augments = ReturnInput()
    elif iInt==7:
        AffineAugments = keras_cv.layers.Augmenter([wFiveOf, wFlip, wRotMap,  wShear, wZoom, wTrans])
        wNoise = ReturnFirstOf2Input()  
        Augments = RandomChance(wFourierHMap, probability=0.5)
    elif iInt==8:
        AffineAugments = tf.keras.models.Sequential()
        AffineAugments.add(wBlend)
        AffineAugments.add(wElastic)
        AffineAugments.add(wFlip2)
        AffineAugments.add(wRotMap2)
        AffineAugments.add(wShear2)
        AffineAugments.add(wZoom2)
        AffineAugments.add(wTrans2)
        wNoise = ReturnFirstOf2Input()  
        Augments = ReturnInput()
    elif iInt==9:
        AffineAugments = tf.keras.models.Sequential()
        AffineAugments.add(wBlend)
        AffineAugments.add(wElastic)
        AffineAugments.add(wPerspective)
        AffineAugments.add(wFlip2)
        AffineAugments.add(wRotMap2)
        AffineAugments.add(wShear2)
        AffineAugments.add(wZoom2)
        AffineAugments.add(wTrans2)
        wNoise = ReturnFirstOf2Input()  
        Augments = ReturnInput()

    elif iInt==-1:
        AffineAugments = ReturnInput()
        wNoise = ReturnFirstOf2Input()
        Augments = ReturnInput()  
    return AffineAugments, Augments, wNoise

def chooseAugmentV2(ioData, iType):
    wDeform =  Pipeline([wBlend, wElastic, wPerspective])
    wAffine = Pipeline([wFlip2, wRotMap2, wShear2, wZoom2, wTrans2])
    
    if iType == 0: #No augments
        return ioData
    
    elif iType == 1: #Base affine only

        return ioData.map(wAffine, num_parallel_calls=tf.data.AUTOTUNE)
    
    elif iType == 2:
        
        ioData = ioData.map(wDeform, num_parallel_calls=tf.data.AUTOTUNE)
        return ioData.map(wAffine, num_parallel_calls=tf.data.AUTOTUNE)
        
    elif iType == 3:
        ioData = ioData.map(wDeform, num_parallel_calls=tf.data.AUTOTUNE)
        ioData = ioData.map(wThreeOf, num_parallel_calls=tf.data.AUTOTUNE)
        
        return ioData.map(wAffine, num_parallel_calls=tf.data.AUTOTUNE)
    
    elif iType == 4:
        ioData = ioData.map(wDeform, num_parallel_calls=tf.data.AUTOTUNE)
        ioData = ioData.map(wFourOf, num_parallel_calls=tf.data.AUTOTUNE)
        
        return ioData.map(wAffine, num_parallel_calls=tf.data.AUTOTUNE)
    
    elif iType == 5:            
        ioData = ioData.map(wDeform, num_parallel_calls=tf.data.AUTOTUNE)
        ioData = ioData.map(wFiveOf, num_parallel_calls=tf.data.AUTOTUNE)
        
        return ioData.map(wAffine, num_parallel_calls=tf.data.AUTOTUNE)
    
    elif iType == 6:
        ioData = ioData.map(wDeform, num_parallel_calls=tf.data.AUTOTUNE)
        ioData = ioData.map(wFiveOf, num_parallel_calls=tf.data.AUTOTUNE)
        ioData = ioData.map(wFourierHMap, num_parallel_calls=tf.data.AUTOTUNE)
        
        return ioData.map(wAffine, num_parallel_calls=tf.data.AUTOTUNE)

    return ioData


# def getAugments():
#     # @tf.function
#     def augment(iBatch):
#         return Augments(iBatch)
#     return augment


# def getNoise():
#     # @tf.function
#     def augment(iBatch, training=True):
#         return wNoise(iBatch, training=training)
#     return augment


# def getAffine():
#     # @tf.function
#     def augment(iBatch):
#         return AffineAugments(iBatch)
#     return augment
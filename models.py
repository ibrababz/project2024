from __future__ import absolute_import, division, print_function, unicode_literals
import os


"""#-----------------------------------Tensorflow Part-------------------------------------"""

import tensorflow as tf    
    
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization, Input, Add, Lambda, concatenate
#from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import backend as B
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import ResNet50, VGG16
#tf.compat.v1.disable_eager_execution()

from dataLoad import file_writing

def save_model_summary(save_dir, model):
    with open(os.path.join(save_dir, 'summary.txt'), 'w' ) as file:
        fwriting = file_writing(file)
        model.summary(print_fn = fwriting.write_file)
        
        
def makeYoloType(iShape, iFlag = 'resnet', iRes = 1):
    H, W, C = iShape
    
    if iFlag == 'resnet':
        base_model = ResNet50(include_top=False, input_shape= iShape, pooling='None', weights='imagenet')
        index =142
        index2 = 80
    elif iFlag == 'vgg':
        base_model = VGG16(include_top=False, input_shape=iShape ,pooling='None', weights='imagenet')
        index = 17
        index2 = 12 #filler
        
    base_model.trainable = False
    
    if iRes == 1:
        base_model2 = tf.keras.models.Model([base_model.inputs], [base_model.output, base_model.layers[index].output], name = 'base_model2')
        base_model2.trainable = False
        top_model = make_top_model_v2(base_model2.output[0].shape[1:], base_model2.output[1].shape[1:])
    elif iRes == 2:
        base_model2 = tf.keras.models.Model([base_model.inputs], [base_model.output, base_model.layers[index].output,  base_model.layers[index2].output], name = 'base_model2')
        base_model2.trainable = False
        top_model = make_top_model_v3(base_model2.output[0].shape[1:], base_model2.output[1].shape[1:], base_model2.output[2].shape[1:])
        
    model = tf.keras.models.Model(base_model2.inputs, top_model(base_model2.output))
    return model

def make_top_model_v2(base_model_output_shape, base_model_layeri_output_shape):
    #x is a tensor
    input1 = tf.keras.Input(shape = base_model_output_shape, name = "input_to_top")

    x = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_14')(input1)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_15')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_16')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
   
    xup = x
    x = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_1')(x)
    x = tf.keras.layers.Softmax(axis = -1)(x)
    
    input_from_layeri = tf.keras.Input(shape=base_model_layeri_output_shape, name = 'input_from_base_layeri')
    x_reduce_dim = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_red_dim_1')(input_from_layeri)
    x_reduce_dim = tf.keras.layers.Dropout(0.5)(x_reduce_dim)

    
    x5 = tf.keras.layers.UpSampling2D(size = (2,2), name = 'top_20')(xup)
    x5 = tf.keras.layers.concatenate([x5,x_reduce_dim],axis = 3)
    

    x5 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_22')(x5)
    x5 = tf.keras.layers.Dropout(0.5)(x5)
    x5 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_23')(x5)
    x5 = tf.keras.layers.Dropout(0.5)(x5)

    x5 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_28')(x5)
    x5 = tf.keras.layers.Dropout(0.5)(x5)
    x5 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_29')(x5)
    x5 = tf.keras.layers.Dropout(0.5)(x5)

    
    x5 = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_2')(x5)
    x5 = tf.keras.layers.Softmax(axis = -1)(x5)
    
    return tf.keras.models.Model([input1, input_from_layeri], [x,x5], name = 'top_model')

def make_top_model_v3(base_model_output_shape, base_model_layeri_output_shape, base_model_layeri_output_shape2):
    #x is a tensor
    input1 = tf.keras.Input(shape = base_model_output_shape, name = "input_to_top")

    x = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_14')(input1)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_15')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_16')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
   
    xup = x
    x = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_1')(x)
    x = tf.keras.layers.Softmax(axis = -1)(x)
    
    input_from_layeri = tf.keras.Input(shape=base_model_layeri_output_shape, name = 'input_from_base_layeri')
    x_reduce_dim = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_red_dim_1')(input_from_layeri)
    x_reduce_dim = tf.keras.layers.Dropout(0.5)(x_reduce_dim)

    
    x1 = tf.keras.layers.UpSampling2D(size = (2,2), name = 'top_20')(xup)
    x1 = tf.keras.layers.concatenate([x1,x_reduce_dim],axis = 3)
    

    x1 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_22')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_23')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)

    x1 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_28')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_29')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    
    xup = x1

    
    x1 = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_2')(x1)
    x1 = tf.keras.layers.Softmax(axis = -1)(x1)
    
    
    
    input_from_layeri2 = tf.keras.Input(shape = base_model_layeri_output_shape2, name = 'input_from_base_layeri2')
    x_reduce_dim = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_red_dim_2')(input_from_layeri2)
    x_reduce_dim = tf.keras.layers.Dropout(0.5)(x_reduce_dim)

    
    x2 = tf.keras.layers.UpSampling2D(size = (2,2), name = 'top_30')(xup)
    x2 = tf.keras.layers.concatenate([x2,x_reduce_dim],axis = 3)
    

    x2 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_31')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_32')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)

    x2 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_33')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_34')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    
    x2 = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_3')(x2)
    x2 = tf.keras.layers.Softmax(axis = -1)(x2)
    
    return tf.keras.models.Model([input1, input_from_layeri, input_from_layeri2], [x, x1, x2], name = 'top_model')



"""
base_model = ResNet50(include_top=False, input_shape=(448,448,3),pooling='None',weights='imagenet')

# layerName = "conv4_block6_out"
# base_model.get_layer(name = layerName).output.shape
# index = None
# for idx, layer in enumerate(base_model.layers):
#     if layer.name == layerName:
#         index = idx
#         break

index =142
base_model.trainable = False

inputs = tf.keras.Input(shape=(448,448,3), name = "input_1")
#layer_output_from_base_i = base_model.layers[index].output
base_model2 = tf.keras.models.Model([base_model.inputs], [base_model.output, base_model.layers[index].output], name = 'base_model2')
base_model2.trainable = False
plot_model(base_model, os.path.join(os.getcwd(), 'resnet50.png'), show_shapes = True)
plot_model(base_model2, os.path.join(os.getcwd(), 'base_model2.png'), show_shapes = True)
top_model = make_top_model((14,14,2048), (28,28,1024))

plot_model(top_model, os.path.join(os.getcwd(), 'top_model.png'), show_shapes = True)

model = tf.keras.models.Model(base_model2.inputs, top_model(base_model2.output))
model.summary()
#plot_model(model, os.path.join(os.getcwd(), 'full_model.png'), show_shapes = True)
"""

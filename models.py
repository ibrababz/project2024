from __future__ import absolute_import, division, print_function, unicode_literals
import os

from helper import adjust_number
"""#-----------------------------------Tensorflow Part-------------------------------------"""

import tensorflow as tf    
    
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization, Input, Add, Lambda, concatenate
#from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import backend as B
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import ResNet50, VGG16
from customModel import CustomModel
from customSliceLayer import Slice
#tf.compat.v1.disable_eager_execution()

from dataLoad import file_writing

def save_model_summary(save_dir, model):
    with open(os.path.join(save_dir, 'summary.txt'), 'w' ) as file:
        fwriting = file_writing(file)
        model.summary(print_fn = fwriting.write_file)
        
def makeYoloTypeFlat(iShape, iFlag = 'resnet', iRes = 2, iDeeper=0, iLegacy=True):
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
    
    if iRes == 2:
        base_model2 = tf.keras.models.Model([base_model.inputs], [base_model.output, base_model.layers[index].output], name = 'base_model2')
        base_model2.trainable = False
        top_model = make_top_model_v2(base_model2.output[0].shape[1:], base_model2.output[1].shape[1:])
    elif iRes == 3:
        # base_model2 = tf.keras.models.Model([base_model.inputs], [base_model.output, base_model.layers[index].output,  base_model.layers[index2].output], name = 'base_model2')
        base_model2 = tf.keras.models.Model(base_model.input, (base_model.output, base_model.layers[index].output,  base_model.layers[index2].output), name = 'base_model2')

        base_model2.trainable = False
        if not iDeeper:
            if iLegacy:
                top_model = make_top_model_v3(base_model2.output[0].shape[1:], base_model2.output[1].shape[1:], base_model2.output[2].shape[1:])
            else:
                top_model = make_top_model_v3p5_flat(*base_model2.output)
                # top_model = make_top_model_tiny_flat(*base_model2.output)
       
        elif iDeeper==1:
            top_model = make_top_model_v5([base_model2.output[0].shape[1:], base_model2.output[1].shape[1:], base_model2.output[2].shape[1:]], iWithTop=True)
        elif iDeeper==2:
            top_model = make_top_model_v6([base_model2.output[0].shape[1:], base_model2.output[1].shape[1:], base_model2.output[2].shape[1:]], iWithTop=True)
        elif iDeeper==-1:
            top_model = make_top_model_v7([base_model2.output[0].shape[1:], base_model2.output[1].shape[1:], base_model2.output[2].shape[1:]], iWithTop=True)
        elif iDeeper==-2:
            top_model = make_top_model_v8([base_model2.output[0].shape[1:], base_model2.output[1].shape[1:], base_model2.output[2].shape[1:]], iWithTop=True)
               
    model = tf.keras.models.Model(base_model2.input, top_model)
    # model = CustomModel(base_model2.inputs, top_model(base_model2.output))
    return model
    
        
def makeYoloType(iShape, iFlag = 'resnet', iRes = 2, iDeeper=0, iLegacy=True):
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
    
    if iRes == 2:
        base_model2 = tf.keras.models.Model([base_model.inputs], [base_model.output, base_model.layers[index].output], name = 'base_model2')
        base_model2.trainable = False
        top_model = make_top_model_v2(base_model2.output[0].shape[1:], base_model2.output[1].shape[1:])
    elif iRes == 3:
        base_model2 = tf.keras.models.Model([base_model.inputs], [base_model.output, base_model.layers[index].output,  base_model.layers[index2].output], name = 'base_model2')
        base_model2.trainable = False
        if not iDeeper:
            if iLegacy:
                top_model = make_top_model_v3(base_model2.output[0].shape[1:], base_model2.output[1].shape[1:], base_model2.output[2].shape[1:])
            else:
                top_model = make_top_model_v3p5(base_model2.output[0].shape[1:], base_model2.output[1].shape[1:], base_model2.output[2].shape[1:])
       
        elif iDeeper==1:
            top_model = make_top_model_v5([base_model2.output[0].shape[1:], base_model2.output[1].shape[1:], base_model2.output[2].shape[1:]], iWithTop=True)
        elif iDeeper==2:
            top_model = make_top_model_v6([base_model2.output[0].shape[1:], base_model2.output[1].shape[1:], base_model2.output[2].shape[1:]], iWithTop=True)
        elif iDeeper==-1:
            top_model = make_top_model_v7([base_model2.output[0].shape[1:], base_model2.output[1].shape[1:], base_model2.output[2].shape[1:]], iWithTop=True)
        elif iDeeper==-2:
            top_model = make_top_model_v8([base_model2.output[0].shape[1:], base_model2.output[1].shape[1:], base_model2.output[2].shape[1:]], iWithTop=True)
               
    model = tf.keras.models.Model(base_model2.inputs, top_model(base_model2.output))
    # model = CustomModel(base_model2.inputs, top_model(base_model2.output))
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
    x = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_1', kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x)
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

    
    x5 = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_2', kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x5)
    x5 = tf.keras.layers.Softmax(axis = -1)(x5)
    
    return tf.keras.models.Model([input1, input_from_layeri], [x,x5], name = 'top_model')

def make_top_model_v3(base_model_output_shape, base_model_layeri_output_shape, base_model_layeri_output_shape2):
    #x is a tensor
    input1 = tf.keras.Input(shape = base_model_output_shape, name = "input_to_top")

    x = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_11')(input1)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(512,(3,3), activation = 'relu', padding = "same", name='top_12')(x)
    x = tf.keras.layers.Dropout(0.5)(x)    
    x = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_13')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_14')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_15')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
   
    xup = x
    x = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_1')(x)#, kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x)
    x = tf.keras.layers.Softmax(axis = -1)(x)
    
    input_from_layeri = tf.keras.Input(shape=base_model_layeri_output_shape, name = 'input_from_base_layeri')
    x_reduce_dim = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_red_dim_1')(input_from_layeri)
    x_reduce_dim = tf.keras.layers.Dropout(0.5)(x_reduce_dim)

    
    x1 = tf.keras.layers.UpSampling2D(size = (2,2), name = 'top_20')(xup)
    x1 = tf.keras.layers.concatenate([x1,x_reduce_dim],axis = 3)
    

    x1 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_21')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_22')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_23')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(512,(3,3), activation = 'relu', padding = "same", name='top_24')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_25')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_26')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_27')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    
    xup = x1

    
    x1 = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_2')(x1)#, kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x1)
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
    x2 = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_33')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(512,(3,3), activation = 'relu', padding = "same", name='top_34')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_35')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_36')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_37')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    
    x2 = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_3')(x2)#, kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x2)
    x2 = tf.keras.layers.Softmax(axis = -1)(x2)
    
    return tf.keras.models.Model([input1, input_from_layeri, input_from_layeri2], [x, x1, x2], name = 'top_model')



def make_top_model_v3p5(base_model_output_shape, base_model_layeri_output_shape, base_model_layeri_output_shape2):
    #x is a tensor
    input1 = tf.keras.Input(shape = base_model_output_shape, name = "input_to_top")

    x = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_11')(input1)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(512,(3,3), activation = 'relu', padding = "same", name='top_12')(x)
    x = tf.keras.layers.Dropout(0.5)(x)    
    x = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_13')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_14')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_15')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
   
    xup = x
    x = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_1')(x)#, kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x)
    # x = tf.keras.layers.Softmax(axis = -1)(x)[...,0, None]
    x = tf.keras.layers.Softmax(axis = -1, name='softmax_1')(x)
    x = Slice(name='slice_1')(x)
    
    input_from_layeri = tf.keras.Input(shape=base_model_layeri_output_shape, name = 'input_from_base_layeri')
    x_reduce_dim = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_red_dim_1')(input_from_layeri)
    x_reduce_dim = tf.keras.layers.Dropout(0.5)(x_reduce_dim)

    
    x1 = tf.keras.layers.UpSampling2D(size = (2,2), name = 'top_20')(xup)
    x1 = tf.keras.layers.concatenate([x1,x_reduce_dim],axis = 3)
    

    x1 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_21')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_22')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_23')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(512,(3,3), activation = 'relu', padding = "same", name='top_24')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_25')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_26')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_27')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    
    xup = x1

    
    x1 = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_2')(x1)#, kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x1)
    # x1 = tf.keras.layers.Softmax(axis = -1)(x1)[...,0,None]
    x1 = tf.keras.layers.Softmax(axis = -1, name='softmax_2')(x1)
    x1 = Slice(name='slice_2')(x1)
    
    
    input_from_layeri2 = tf.keras.Input(shape = base_model_layeri_output_shape2, name = 'input_from_base_layeri2')
    x_reduce_dim = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_red_dim_2')(input_from_layeri2)
    x_reduce_dim = tf.keras.layers.Dropout(0.5)(x_reduce_dim)

    
    x2 = tf.keras.layers.UpSampling2D(size = (2,2), name = 'top_30')(xup)
    x2 = tf.keras.layers.concatenate([x2,x_reduce_dim],axis = 3)
    

    x2 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_31')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_32')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_33')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(512,(3,3), activation = 'relu', padding = "same", name='top_34')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_35')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_36')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_37')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    
    x2 = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_3')(x2)#, kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x2)
    # x2 = tf.keras.layers.Softmax(axis = -1)(x2)[...,0, None]
    x2 = tf.keras.layers.Softmax(axis = -1, name='softmax_3')(x2)
    x2 = Slice(name='slice_3')(x2)
    
    return tf.keras.models.Model([input1, input_from_layeri, input_from_layeri2], [x, x1, x2], name = 'top_model')

def make_top_model_v3p5_flat(x, x1, x2):
    #x is a tensor
    x = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_11', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x)
    x = tf.keras.layers.BatchNormalization(name='top_batch_norm_11')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(512,(3,3), activation = 'relu', padding = "same", name='top_12', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x)
    x = tf.keras.layers.BatchNormalization(name='top_batch_norm_12')(x)
    x = tf.keras.layers.Dropout(0.5)(x)    
    x = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_13', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x)
    x = tf.keras.layers.BatchNormalization(name='top_batch_norm_13')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_14', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x)
    x = tf.keras.layers.BatchNormalization(name='top_batch_norm_14')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_15', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x)
    x = tf.keras.layers.BatchNormalization(name='top_batch_norm_15')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
   
    xup = x
    x = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_1')(x)#, kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x)
    # x = tf.keras.layers.Softmax(axis = -1)(x)[...,0, None]
    x = tf.keras.layers.Softmax(axis = -1, name='softmax_1')(x)
    # x = tf.keras.layers.BatchNormalization(name='top_batch_norm_softmax_1')(x)#CREATES NAN's
    x = Slice(name='slice_1')(x)
    
    # input_from_layeri = tf.keras.Input(shape=base_model_layeri_output_shape, name = 'input_from_base_layeri')
    x_reduce_dim = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_red_dim_1', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x1)
    x_reduce_dim = tf.keras.layers.BatchNormalization(name='top_batch_norm_red_dim_1')(x_reduce_dim)
    x_reduce_dim = tf.keras.layers.Dropout(0.5)(x_reduce_dim)

    
    x1 = tf.keras.layers.UpSampling2D(size = (2,2), name = 'top_20')(xup)
    x1 = tf.keras.layers.concatenate([x1,x_reduce_dim],axis = 3)
    

    x1 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_21', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x1)
    x1 = tf.keras.layers.BatchNormalization(name='top_batch_norm_21')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_22', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x1)
    x1 = tf.keras.layers.BatchNormalization(name='top_batch_norm_22')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_23', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x1)
    x1 = tf.keras.layers.BatchNormalization(name='top_batch_norm_23')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(512,(3,3), activation = 'relu', padding = "same", name='top_24', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x1)
    x1 = tf.keras.layers.BatchNormalization(name='top_batch_norm_24')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_25', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x1)
    x1 = tf.keras.layers.BatchNormalization(name='top_batch_norm_25')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_26', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x1)
    x1 = tf.keras.layers.BatchNormalization(name='top_batch_norm_26')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_27', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x1)
    x1 = tf.keras.layers.BatchNormalization(name='top_batch_norm_27')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    
    xup = x1

    
    x1 = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_2')(x1)#, kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x1)
    # x1 = tf.keras.layers.Softmax(axis = -1)(x1)[...,0,None]
    x1 = tf.keras.layers.Softmax(axis = -1, name='softmax_2')(x1)
    # x1 = tf.keras.layers.BatchNormalization(name='top_batch_norm_softmax_2')(x1)#CREATES NAN's
    x1 = Slice(name='slice_2')(x1)
    
    
    # input_from_layeri2 = tf.keras.Input(shape = base_model_layeri_output_shape2, name = 'input_from_base_layeri2')
    x_reduce_dim = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_red_dim_2', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x2)
    x_reduce_dim = tf.keras.layers.BatchNormalization(name='top_batch_norm_red_dim_2')(x_reduce_dim)
    x_reduce_dim = tf.keras.layers.Dropout(0.5)(x_reduce_dim)

    
    x2 = tf.keras.layers.UpSampling2D(size = (2,2), name = 'top_30')(xup)
    x2 = tf.keras.layers.concatenate([x2,x_reduce_dim],axis = 3)
    

    x2 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_31', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x2)
    x2 = tf.keras.layers.BatchNormalization(name='top_batch_norm_31')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_32', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x2)
    x2 = tf.keras.layers.BatchNormalization(name='top_batch_norm_32')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_33', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x2)
    x2 = tf.keras.layers.BatchNormalization(name='top_batch_norm_33')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(512,(3,3), activation = 'relu', padding = "same", name='top_34', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x2)
    x2 = tf.keras.layers.BatchNormalization(name='top_batch_norm_34')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(256,(1,1), activation = 'relu', padding = "same", name='top_35', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x2)
    x2 = tf.keras.layers.BatchNormalization(name='top_batch_norm_35')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(256,(3,3), activation = 'relu', padding = "same", name='top_36', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x2)
    x2 = tf.keras.layers.BatchNormalization(name='top_batch_norm_36')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(128,(1,1), activation = 'relu', padding = "same", name='top_37', kernel_regularizer =tf.keras.regularizers.l2( l2=0.01))(x2)
    x2 = tf.keras.layers.BatchNormalization(name='top_batch_norm_37')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    
    x2 = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_3')(x2)#, kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x2)
    # x2 = tf.keras.layers.Softmax(axis = -1)(x2)[...,0, None]
    x2 = tf.keras.layers.Softmax(axis = -1, name='softmax_3')(x2)
    # x2 = tf.keras.layers.BatchNormalization(name='top_batch_norm_softmax_3')(x2) #CREATES NAN's
    x2 = Slice(name='slice_3')(x2)
    
    # return tf.keras.models.Model([input1, input_from_layeri, input_from_layeri2], [x, x1, x2], name = 'top_model')
    return x, x1, x2

def make_top_model_tiny_flat(x, x1, x2):
    #x is a tensor
    x = tf.keras.layers.Conv2D(32,(1,1), activation = 'relu', padding = "same", name='top_11')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', padding = "same", name='top_12')(x)
    x = tf.keras.layers.Dropout(0.5)(x)    
    x = tf.keras.layers.Conv2D(32,(1,1), activation = 'relu', padding = "same", name='top_13')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    
   
    xup = x
    x = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_1')(x)#, kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x)
    # x = tf.keras.layers.Softmax(axis = -1)(x)[...,0, None]
    x = tf.keras.layers.Softmax(axis = -1, name='softmax_1')(x)
    x = Slice(name='slice_1')(x)
    
    # input_from_layeri = tf.keras.Input(shape=base_model_layeri_output_shape, name = 'input_from_base_layeri')
    x_reduce_dim = tf.keras.layers.Conv2D(32,(1,1), activation = 'relu', padding = "same", name='top_red_dim_1')(x1)
    x_reduce_dim = tf.keras.layers.Dropout(0.5)(x_reduce_dim)

    
    x1 = tf.keras.layers.UpSampling2D(size = (2,2), name = 'top_20')(xup)
    x1 = tf.keras.layers.concatenate([x1,x_reduce_dim],axis = 3)
    

    x1 = tf.keras.layers.Conv2D(32,(1,1), activation = 'relu', padding = "same", name='top_21')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', padding = "same", name='top_22')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(64,(1,1), activation = 'relu', padding = "same", name='top_23')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', padding = "same", name='top_24')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(32,(1,1), activation = 'relu', padding = "same", name='top_25')(x1)

    
    xup = x1

    
    x1 = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_2')(x1)#, kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x1)
    # x1 = tf.keras.layers.Softmax(axis = -1)(x1)[...,0,None]
    x1 = tf.keras.layers.Softmax(axis = -1, name='softmax_2')(x1)
    x1 = Slice(name='slice_2')(x1)
    
    
    # input_from_layeri2 = tf.keras.Input(shape = base_model_layeri_output_shape2, name = 'input_from_base_layeri2')
    x_reduce_dim = tf.keras.layers.Conv2D(32,(1,1), activation = 'relu', padding = "same", name='top_red_dim_2')(x2)
    x_reduce_dim = tf.keras.layers.Dropout(0.5)(x_reduce_dim)

    
    x2 = tf.keras.layers.UpSampling2D(size = (2,2), name = 'top_30')(xup)
    x2 = tf.keras.layers.concatenate([x2,x_reduce_dim],axis = 3)
    

    x2 = tf.keras.layers.Conv2D(32,(1,1), activation = 'relu', padding = "same", name='top_31')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', padding = "same", name='top_32')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(64,(1,1), activation = 'relu', padding = "same", name='top_33')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', padding = "same", name='top_34')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)
    x2 = tf.keras.layers.Conv2D(32,(1,1), activation = 'relu', padding = "same", name='top_35')(x2)
    x2 = tf.keras.layers.Dropout(0.5)(x2)

    
    x2 = tf.keras.layers.Conv2D(2,(1,1), padding = "same", name='top_out_3')(x2)#, kernel_regularizer=tf.keras.regularizers.l2(1e-6), kernel_constraint=tf.keras.constraints.min_max_norm(min_value=1e-30, max_value=1.0))(x2)
    # x2 = tf.keras.layers.Softmax(axis = -1)(x2)[...,0, None]
    x2 = tf.keras.layers.Softmax(axis = -1, name='softmax_3')(x2)
    x2 = Slice(name='slice_3')(x2)
    
    # return tf.keras.models.Model([input1, input_from_layeri, input_from_layeri2], [x, x1, x2], name = 'top_model')
    return x, x1, x2

def removeClassificationLayers(iEncoderDecoder, iEncoderOutputIdxList, iDecoderOutputNames):
    wEncoderOutputs = [iEncoderDecoder.layers[wIdx].output for wIdx in iEncoderOutputIdxList]
    wEncoder = tf.keras.models.Model([iEncoderDecoder.inputs], wEncoderOutputs, name='base_model2')
    wDecoderOutputs = [iEncoderDecoder.layers[-1].get_layer(wName).output for wName in iDecoderOutputNames]
    wDecoder = tf.keras.models.Model(iEncoderDecoder.layers[-1].input, wDecoderOutputs, name=iEncoderDecoder.layers[-1].name)
    wX = wDecoder(wEncoder.output)
    return tf.keras.models.Model(wEncoder.input, wX, name='interim')

def removeClassificationLayersV2(iEncoderDecoder, iEncoderOutputIdxList, iDecoderOutputNames):
    wEncoderOutputs = [iEncoderDecoder.layers[wIdx].output for wIdx in iEncoderOutputIdxList]
    wEncoder = tf.keras.models.Model([iEncoderDecoder.inputs], wEncoderOutputs, name='base_model2')
    wDecoderOutputs = [iEncoderDecoder.layers[-1].get_layer(wName).output for wName in iDecoderOutputNames]
    wDecoder = tf.keras.models.Model(iEncoderDecoder.layers[-1].input, wDecoderOutputs, name=iEncoderDecoder.layers[-1].name)
    wInput = tf.keras.Input(shape=iEncoderDecoder.input.shape[1:], name='input')

    wX = wEncoder(wInput)
    
    wX = wDecoder(wX)
    return tf.keras.models.Model(wInput, wX, name='interim')

def addTransferLearnLayers(iModel, iDepthList, iKernelList, iActivation='relu', iPadding='same', iDropout=0.5, iPrefix='translearn'):
    wX = iModel(iModel.input)
    wOutList = []
    i=1
    for wXi in wX:
        wName = '_'.join([iPrefix, adjust_number(i,2)])
        wOut = addConv2DBlock(wXi, iDepthList, iKernelList, iActivation, iPadding, iDropout, wName)
        wOut = tf.keras.layers.Conv2D(2, 1, padding=iPadding, name=f'{iPrefix}_out_{i+1}')(wOut)
        wOutList.append(tf.keras.layers.Softmax(axis=-1)(wOut))
        i+=1
    return tf.keras.models.Model(iModel.input, wOutList, name='transfer_learn_model')

def addTransferLearnLayersV2(iModel, iDepthList, iKernelList, iActivation='relu', iPadding='same', iDropout=0.5, iPrefix='tl'):
    wX = iModel(iModel.input)
    wInputList=[]
    for i in range(len(wX)):
        wInputList.append(tf.keras.Input(shape=wX[i].shape[1:], name=f'input_tl_{i+1}'))
    wOutList = []
    i=1
    for wXi in wInputList:
        wName = '_'.join([iPrefix, adjust_number(i,2)])
        wOut = addConv2DBlock(wXi, iDepthList, iKernelList, iActivation, iPadding, iDropout, wName)
        wOut = tf.keras.layers.Conv2D(2, 1, padding=iPadding, name=f'{iPrefix}_out_{i+1}')(wOut)
        wOutList.append(tf.keras.layers.Softmax(axis=-1)(wOut))
        i+=1
    wTLModel = tf.keras.models.Model(wInputList, wOutList, name='tl_model')
    wX= wTLModel(wX)
    
    return tf.keras.models.Model(iModel.input, wX, name='full_tl_model')

def addTransferLearnLayersV3(iModel, iEncoderOutputIdxList, iDecoderOutputNames, iDepthList, iKernelList, iActivation='relu', iPadding='same', iDropout=0.5, iBatchNorm=True, iPrefix='tl'):
    wEncoderOutputs = [iModel.layers[i].output for i in iEncoderOutputIdxList]
    wEncoder = tf.keras.models.Model([iModel.inputs], wEncoderOutputs, name='base_model2')
    wDecoderOutputs = [iModel.layers[-1].get_layer(wName).output for wName in iDecoderOutputNames]
    wDecoder = tf.keras.models.Model(iModel.layers[-1].input, wDecoderOutputs, name=iModel.layers[-1].name)
    # wInput = tf.keras.Input(shape=iModel.input.shape[1:], name='input') 
    # wX = wEncoder(wInput)
    # wX = wDecoder(wX)
    wX = wDecoder(wEncoder.output)
    wInputList=[]
    
    for i in range(len(wX)):
        wInputList.append(tf.keras.Input(shape=wX[i].shape[1:], name=f'input_tl_{i+1}'))
    wOutList = []
    i=1
    for wXi in wInputList:
        wName = '_'.join([iPrefix, adjust_number(i,2)])
        wOut = addConv2DBlock(wXi, iDepthList, iKernelList, iActivation, iPadding, iDropout, iBatchNorm, wName)
        wOut = tf.keras.layers.Conv2D(2, 1, padding=iPadding, name=f'{iPrefix}_out_{i+1}')(wOut)
        wOutList.append(tf.keras.layers.Softmax(axis=-1)(wOut))
        i+=1
    wTLModel = tf.keras.models.Model(wInputList, wOutList, name='tl_model')

    wX = wTLModel(wX)
    
    return tf.keras.models.Model(wEncoder.input, wX, name='full_tl_model')

        
def addConv2DBlock(iInput, iDepthList, iKernelList, iActivation='relu', iPadding='same', iDropout=0.5, iBatchNorm=True, iNamePrefix='top', iNameList=None):
    wX = iInput
    
    for i in range(len(iDepthList)):
        wDepth = iDepthList[i]
        wPool = iKernelList[i]
        if iNameList is not None:
            wName = iNameList[i]
        else:
            wName = '_'.join([iNamePrefix, adjust_number(i,2)])
        wX =tf.keras.layers.Conv2D(wDepth, wPool, activation=None, padding=iPadding, name=wName)(wX)
        wX = tf.keras.layers.ReLU(name='_'.join(['relu']+wName.split('_')[1:]))(wX)
        if iDropout:
            wX = tf.keras.layers.Dropout(iDropout, name='_'.join(['dropout']+wName.split('_')[1:]))(wX)
        if iBatchNorm:
            wX = tf.keras.layers.BatchNormalization(name='_'.join(['batchnorm']+wName.split('_')[1:]))(wX)
    return wX

def addMultiConv2DBlocks(iInputList, iDepthListList, iKernelListList, iActivation='relu', iPadding='same', iDropout=0.5, iNamePrefix='top'):
    nInputs = len(iInputList)
    nDepthLists = len(iDepthListList)
    nKernelLists = len(iKernelListList)
    wXList =[]
    
    for i in range(nInputs):
        if nDepthLists < nInputs:
            wDepthList = iDepthListList[0]
        else:
            wDepthList = iDepthListList[i]
            
        if nKernelLists < nInputs:
            wKernelList = iKernelListList[0]
        else:
            wKernelList = iKernelListList[i]

        wInput = iInputList[i]
        wSuffix = str(i+1)
        wNameList= ['_'.join([iNamePrefix, wSuffix+f"{j+1}"]) for j in range(len(wDepthList))]
        
        wX = addConv2DBlock(wInput, wDepthList, wKernelList, iActivation, iPadding, iDropout, iNameList=wNameList)
        wXList.append(wX)
    return wXList


def upsampleInputs(iInputList, iUpSampleFlags, iPrefix='top'):
    nEncoderOutputs = len(iInputList)
    oProcessedInputList =[]
    for i in range(nEncoderOutputs):
        wUpSampleFlag = iUpSampleFlags[i]
        wInput = iInputList[i]
        if wUpSampleFlag:
            wUpSampleName = '_'.join([iPrefix, f"{i}0"])
            wUpSample = tf.keras.layers.UpSampling2D(size = (2,2), name=wUpSampleName)(wInput)
            wInput = wUpSample
        oProcessedInputList.append(wInput)
    return oProcessedInputList

def reduceInputDepths(iInputList, iReduceDimDepths, iActivation='relu', iPadding='same', iDropout=0.5, iPrefix ='top'):
    nEncoderOutputs = len(iInputList)
    oProcessedInputList =[]
    for i in range(nEncoderOutputs):
        wReduceDimDepth = iReduceDimDepths[i]
        wInput = iInputList[i]
        if wReduceDimDepth is not None:
            wReduceDimName = '_'.join([iPrefix, 'red_dim', str(i)])
            wXReduceDim = tf.keras.layers.Conv2D(wReduceDimDepth,(1,1), activation=iActivation, padding=iPadding, name=wReduceDimName)(wInput)
            if iDropout:
                wXReduceDim = tf.keras.layers.Dropout(iDropout)(wXReduceDim)
            wInput = wXReduceDim
        oProcessedInputList.append(wInput)
    return oProcessedInputList

def reduceInputDepth(iInput, iReduceDimDepth, iIdx, iActivation='relu', iPadding='same', iDropout=0.5, iPrefix ='top'):
    oInput = iInput
    if iReduceDimDepth is not None:
        wReduceDimName = '_'.join([iPrefix, 'red_dim', str(iIdx)])
        wXReduceDim = tf.keras.layers.Conv2D(iReduceDimDepth,(1,1), activation=iActivation, padding=iPadding, name=wReduceDimName)(iInput)
        if iDropout:
            wXReduceDim = tf.keras.layers.Dropout(iDropout, name='_'.join(['dropout']+wReduceDimName.split('_')[1:]))(wXReduceDim)
        oInput = wXReduceDim

    return oInput


def createDecoderModel(iShapeList, iInputNameList, iReduceDimDepthList, iDepthListList, iKernelListList, iActivation='relu', iPadding='same', iDropout=0.5, iBatchNorm=True, iWithTop=False, iPrefix='top'):
    wEncoderOutputList = []
    wReduceDepthInputs = []
    wXConvBlockList =[]
    wXUpList=[]
    wXCatList=[]
    for i in range(len(iShapeList)):
        wEncoderOutputList.append(tf.keras.Input(shape = iShapeList[i], name=iInputNameList[i]))
        wReduceDepthInputs.append(reduceInputDepth(wEncoderOutputList[i], iReduceDimDepthList[i], i, iActivation, iPadding, iDropout, iPrefix))
        wNameList = [f"top_{i+1}{j+1}" for j in range(len(iDepthListList[i]))]
        if i==0:
            wXConvBlockList.append(addConv2DBlock(wReduceDepthInputs[i], iDepthListList[i], iKernelListList[i], iActivation, iPadding, iDropout, iBatchNorm, iPrefix, wNameList))
        else:
            wName=f"top_{i+1}0"
            wXUpList.append(tf.keras.layers.UpSampling2D(size=(2,2), name=wName)(wXConvBlockList[i-1]))
            wXCatList.append(tf.keras.layers.concatenate([wXUpList[i-1],wReduceDepthInputs[i]], axis=3, name = '_'.join(['concatenate']+wName.split('_')[1:])))
            wXConvBlockList.append(addConv2DBlock(wXCatList[i-1], iDepthListList[i], iKernelListList[i], iActivation, iPadding, iDropout, iBatchNorm, iPrefix, wNameList))
       
    if iWithTop:
        for i in range(len(wXConvBlockList)):
            wXConvBlockList[i] = tf.keras.layers.Conv2D(2, 1, padding=iPadding, name=f'top_out_{i+1}')(wXConvBlockList[i])
            wXConvBlockList[i] = tf.keras.layers.Softmax(axis=-1, name=f'softmax_{i+1}')(wXConvBlockList[i])[...,0,None]
    
    return tf.keras.models.Model(wEncoderOutputList, wXConvBlockList, name='decoder')


def make_top_model_v4(iWithTop):
    wShapeList = [(14,14,2048), (28,28,1024), (56,56,512)]
    wReduceDimDepthList = [None, 256, 256]
    wDepthListList = [[256,512,256,256,128], [128,256,256,512,256,256,128], [128,256,256,512,256,256,128]]
    wKernelListList= [[1,3,1,3,1], [1,3,1,3,1,3,1], [1,3,1,3,1,3,1]]
    wInputNameList= ['input_to_top', "input_from_base_layeri", "input_from_base_layeri2"]

    return createDecoderModel(wShapeList, wInputNameList, wReduceDimDepthList, wDepthListList, wKernelListList, iActivation='relu', iPadding='same', iDropout=0.5, iBatchNorm=True, iWithTop=iWithTop)
def make_top_model_v5(iShapeList, iWithTop):
    # wShapeList = [(14,14,2048), (28,28,1024), (56,56,512)]
    wReduceDimDepthList = [None, 256, 256]
    wDepthListList = [[128,256,256,512,512,1024,512,512,256,256,128]]*3
    wKernelListList= [[1,3,1,3,1,3,1,3,1,3,1]]*3
    wInputNameList= ['input_to_top', "input_from_base_layeri", "input_from_base_layeri2"]

    return createDecoderModel(iShapeList, wInputNameList, wReduceDimDepthList, wDepthListList, wKernelListList, iActivation='relu', iPadding='same', iDropout=0.5, iBatchNorm=True, iWithTop=iWithTop)

def make_top_model_v6(iShapeList, iWithTop):
    # wShapeList = [(14,14,2048), (28,28,1024), (56,56,512)]
    wReduceDimDepthList = [None, None, None]
    wDepthListList = [[1024,512,256], [512,256,128], [256, 128,64]]
    wKernelListList= [[3]*len(wList) for wList in wDepthListList]
    wInputNameList= ['input_to_top', "input_from_base_layeri", "input_from_base_layeri2"]

    return createDecoderModel(iShapeList, wInputNameList, wReduceDimDepthList, wDepthListList, wKernelListList, iActivation='relu', iPadding='same', iDropout=0.5, iBatchNorm=True, iWithTop=iWithTop)


def make_top_model_v7(iShapeList, iWithTop):
    # wShapeList = [(14,14,2048), (28,28,1024), (56,56,512)]
    wReduceDimDepthList = [None, 128, 128]
    wDepthListList = [[128,128,64], [64,128,128,128,64], [64,128,128,128,64]]
    wKernelListList= [[1,3,1], [1,3,1,3,1], [1,3,1,3,1]]
    wInputNameList= ['input_to_top', "input_from_base_layeri", "input_from_base_layeri2"]

    return createDecoderModel(iShapeList, wInputNameList, wReduceDimDepthList, wDepthListList, wKernelListList, iActivation='relu', iPadding='same', iDropout=0.5, iBatchNorm=True, iWithTop=iWithTop)

def make_top_model_v8(iShapeList, iWithTop):
    # wShapeList = [(14,14,2048), (28,28,1024), (56,56,512)]
    wReduceDimDepthList = [None, 64, 64]
    wDepthListList = [[64,64,32], [32,64,64,64,32], [32,64,64,64,32]]
    wKernelListList= [[1,3,1], [1,3,1,3,1], [1,3,1,3,1]]
    wInputNameList= ['input_to_top', "input_from_base_layeri", "input_from_base_layeri2"]

    return createDecoderModel(iShapeList, wInputNameList, wReduceDimDepthList, wDepthListList, wKernelListList, iActivation='relu', iPadding='same', iDropout=0.5, iBatchNorm=True, iWithTop=iWithTop)


if __name__ =='__main__':
    
    wModel = makeYoloTypeFlat(iShape=(448,448,3), iRes=3, iDeeper=0, iLegacy=False)

    wDepthList = [128, 256, 256, 256, 128]
    wKernelList = [1, 3, 1, 3, 1]
    
    wNoClassModel=removeClassificationLayers(wModel, [-2, 142, 80], ['top_15', 'top_27', 'top_37'])
    wTLModel = addTransferLearnLayersV3(wNoClassModel, [-2, 142, 80], ['top_15', 'top_27', 'top_37'], wDepthList, wKernelList)

# wShapeList = [(14,14,2048), (28,28,1024), (56,56,512)]
# wReduceDimDepthList = [None, 256, 256]
# wDepthListList = [[256,512,256,256,128], [128,256,256,512,256,256,128], [128,256,256,512,256,256,128]]
# wKernelListList= [[1,3,1,3,1], [1,3,1,3,1,3,1], [1,3,1,3,1,3,1]]
# wInputNameList= ['input_to_top', "input_from_base_layeri", "input_from_base_layeri2"]

# createDecoderModel(wShapeList, wInputNameList, wReduceDimDepthList, wDepthListList, wKernelListList, iActivation='relu', iPadding='same', iDropout=0.5, iWithTop=True).summary()

# iModel = wModel
# iDecoderName = 'top_model'
            
# for wLayer in iModel.layers:
#     if wLayer.name == iDecoderName:
#         wDecoder= wLayer
#         break
# wOutputLayers = []
# for wLayer in wDecoder.layers:
#     if 'top_out' in wLayer.name:
#         print(wLayer.name)
#         wOutputLayers.append(wLayer.output)
        
# wInputList =[]

# for wIdx, wName in zip([-2, 142, 80], ["input_to_top", 'input_from_base_layeri', 'input_from_base_layeri2']):
#     wInputList.append(tf.keras.Input(shape = mModel.layers[wIdx].output.shape, name = wName))
# wDecoder = tf.keras.models.Model(wInputList, iModel.layers[-1].output, 'top_model_2')    

# mModel = tf.keras.models.Model([iModel.inputs], [iModel.layers[-2].output,iModel.layers[142].output, iModel.layers[80].output] , name = 'transfer_learn_model')
# wLayersOut = [wLayer.output for wLayer in iModel.layers[-1].layers if 'top_out' in wLayer.name]
# mDecoder = tf.keras.models.Model(iModel.layers[-1].input, wLayersOut, name = 'decoder')

# wInput = tf.keras.Input(shape = mModel.inputs[0].shape[1:], name = 'input')
# wY = mModel(wInput)
# wY = mDecoder(wY)


# """
# base_model = ResNet50(include_top=False, input_shape=(448,448,3),pooling='None',weights='imagenet')

# # layerName = "conv4_block6_out"
# # base_model.get_layer(name = layerName).output.shape
# # index = None
# # for idx, layer in enumerate(base_model.layers):
# #     if layer.name == layerName:
# #         index = idx
# #         break

# index =142
# base_model.trainable = False

# inputs = tf.keras.Input(shape=(448,448,3), name = "input_1")
# #layer_output_from_base_i = base_model.layers[index].output
# base_model2 = tf.keras.models.Model([base_model.inputs], [base_model.output, base_model.layers[index].output], name = 'base_model2')
# base_model2.trainable = False
# plot_model(base_model, os.path.join(os.getcwd(), 'resnet50.png'), show_shapes = True)
# plot_model(base_model2, os.path.join(os.getcwd(), 'base_model2.png'), show_shapes = True)
# top_model = make_top_model((14,14,2048), (28,28,1024))

# plot_model(top_model, os.path.join(os.getcwd(), 'top_model.png'), show_shapes = True)

# model = tf.keras.models.Model(base_model2.inputs, top_model(base_model2.output))
# model.summary()
# #plot_model(model, os.path.join(os.getcwd(), 'full_model.png'), show_shapes = True)
# """

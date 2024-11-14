# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 08:27:55 2022

@author: i_bab
"""
import file
import csv
import cv2 as cv
from joblib import dump, load
import os

import numpy as np
from dataLoad import image_list_from_batch, generate_batch, show_batch
from dataLoad import process_batch_3D, flat_map_list_v2, name_list_from_batch
from dataLoad import ProcessMapList3D
from dataLoad import GenerateWeedDataList, WeedDataLoader, RandomWeedBatchGenerator
from dataLoad import loadDataFilesAsObjects, genDataFiles, getMapListsFromBatch, getImageListFromBatch
from helper import show_wait, adjust_number
from augment_utils import affine_augments, augments
import tensorflow as tf 
from copy import deepcopy

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization, Input, Add, Lambda, concatenate
from tensorflow.keras import backend as B  
from tensorflow.keras.utils import plot_model

from tensorflow.keras.applications import ResNet50
from models import make_top_model_v2
from plot_utils import plot_test_train

from loss_functions import loss_from_map_tensor, tensor_pos_neg_loss
from loss_functions import act_list_3D, dist_loss_from_list, tensor_map_loss, tensor_dice_loss

from transformers import  AutoImageProcessor
from swinBackbones import getSwinTBackBone#, getSwinLBackBone


#%%
def save_model_summary(save_dir, model):
    with open(os.path.join(save_dir, 'summary.txt'), 'w' ) as file:
        fwriting = file_writing(file)
        model.summary(print_fn = fwriting.write_file)
        
def save_dataset_names(save_dir, data_names, subset_name, label =''):
    with open(os.path.join(save_dir, subset_name+ '_'+label+ '.txt'), 'w' ) as file:
        fwriting = file_writing(file)
        for name in data_names:
            fwriting.write_file(name)
            
def save_original_image_names(save_dir, XY_data, subset_name, label ='1'): #just for orignal image objects
    with open(os.path.join(save_dir, 'original_image_name_' + subset_name+ '_'+label+ '.txt'), 'w' ) as file:
        fwriting = file_writing(file)
        for XY in XY_data:
            fwriting.write_file(XY.name)
      
        
def remove_old_model(old_name):
    #print('removing ', old_name)
    if os.path.exists(old_name):
        os.remove(old_name)
    else:
        print("Cannot remove a file that does not exist") 

def copyTemp(temp):
    if temp is not None:
        return temp.deepcopy()
    else:
        return temp
    
def augmentXData(X_train, seq):
    if seq is not None:
        return seq.deepcopy()(images = X_train)
    else:
        return deepcopy(X_train)
    
def normListBy(iList, iNorm):
    oList = []
    for wEle in iList:
        oList.append(wEle/iNorm)
    return oList

def ComputeLosses(X_train, map_lists, model, modelFlag, dim, image_processor, batch_loss_tracker, batch_acc_tracker, i, j, seq = None, train = True, iNorm = 0):
    weight = 5.
    temp = copyTemp(seq)        
    X_train_aug = augmentXData(X_train, temp)
    X_show = X_train_aug.copy()
    
    if iNorm:
        X_train_aug = normListBy(X_train_aug, iNorm)
    
    # X_train_aug = tf.keras.applications.resnet50.preprocess_input(np.array(X_train_aug, dtype = np.float32)[...,::-1]) 
    # # [...,::-1]index to RGB so preprocess can reconvert to BGR and Zero center the data and don't divide by 255.0
    if modelFlag == 'swin':
        X_train_aug = image_processor(np.array(X_train_aug), return_tensors="tf")['pixel_values']
    elif modelFlag =='resnet':
        X_train_aug = image_processor(np.array(X_train_aug, dtype = np.float32)[...,::-1])
    
    y_pred = model(X_train_aug, training = train)
    y_pred0, y_pred1 = y_pred[0], y_pred[1]
    dim_grid1, dim_grid2 = y_pred0[0,...,0].shape, y_pred1[0,...,0].shape
    dim_grid1, dim_grid2 = (dim_grid1[0], dim_grid1[1]), (dim_grid2[0], dim_grid2[1])
    # print('dim_grid1 = ', dim_grid1, 'dim_grid2 = ', dim_grid2)
    
    map_aug_list1, weight_list1, _ = ProcessMapList3D(map_lists[0], dim, dim_grid1, copyTemp(temp))
    map_aug_list2, weight_list2, _ = ProcessMapList3D(map_lists[1], dim, dim_grid2,  copyTemp(temp))
    
    act_aug_list1 = act_list_3D(map_aug_list1)
    act_aug_list2 = act_list_3D(map_aug_list2)
    
    pos1,neg1 = tensor_pos_neg_loss(y_pred0[...,0], map_aug_list1, act_aug_list1)
    pos2,neg2 = tensor_pos_neg_loss(y_pred1[...,0], map_aug_list2, act_aug_list2)
    
    # map1 = tensor_map_loss(y_pred0[...,0], map_aug_list1, act_aug_list1)
    map1 = tf.constant(0.)
    
    map2 = tf.constant(0.)
    # map2 = tensor_map_loss(y_pred1[...,0], map_aug_list2, act_aug_list2)
    
    #dist1 = dist_loss_from_list(weight_list1, y_pred0[...,0], 1)
    #dist2 = dist_loss_from_list(weight_list2, y_pred1[...,0], 1)
    # dist2 = tf.constant(0.)
    
    # dist1 = tf.constant(0.)
    # dist2 = tf.constant(0.)
    
    # loss1 = pos1 + neg1
    # loss1_update = pos1 + weight*neg1 + weight2*dist1 + weight3*map1
    # loss1 = tf.constant(0.)
    # loss1_update = loss1
    
    # loss2 = pos2 + neg2
    # loss2_update = pos2 + weight*neg2 + weight3*map2 + weight2*dist2
    # loss2 = tf.constant(0.)
    # loss2_update = loss2
    
    # loss1_update = pos1 + weight*neg1 + B.abs(B.sum([pos1, -neg1]))/B.sum([pos1, neg1]) + dist1
    
    
    #==============================DICE UPDATE START====================
    dice1 = tensor_dice_loss(y_pred0[...,0], map_aug_list1, act_aug_list1)
    # dice2, pos2, neg2, dist2 = tf.constant(0.), tf.constant(0.), tf.constant(0.), tf.constant(0.)
    
    # dice1, pos1, neg1, dist1 = tf.constant(0.), tf.constant(0.), tf.constant(0.), tf.constant(0.)
    dice2 = tensor_dice_loss(y_pred1[...,0], map_aug_list2, act_aug_list2)
    
    loss1 = dice1 + pos1 + neg1 #+ dist1
    loss2 = dice2 + pos2 + neg2 #+ dist2
    
    loss1_update = dice1 + 5.*(pos1 + weight*neg1)# + weight2*dist1)
    loss2_update = dice2 + 5.*(pos2 + weight*neg2)# + weight2*dist2)
    #==============================DICE UPDATE END====================

    loss_value = loss1 + loss2
    # acc_value = dice1 + dice2 #tf.constant(0.) #loss2
    acc_value = map1 + map2
    if j == 5 and i%15 == 0:
        if train:
            print("dim_grid1:",dim_grid1, "dim_grid2:", dim_grid2)
        show_batch(list(X_train))
        
        if i%2 ==0:
            map_list1, _, _ = ProcessMapList3D(map_lists[0], dim, dim_grid1, None)
            show_batch(flat_map_list_v2(map_list1))
        else:
            map_list2, _, _ = ProcessMapList3D(map_lists[1], dim, dim_grid2, None)
            show_batch(flat_map_list_v2(map_list2))
            
        # # show_batch(list(X_train_aug))
        
        show_batch(X_show)
        if i%2 ==0:
            show_batch(flat_map_list_v2(map_aug_list1))
            show_batch(flat_map_list_v2(act_aug_list1))
            show_batch(list(y_pred0[...,0].numpy()[...,None]))
        else:
            show_batch(flat_map_list_v2(map_aug_list2))
            show_batch(flat_map_list_v2(act_aug_list2))
            show_batch(list(y_pred1[...,0].numpy()[...,None]))
        
    return loss1_update, loss2_update, loss_value, acc_value

def TrainEpochV3(model, modelFlag, dim, opt, batch_generator_01, image_processor, epoch, seq, train = True, iNorm = 0):
    i = epoch

    batch_loss_tracker = []
    batch_acc_tracker = []
    j = 0

    for batch in batch_generator_01:

        X_train = getImageListFromBatch(batch)
        map_lists = getMapListsFromBatch(batch)
      
        if train:
            
            with tf.GradientTape() as tape:
                  
                loss1_update, loss2_update, loss_value, acc_value = ComputeLosses(X_train, map_lists, model, modelFlag, dim, image_processor, batch_loss_tracker, batch_acc_tracker, i, j, seq, train, iNorm)
            
            grads = tape.gradient([loss1_update, loss2_update], model.trainable_weights)
            opt.apply_gradients((grad, var) for (grad, var) in zip(grads, model.trainable_weights) if grad is not None)
            
        else:
            loss1_update, loss2_update, loss_value, acc_value = ComputeLosses(X_train, map_lists, model, modelFlag, dim, image_processor, batch_loss_tracker, batch_acc_tracker, i, j, seq, train, iNorm)
        
        batch_loss_tracker.append(loss_value.numpy())
        batch_acc_tracker.append(acc_value)

        j+=1
    loss_per_epoch = np.mean(batch_loss_tracker)
    acc_per_epoch = np.mean(batch_acc_tracker)

    return loss_per_epoch, acc_per_epoch

def check_train_loss(min_train_loss_counter, loss_per_epoch, min_train_loss):
    if(loss_per_epoch < min_train_loss):
        min_train_loss = loss_per_epoch
        min_train_loss_counter = 0
    else:
        min_train_loss_counter +=1
    
    return min_train_loss, min_train_loss_counter

def check_val_loss(min_val_loss_counter, valid_loss_per_epoch, min_val_loss, old_name, epoch):
    if(valid_loss_per_epoch < min_val_loss):
      min_val_loss = valid_loss_per_epoch
      current_name = save_and_print(epoch, model, save_dir, add_flag = '_min_val')
      if old_name is not None:
          remove_old_model(old_name)
      old_name = current_name
      min_val_loss_counter =0
    else:
        min_val_loss_counter +=1
    return min_val_loss, min_val_loss_counter, old_name

def check_for_break(min_train_loss_counter, number_of_epochs_to_break_train, min_val_loss_counter, number_of_epochs_to_break_val):
      if min_train_loss_counter >= number_of_epochs_to_break_train: 
          print('BREAKING NEWS: no improvement training')
          return 1
      elif min_val_loss_counter >= number_of_epochs_to_break_val: #break from training
          print('BREAKING NEWS: no improvement validating')
          return 1
      else:
          return 0    


def getUnfreezeIndexList(dw, n):
    n_dw = int(n/dw)
    indexes = []
    for w in range(n_dw+1):
        idx1 = -(w)*dw-1
        if idx1 > -n:
            indexes.append(idx1)
            idx2 = -(w+1)*dw-1
            if idx2 <=-n:
                indexes.append(-n)
    return indexes

def TrainAndTestV2(XY_train, XY_valid, model, modelFlag, opt, batch_size, image_processor, epochs, save_dir, unfreezeFlag = 0, iNorm = 0):

    min_val_loss = np.inf
    min_train_loss = np.inf
    
    train_loss_tracker = []
    valid_loss_tracker = []
    train_acc_tracker = []
    valid_acc_tracker = []
    old_name = None
    
    min_train_loss_counter = 0
    min_val_loss_counter = 0
    number_of_epochs_to_break_train = np.inf
    number_of_epochs_to_break_val = np.inf #200
    #train_flag, valid_flag = 0, 0
    # five_saves_ago = 15
    # start_saving = 15
    # save_freq = 5
    # save_names = []
    
    last_n = 5
    min_val_loss_list = [np.inf]
    min_name_list = ['inf']
    
    last_n_name_list = []    
    
    dim = XY_train[0].getDim()
    start, end = epochs
    wUnfrzIdx = ''
    if unfreezeFlag:
        dUnfrzSeg= 3
        n = len(model.layers)
        wIndexList = getUnfreezeIndexList(dUnfrzSeg, n)
    
    for i in range(start, end):
        #print("\nStart of epoch %d" % (i,))
        seq = augments() #affine_augments() (if iNorm =0 i.e. loaded data are already floats b/w 0 and 1)
        seq.seed_(i) 
         
        batch_gen_train = generate_batch(XY_train, batch_size, i)
        loss_per_epoch, acc_per_epoch= TrainEpochV3(model, modelFlag, dim, opt, batch_gen_train, image_processor, i, seq, True, iNorm) 
        # loss_per_epoch, acc_per_epoch = train_epoch_NEW(XY_train, model, dim, None, batch_size, i, None, False)
        #compute loss on unaugmented training data for metrics, but train using augmented training data
        
        train_loss_tracker.append(loss_per_epoch)
        train_acc_tracker.append(acc_per_epoch)

        batch_gen_valid = generate_batch(XY_valid, batch_size, 0)
        valid_loss_per_epoch, valid_acc_per_epoch = TrainEpochV3(model, modelFlag, dim, None,  batch_gen_valid, image_processor, i,None, False, iNorm)

   
        valid_loss_tracker.append(valid_loss_per_epoch)
        valid_acc_tracker.append(valid_acc_per_epoch)

        sum_loss_per_epoch = np.sum([loss_per_epoch])
        sum_val_loss_per_epoch = np.sum([valid_loss_per_epoch])
        
        sum_acc_per_epoch = np.sum([acc_per_epoch])
        sum_val_acc_per_epoch = np.sum([valid_acc_per_epoch])
        
        min_train_loss, min_train_loss_counter = check_train_loss(min_train_loss_counter, sum_loss_per_epoch, min_train_loss)
        min_val_loss, min_val_loss_counter, old_name = check_val_loss(min_val_loss_counter, sum_val_loss_per_epoch, min_val_loss, old_name, i)

        max_idx_sorted = np.flip(np.argsort(min_val_loss_list))
        max_idx = max_idx_sorted[0]
        max_min_val_loss = min_val_loss_list[max_idx]
        # max_min_val_loss = np.inf
        if sum_val_loss_per_epoch <= max_min_val_loss:
            min_val_loss_list.append(sum_val_loss_per_epoch)
            min_name_list.append(save_and_print(i,model,save_dir,'_list_'+str(sum_val_loss_per_epoch)+ wUnfrzIdx,'.hdf5', 0))
            saveAndPrintOpt(i, opt, save_dir,'_list_'+str(sum_val_loss_per_epoch)+ wUnfrzIdx +'_opt', '.npy',0)

            if len(min_val_loss_list)  > last_n:
                min_val_loss_list.pop(max_idx)
                delete_name = min_name_list.pop(max_idx)
                remove_old_model(delete_name)
                remove_old_model(delete_name.strip('.hdf5')+'_opt.npy')
            
        last_n_name_list.append(save_and_print(i,model,save_dir,'_last_'+str(last_n) +'_'+str(sum_val_loss_per_epoch)+ wUnfrzIdx,'.hdf5', 0))
        saveAndPrintOpt(i,opt,save_dir,'_last_'+str(last_n) +'_'+str(sum_val_loss_per_epoch)+ wUnfrzIdx+'_opt', '.npy',0)
        
        if len(last_n_name_list)  > last_n:
            delete_name = last_n_name_list.pop(0)
            remove_old_model(delete_name)        
            remove_old_model(delete_name.strip('.hdf5')+'_opt.npy')
            
        if (i-start) > 0 and (i-start)%25 ==0:
            save_and_print(i,model,save_dir,'_savept_'+str(sum_val_loss_per_epoch)+ wUnfrzIdx, '.hdf5', 0)
            saveAndPrintOpt(i, opt, save_dir,'_savept_'+str(sum_val_loss_per_epoch)+ wUnfrzIdx +'_opt', '.npy', 0)
            
        B.set_value(opt.iterations, i)
        if (i-start) == 400 or (i - start) == 650: #250
            B.set_value(opt.learning_rate, opt.learning_rate.numpy()/10.)
            print('changed learning rate')
        #B.set_value(opt.lr, opt._decayed_lr(tf.float32).numpy())
        #print('tr L:', loss_per_epoch,'A:',loss1_per_epoch,'+ ',loss2_per_epoch ,'  val L:', valid_loss_per_epoch, '+',valid_loss1_per_epoch,'+ ',valid_loss2_per_epoch, ' epoch: ', i, ' min_val: ', min_val_loss)
        print('tr L:', np.round(loss_per_epoch,3),'A:',np.round(acc_per_epoch, 3), 'val L:', np.round(valid_loss_per_epoch,3), 'A:',np.round(valid_acc_per_epoch,3), 'ep:', i, 'min_V:', np.round(min_val_loss, 3))
        
        
        if check_for_break(min_train_loss_counter, number_of_epochs_to_break_train, min_val_loss_counter, number_of_epochs_to_break_val):
            break
        
        #break #FOR DEBUGGING
        if unfreezeFlag:
            if (i-start)%5 == 0 and (i-start) >= 400 and len(wIndexList) >1:
                idx1 = wIndexList.pop(0)
                idx2 = wIndexList[0] 
                print("setting model.layers[%s:%s] to trainable"%(idx2,idx1))
                for layer in model.layers[idx2:idx1]:
                    layer.trainable = True
                wUnfrzIdx = '_Unfrz_'+ str(idx2)                    
                
    return [train_loss_tracker, train_acc_tracker], [valid_loss_tracker,  valid_acc_tracker]


def save_and_print(i, model, save_dir, add_flag = '', iFileFormat = '.hdf5',verbose = 1):
    if verbose:
        print('-----saving weights------')
    number = adjust_number(i)
    full_name = os.path.join(save_dir, 'model_epoch_'+number+add_flag+iFileFormat)
    model.save_weights(full_name)
    return full_name

def saveAndPrintOpt(i, iOpt, save_dir, add_flag = '', iFileFormat = '.hdf5', verbose = 1):
    if verbose:
        print('-----saving weights------')
    number = adjust_number(i)
    full_name = os.path.join(save_dir, 'model_epoch_'+number+add_flag+iFileFormat)
    np.save(full_name, iOpt.get_weights())
    return full_name

class file_writing():
    def __init__(self, iFile):
        self.file = iFile
    def write_file(self, string):
        self.file.write(string)
        self.file.write('\n')
    

def load_model_weight_2_new_model(model1, model2, shape, shape2, n_weights, load_path, trainable = 0):
    model_loaded = model1(shape)
    model_loaded.load_weights(load_path)

    ##LOAD MODEL WITH PRETTY TRAINING/VALIDATING SCORE AND TEST OUT ON SOME PICS
    ##ALSO USE DISTANCE IOU AS DIFFERENT LOSS METRIC

    new_model = model2(shape2)
    for layer, layer2 in zip(new_model.layers[:n_weights], model_loaded.layers):
        print(layer.name)
        if not trainable:
            layer.trainable = False
        layer.set_weights(layer2.get_weights())
    
    for layer in new_model.layers:
        print(layer.name, ' trainable = ', layer.trainable)
    return new_model

def load_model_weight_2_new_model_by_layer_name(model1, model2, shape, shape2, load_path, trainable = 0):
    model_loaded = model1(shape)
    model_loaded.load_weights(load_path)
    ##LOAD MODEL WITH PRETTY TRAINING/VALIDATING SCORE AND TEST OUT ON SOME PICS
    ##ALSO USE DISTANCE IOU AS DIFFERENT LOSS METRIC

    new_model = model2(shape2)
    
    for layer in new_model.layers:
        for layer2 in model_loaded.layers:
            if layer.name == layer2.name:
                if not trainable:
                    layer.trainable = False
                layer.set_weights(layer2.get_weights())
    
    for layer in new_model.layers:
        print(layer.name, ' trainable = ', layer.trainable)
    return new_model


def set_all_trainable(model):
    for layer in model.layers:
        layer.trainable = True

def test_pop(some_list):
    some_list.pop(0)

#%%
if __name__ == '__main__':

    physical_devices = tf.config.list_physical_devices('GPU')
    
    ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))
    
#%%
    iNorm = 255.
    iSrcDir = "data2019\\1\\train_synth_8bit"
    # iSrcDir = "data4k\\train_real"
    iSrcPath = os.path.join(ROOT_DIR, iSrcDir)         
    wDataObjectList = loadDataFilesAsObjects(iSrcPath)
#%%
    # for wDataObj in wDataObjectList[:15]:
    #     show_wait(wDataObj.getImage(), 2)
        
#%%
    iValidSrcDir = "data2019\\1\\valid_synth_8bit"
    # iValidSrcDir = "data4k\\valid_real"
    iValidSrcPath = os.path.join(ROOT_DIR, iValidSrcDir)         
    wValidDataObjectList = loadDataFilesAsObjects(iValidSrcPath)
    
  
#%%
    ###================================BASE MODEL CNN BEGIN======================================
    shape = wDataObjectList[0].getShape()
    H, W, C = shape
    dim = (W, H)
    base_model = ResNet50(include_top=False, input_shape= shape,pooling='None',weights='imagenet')
    index =142
    base_model.trainable = False

    inputs = tf.keras.Input(shape=shape, name = "input_1")
    base_model2 = tf.keras.models.Model([base_model.inputs], [base_model.output, base_model.layers[index].output], name = 'base_model2')
    base_model2.trainable = False
    modelFlag = 'resnet'
    
    ###================================BASE MODEL CNN END========================================
#%%    
    ###================================BASE MODEL SWIN BEGIN======================================
    # project_folder = "project2024"
    # swin_folder = "archive" 
    # swin_pretrn_dir = os.path.join(ROOT_DIR, project_folder, swin_folder)
    # swin_pretrn_file = "saved_model.pb"
    # swin_pretrn_path = os.path.join(swin_pretrn_dir, swin_pretrn_file)
    # base_model = tf.keras.models.load_model(swin_pretrn_dir)
    # output = base_model(tf.ones((5,224,224,3)), training = True)
    
    
    # swin_pre_layers = base_model.layers
    # for i in range(len(swin_pre_layers)):
    #     print("layer[",i,"]: ",swin_pre_layers[i].name)
        
    # pretr_input = tf.keras.Input((224,224,3))
    # x = swin_pre_layers[0](pretr_input)
    # x = swin_pre_layers[1](x)
    # x = swin_pre_layers[2](x)
    # x = swin_pre_layers[3](x)
    # x = swin_pre_layers[4].layers[0](x)
    # x = swin_pre_layers[4].layers[1](x)
    # x = swin_pre_layers[4].layers[2](x)
    # x = swin_pre_layers[4].layers[3](x)
    # x = swin_pre_layers[4].layers[4](x)
    # x1 = swin_pre_layers[4].layers[5](x)
    
    # x = swin_pre_layers[4].layers[6](x1)
    # x = swin_pre_layers[5](x)
    # x2 = swin_pre_layers[6](x)
    
    # h1, w1, d1 = int(np.sqrt(x1.shape[1])), int(np.sqrt(x1.shape[1])), int(x1.shape[-1])
    # x1 = tf.keras.layers.Reshape((h1,w1,d1)) (x1)   
    
    # h2, w2, d2 = int(np.sqrt(x2.shape[1])), int(np.sqrt(x2.shape[1])), int(x2.shape[-1])
    # x2 = tf.keras.layers.Reshape((h2,w2,d2)) (x2)
    
    # base_model2 = tf.keras.models.Model(pretr_input, [x2,x1])
    # base_model2.trainable = False
    
    # output = base_model2(tf.ones((5,224,224,3)))
    
    # print(output[0].shape)
    # print(output[1].shape)
    # prep = tf.keras.applications.resnet50.preprocess_input(tf.ones((5,224,224,3)))
    # print(prep.shape)
    ###================================BASE MODEL SWIN END========================================
    
#%%    
    # ###================================BASE MODEL SWIN BEGIN -ALTERNATE======================================
    
    # base_model2 = getSwinTBackBone() #getSwinTBackBone()
    # base_model2.trainable = False
    # modelFlag = 'swin'

    # ###================================BASE MODEL SWIN END -ALTERNATE========================================
#%%    
    ###================================TOP MODEL BEGIN======================================
    
    top_model = make_top_model_v2(base_model2.output[0].shape[1:], base_model2.output[1].shape[1:])

    ###================================TOP MODEL END========================================
    model = tf.keras.models.Model(base_model2.inputs, top_model(base_model2.output))
    unfreezeFlag = 1 # 1
#%%
    ###==============================LOADING MODEL TO TRAIN START==============================
    # for layer in model.layers:
    #     print(layer.trainable, ' name: ', layer.name)

    
    # model.layers[-1].summary()
    # load_folder = "resnet_t3000_v900_A_ep_200_lr_1e-03"
    # load_dir = os.path.join(os.getcwd(), load_folder)
    # model.load_weights(os.path.join(load_dir,"model_epoch_0199_list_2.6824248.hdf5" ))

    # for layer in model.layers[-1].layers:
    #     if layer.name in ['top_14', 'top_15', 'top_16', 'top_out_1']:
    #         layer.trainable = False
    #     print(layer.trainable, ' name: ', layer.name)
    

    ###==============================LOADING MODEL TO TRAIN END================================

#%%
    ###==============================LOADING MODEL TO TRAIN START (ALT-USE)==============================
    for layer in model.layers:
        print(layer.trainable, ' name: ', layer.name)
    
    # model.layers[0].trainable = True
    # for layer in model.layers:
    #     layer.trainable = True
    

    model.layers[-1].summary()
    
    load_folder ="resnet_t3000_v900_fullcycle_8bit_test2_ep_(0, 1200)_lr_1e-04" 
    
    opt_file = "model_epoch_0425_savept_3.789702_Unfrz_-16_opt.npy"
    opt_weights = np.load(os.path.join(load_folder, opt_file), allow_pickle=True)
    optimizer = tf.keras.optimizers.Adam(1e-5)
    
    for layer in model.layers:
        layer.trainable = False
    for i in range(len(model.layers)):
        if i <16:
            model.layers[-i].trainable = True
            
    with tf.GradientTape() as tape:
        a = 0
    grads = tape.gradient([tf.constant(0.), tf.constant(0.)], model.trainable_weights)
    optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, model.trainable_weights))
    
    optimizer.set_weights(opt_weights)
    
    for (opt_weight,grad_var) in zip(opt_weights, model.trainable_weights):
        print(opt_weight.shape, '\t\t\t\t\t', grad_var.shape)
    
            
        
    #load_folder = "resnet_t3000_v900_light_ep_(0, 400)_lr_1e-04"
    load_dir = os.path.join(os.getcwd(), load_folder)
    model.load_weights(os.path.join(load_dir,"model_epoch_0425_savept_3.789702_Unfrz_-16.hdf5" ))
    # model.layers[-1].trainable = True
        
    ###==============================LOADING MODEL TO TRAIN END (ALT)================================
#%%    
    ###===============================FINE TUNING UNFREEZE BEGIN===============
    
    # for layer in model.layers[-1].layers:
    #     if layer.name in ['top_14', 'top_15', 'top_16', 'top_out_1']:
    #         layer.trainable = False
    #     print(layer.trainable, ' name: ', layer.name)
    
    # load_folder = "resnet_t3000_v900_B_ep_200_lr_1e-03"
    # load_dir = os.path.join(os.getcwd(), load_folder)
    # model.load_weights(os.path.join(load_dir,"model_epoch_0199_list_2.8649452.hdf5" ))
    

    # for layer in model.layers:
    #     layer.trainable = True
    #     print(layer.trainable, ' name: ', layer.name)
    
    # for layer in model.layers[-1].layers:
    #     layer.trainable = True
    #     print(layer.trainable, ' name: ', layer.name)
    
    
   ###===============================FINE TUNING UNFREEZE END==================
   
#%%
    ###===============================RAW UNFREEZE BEGIN===============
    
    
    
 
    ###===============================RAW UNFREEZE END=================   
#%%
    model.summary(expand_nested = True)

    
    #model = fconv_locate_deep_v2_w_upsampling(shape)#model_loaded #model instance

    model_name = 'resnet_t3000_v900_fullcycle_8bit_test2'
    batch_size = 8
    epochs = (0,1200)

    lr = 0.0001 #WHEN FINE-TUNING, LOWER LEARNING RATE
    opt_dict = {'Adam': tf.keras.optimizers.Adam(learning_rate= lr ), 'Adam_clip': tf.keras.optimizers.Adam(lr= lr, clipnorm = 0.5 ), 'sgd':tf.keras.optimizers.SGD(lr=lr, decay= 0.75*lr, momentum=0.9, nesterov=True) }
    opt_index = 'Adam'
    opt = opt_dict[opt_index]
    print('optimizer = ', opt_index, ' ||  lr = ', lr)

    save_folder = model_name + '_ep_' +str(epochs)+"_lr_" + "{:.0e}".format(lr)
    save_dir = os.path.join(os.getcwd(), save_folder)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plot_model(model.layers[-1], os.path.join(save_dir, 'top_model.png'), show_shapes = True)
    plot_model(model, os.path.join(save_dir, 'model.png'), show_shapes = True)
    save_model_summary(save_dir, model.layers[-1])
    save_model_summary(save_dir, model)
    print('------------------------------')
    print('')
    print(model_name)

#%%
    if modelFlag =='swin':
        image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    elif modelFlag =='resnet':
        image_processor = tf.keras.applications.resnet50.preprocess_input
    # unfreezeFlag = 0 # 1
#%%
    train_loss_tracker, test_loss_tracker = TrainAndTestV2(wDataObjectList, wValidDataObjectList, model, modelFlag, opt, batch_size, image_processor, epochs, save_dir, unfreezeFlag, iNorm)
    fig = plot_test_train(train_loss_tracker[0], test_loss_tracker[0])
    fig.savefig(os.path.join(save_dir, 'train_test_plot_loss1.png'))
    fig = plot_test_train(train_loss_tracker[1], test_loss_tracker[1])
    fig.savefig(os.path.join(save_dir, 'train_test_plot_loss2.png'))
    #fig = plot_test_train(train_loss_tracker[2], test_loss_tracker[2])
    #fig.savefig(os.path.join(save_dir, 'train_test_plot_loss2.png'))
    
#%% 
    with open(os.path.join(save_dir, 'train_test_values.csv'), 'w', newline ='') as file:
        writer = csv.writer(file)
        min_test_loss = np.inf
        epoch_no = 0
        for tr1, tr2, t1, t2 in zip(train_loss_tracker[0], train_loss_tracker[1],  test_loss_tracker[0], test_loss_tracker[1]):
            t_tot = t1
            if t_tot<min_test_loss:
                min_test_loss = t_tot
            row = ['train L: ', tr1, ' |A: ', tr2,  ' ||test L: ', t1,' |A: ', t2,  ' ||epoch: ', epoch_no, ' ||min_t_L: ', min_test_loss]
            writer.writerow(row)
            epoch_no+=1
#%%

    iTestSrcDir = "data4k\\test_real"
    iTestSrcPath = os.path.join(ROOT_DIR, iTestSrcDir)
    wTestDataObjectList = loadDataFilesAsObjects(iTestSrcPath)
    wTestBatchGen = generate_batch(wTestDataObjectList, 8)
    
    
#%%
    train = False
    for batch in wTestBatchGen:
        X_train = getImageListFromBatch(batch, 255.)
        map_lists = getMapListsFromBatch(batch)
        
        if modelFlag == 'swin':
            X_train = image_processor(np.array(X_train), return_tensors="tf")['pixel_values']
        elif modelFlag =='resnet':
            X_train = image_processor(np.array(X_train, dtype = np.float32)[...,::-1])
        
        y_pred = model(X_train, training = train)
        y_pred0, y_pred1 = y_pred[0], y_pred[1]
        
        show_batch(flat_map_list_v2(map_lists[0]))
        #show_batch(flat_map_list_v2(act_aug_list1))
        show_batch(list(y_pred0[...,0].numpy()[...,None]))
        show_batch(flat_map_list_v2(map_lists[1]))
        #show_batch(flat_map_list_v2(act_aug_list2))
        show_batch(list(y_pred1[...,0].numpy()[...,None]))
        
        
    #TrainEpochV3(model, modelFlag, wTestDataObjectList[1].getDim(), None, wTestBatchGen, image_processor, 0, None, False)

#%%
'''    
    # model_loaded = fconv_locate_deep(shape)
    # load_path = os.path.join(ROOT_DIR, 'fconv_locate_single_point_dist_loss_epochs_1500', 'model_epoch_1488_min_val.hdf5')
    # model_loaded.load_weights(load_path)
    # model = model_loaded
    metrics = ['loss1', 'loss2']
    # for XY, name in zip([XY_train, XY_valid, XY_test], ['train', 'valid', 'test']):
    #     # test_loss_tracker, test_acc_tracker =  train_epoch_1st_cycle(XY, model, dim, None, batch_size, 0, None, ROOT_DIR, folder_0, save_dir, 0, cv.IMREAD_COLOR, False)
    #     #=============CLASSIFICATION================
    #     # test_loss_eval, test_acc_eval =  train_epoch_1st_cycle(XY, model, dim, None, batch_size, 0, None, ROOT_DIR, folder_0, save_dir, 0, cv.IMREAD_COLOR, False)
    #     #=============DETECTION=====================
    #     test_loss_eval, test_acc_eval =  train_epoch_1st_cycle_mod_3(XY, model, dim, None, batch_size, 0, None, ROOT_DIR, folder_0, save_dir, 0, cv.IMREAD_COLOR, False)
    #     print(name + metrics[0], test_loss_eval, name + metrics[1], test_acc_eval)

    
    with open(os.path.join(save_dir, 'train_val_test_scores.csv'), 'w' ) as file:
        writer = csv.writer(file)
        for XY, name in zip([X_train, X_valid, X_test], ['train', 'valid', 'test']):
            #=============CLASSIFICATION================
            #test_loss_eval, test_acc_eval =  train_epoch_1st_cycle(XY, model, dim, None, batch_size, 0, None, ROOT_DIR, folder_0, save_dir, 0, cv.IMREAD_COLOR, False)
            #=============DETECTION=====================
            test_loss_eval, test_acc_eval =  train_epoch_NEW(XY, model, dim, None, batch_size, 0, None, False)
            writer.writerow([name + ' loss:', test_loss_eval, name + ' acc:', test_acc_eval])
            print(name + metrics[0], test_loss_eval, name + metrics[1], test_acc_eval)
            

    
'''
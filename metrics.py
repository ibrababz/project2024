# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 08:27:55 2022

@author: i_bab
"""
import file
import cv2 as cv

import os
from pathlib import Path
import numpy as np
from dataLoad import image_list_from_batch, generate_batch, adjust_number, show_batch
from dataLoad import generate_data_list, process_batch_3D, flat_map_list_v2, flatten_map_v2, name_list_from_batch
from augment_utils import augments
import time

import tensorflow as tf 


from tensorflow.keras import backend as B  

from tensorflow.keras.applications import ResNet50
from models import make_top_model_v2
from sklearn.model_selection import train_test_split

from loss_functions import tensor_pos_neg_loss
from loss_functions import act_list_3D, act_3D

def threshold_list(img_list, thresh_val, max_val, flag = cv.THRESH_TOZERO):
    # thresh_list = []
    # for im in img_list:
    #     _,thresh = cv.threshold(im, thresh_val, max_val, flag)
    #     thresh_list.append(thresh[...,None])
    # return thresh_list
    return [cv.threshold(im, thresh_val, max_val, flag)[1][...,None] for im in img_list]

def get_cntrs_list(img_list, ret_flag = cv.RETR_TREE, app_flag = cv.CHAIN_APPROX_SIMPLE, thresh_flag = cv.THRESH_BINARY):
    # cntrs_list = []
    # for im in img_list:
    #     _,thresh = cv.threshold(im, 0.01, 1., thresh_flag)
    #     contours, hierarchy = cv.findContours(np.uint8(thresh), ret_flag, app_flag)
    #     cntrs_list.append(contours)
    # return cntrs_list
    return [cv.findContours(np.uint8(cv.threshold(im, 0.01, 1., thresh_flag)[1]), ret_flag, app_flag)[0] for im in img_list] 

def get_contours(im, ret_flag = cv.RETR_TREE, app_flag = cv.CHAIN_APPROX_SIMPLE, thresh_flag = cv.THRESH_BINARY):
        thresh = cv.threshold(im, 0.01, 1., thresh_flag)[1]
        contours, hierarchy = cv.findContours(np.uint8(thresh), ret_flag, app_flag)
        return contours
        
def refine_thresh(pred, thresh):
    #print('1', pred.shape)
    #print('2',thresh.shape)
    thresh = tf.where(thresh > 0., 1., 0.)
    #print('3',thresh.shape)
    thresh = cv.dilate(np.float32(thresh[...,0]), np.ones((3,3)))[...,None]
    #print('4',thresh.shape)
    
    return pred*thresh
    
def refine_thresh_list(pred_list, thresh_list):
    # new_list = []from
    # for pred, thresh in zip(pred_list, thresh_list):
    #     new_list.append(refine_thresh(pred, thresh))
    # return new_list
    return [refine_thresh(pred, thresh) for pred, thresh in zip(pred_list, thresh_list)]
    
def blend_pred_truth(pred, truth, alpha =0.5, beta =0.5):
    
    blend = alpha*pred + beta*truth
    return blend/np.max(blend)

def add_clr_im(im1, im2):
    if im1.dtype == im2.dtype or (im1.dtype in ['float32', 'float64'] and im2.dtype in ['float32', 'float64']):
        if im1.dtype in ['float32', 'float64']:
            clip_value = 1.
        else:
            clip_value = 255
        im3 = np.clip(im1+im2, 0, clip_value)
        return im3
    else:
        print("NOT SAME TYPE")
        
def add_clr_im_list(im_list1, im_list2):
    add_list = []
    for im1, im2 in zip(im_list1, im_list2):
        add_list.append(add_clr_im(im1, im2))
    return add_list
    
def cmb_3_images(im1, im2, mask):
    hm, wm, cm = mask.shape
    #mask = np.reshape(mask, (hm*wm, cm))
    new = np.where(np.sum(mask, axis = 2)[...,None] !=0 , im2, im1 )
    return new

def zero_clr_channel(im, channel):
    for c in channel:
        im[..., c] = im[..., c]*0
    return im

def zero_clr_channel_list(im_list, channel):
    new_list = []
    
    for im in im_list:
        new_list.append(zero_clr_channel(im, channel))
    return new_list

def cmb_3_images_list(im1_list, im2_list, mask_list):
    cmb_list = [] 
    for im1, im2, mask in zip(im1_list, im2_list, mask_list):
        cmb_list.append(cmb_3_images(im1, im2, mask))
        
    return cmb_list
    
def blend_pred_truth_list(pred_list, truth_list, alpha = 0.5, beta = 0.5):
    blend_list = [] 
    for pred, truth in zip(pred_list, truth_list):
        
        blend_list.append(cv.normalize(blend_pred_truth(pred,truth, alpha, beta), None, 255, 0, cv.NORM_MINMAX, cv.CV_8U))
        
    return blend_list


    
def cmb_pred_truth(pred, truth):
    #pred = tf.where(pred > 0., 1.,0.)
    #truth = tf.where(truth > 0., 1.,0.)
    
    truth = truth*(1-tf.where(pred > 0., 1.,0.))
    print(truth.shape)
    print(pred.shape)
    cmb= np.concatenate([np.zeros(truth.shape), truth, pred], 2)
    print(cmb.shape)
    
    return cmb

def cmb_pred_truth_list(pred_list, truth_list):
    cmb_list = [] 
    for pred, truth in zip(pred_list, truth_list):
        cmb_list.append(cmb_pred_truth(pred,truth))
        
    return cmb_list

def resize_list(pred_list, size, interpolation = cv.INTER_NEAREST):
    new_list = []
    for pred in pred_list:
        if len(pred.shape)>2:
            new_list.append(cv.resize(pred, size, interpolation = interpolation)[...,None])
        else:
            new_list.append(cv.resize(pred, size, interpolation = interpolation))
    return new_list
        
def cvtColor_list(im_list, code = cv.COLOR_GRAY2BGR):
    new_list = []
    for im in im_list:
        new_list.append(cv.cvtColor(im, code))
    return new_list

def colorMap_list(im_list, code = cv.COLORMAP_VIRIDIS):
    new_list = []
    for im in im_list:
        if im.dtype == 'float32' or im.dtype == 'float64':
            im = np.uint8(im*255.0)
        im = cv.applyColorMap(im, code)
        # im = im/255.0
        # im = im/(np.max(im)+1e-15)
        new_list.append(im)
  
    return new_list

def draw_cntrs_list(img_list_copy, cntrs_list, cntrs_i = -1, colour = 1, thickness= 1, on_black = 0.):
    im_list_cntrs = []
    for im, cntrs in zip(img_list_copy, cntrs_list):
        im_list_cntrs.append(cv.drawContours(im*(1.-on_black), cntrs, cntrs_i, colour, thickness))
        #im_list_cntrs.append(cv.drawContours(im, cntrs, cntrs_i, colour, thickness))
    return im_list_cntrs

def  draw_cntrs_exp_list(img_list_copy, cntrs_list, colour = 1, thickness= -1):
    im_list_cntrs = []
    for im, cntrs in zip(img_list_copy, cntrs_list):
        layers = []
        for i, cnt in zip(range(len(cntrs)),cntrs):
            blank = np.zeros(im.shape, dtype = im.dtype)
            layers.append(cv.drawContours(blank, cntrs, i, colour, thickness))
        
        if len(layers)>0:
            expanded = np.concatenate(layers, axis = -1)
        else:
            expanded = np.zeros(im.shape, dtype = im.dtype)
        
        im_list_cntrs.append(expanded)
    return im_list_cntrs


def  draw_cntrs_exp(im, cntrs, colour = 1, thickness= -1):
    layers = []
    for i, cnt in zip(range(len(cntrs)),cntrs):
        blank = np.zeros(im.shape, dtype = im.dtype)
        layers.append(cv.drawContours(blank, cntrs, i, colour, thickness))
    
    if len(layers)>0:
        expanded = np.concatenate(layers, axis = -1)
    else:
        expanded = np.zeros(im.shape, dtype = im.dtype)
    return expanded

def expand_pred_list(pred_batch, pred_drawn_expanded_list):
    # pred_exp_list = []      
    # for pred, exp in zip(pred_batch, pred_drawn_expanded_list):
    #     pred_exp = pred*exp
    #     pred_exp_list.append(pred_exp)
 
    # return pred_exp_list
    return [pred*exp for pred, exp in zip(pred_batch, pred_drawn_expanded_list)]

def expand_pred(pred, exp):
    pred_exp = pred*exp
    return pred_exp
    

def act_from_pred_list(pred_batch):
    pred_cntrs_batch = get_cntrs_list(pred_batch)
    pred_drawn_expanded_list = draw_cntrs_exp_list(pred_batch.copy(), pred_cntrs_batch, thickness = -1)
    pred_exp_list = expand_pred_list(pred_batch, pred_drawn_expanded_list)
    pred_act_list = act_list_3D(pred_exp_list)
    flat_pred_act_list = flat_map_list_v2(pred_act_list)
    return flat_pred_act_list

def act_from_pred(pred):
    pred_cntrs = get_contours(pred)
    exp = draw_cntrs_exp(pred.copy(), pred_cntrs, thickness=-1 )
    pred_exp = expand_pred(pred, exp)
    pred_act = act_3D(pred_exp)
    flat_pred_act = flatten_map_v2(pred_act)
    return flat_pred_act
        

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

    



def train_epoch_NEW(XY_train, model, dim, opt, batch_size, epoch, seq, train = True):
    i = epoch
    
    if train:
        seed = i+1
        seq.seed_(seed) 
    else:
        seed = 0
     
    batch_generator_01 = generate_batch(XY_train.copy(), batch_size, seed)
    batch_loss_tracker = []
    batch_acc_tracker = []
    j = 0
    weight = 0.7
    for batch in batch_generator_01:
    
        X_train = image_list_from_batch(batch, dim = None)
        name_list = name_list_from_batch(batch)
      
        if train:
            
            with tf.GradientTape() as tape:
                  
                temp = seq.deepcopy()
                
                X_train_aug = temp.deepcopy()(images = X_train)
                X_show = X_train_aug.copy()
                X_train_aug = tf.keras.applications.resnet50.preprocess_input(np.array(X_train_aug, dtype = np.float32)[...,::-1]) 
                # [...,::-1]index to RGB so preprocess can reconvert to BGR and Zero center the data and don't divide by 255.0
                y_pred = model(X_train_aug, training = train)
                
                y_pred0, y_pred1 = y_pred[0], y_pred[1]
                dim_grid1, dim_grid2 = y_pred0[0,...,0].shape, y_pred1[0,...,0].shape
                dim_grid1, dim_grid2 = (dim_grid1[0], dim_grid1[1]), (dim_grid2[0], dim_grid2[1])
                #print('dim_grid1 = ', dim_grid1, 'dim_grid2 = ', dim_grid2)
                # flat_arr1, weight_arr1, _ = process_batch(batch, dim, dim_grid1, temp.deepcopy())
                # flat_arr2, weight_arr2, seq = process_batch(batch, dim, dim_grid2, temp.deepcopy())
                # loss1 =  loss_from_map_tensor(y_pred0[...,0], flat_arr1, weight_arr1)
                # loss2 =  loss_from_map_tensor(y_pred1[...,0], flat_arr2, weight_arr2)
                map_aug_list1, weight_list1, _ = process_batch_3D(batch, dim, dim_grid1, temp.deepcopy())
                map_aug_list2, weight_list2, _ = process_batch_3D(batch, dim, dim_grid2, temp.deepcopy())
                
                act_aug_list1 = act_list_3D(map_aug_list1)
                act_aug_list2 = act_list_3D(map_aug_list2)
                
                pos1,neg1 = tensor_pos_neg_loss(y_pred0[...,0], map_aug_list1, act_aug_list1)
                pos2,neg2 = tensor_pos_neg_loss(y_pred1[...,0], map_aug_list2, act_aug_list2)
                
                loss1 = pos1 + neg1
                loss2 = pos2 + neg2
                loss1_update = pos1 + weight*neg1 + B.abs(B.sum([pos1, -neg1]))/B.sum([pos1, neg1])
                loss2_update = pos2 + weight*neg2 + B.abs(B.sum([pos2, -neg2]))/B.sum([pos2, neg2])
                '''pos_loss, neg_loss, dist_loss, cellOb_list = map_loss_from_batch_koi(y_pred[0], y_batch_koi, pt_index)
                #loss_value = pos_loss + weight*neg_loss + B.abs(B.sum([pos_loss, -neg_loss]))/B.sum([pos_loss, neg_loss]) + dist_loss #MAKE SURE YOU CHANGE FOR BOTH VAL AND TRAIN
                loss_value = dist_loss + pos_loss + weight*neg_loss + B.abs(B.sum([pos_loss, -neg_loss]))/B.sum([pos_loss, neg_loss])
                '''
                loss_value = loss1 + loss2
                acc_value = loss2#tf.constant(0) loss2
                
                if j == 0 and i%10 == 0:
                    show_batch(list(X_train), name_list)
                    map_list2, _, _ = process_batch_3D(batch, dim, dim_grid2, None)
                    show_batch(flat_map_list_v2(map_list2), name_list)
                    #show_batch(list(X_train_aug), name_list)
                    
                    show_batch(X_show, name_list)
                    show_batch(flat_map_list_v2(map_aug_list1), name_list)
                    
                    show_batch(list(y_pred0[...,0].numpy()[...,None]))
                    
                    show_batch(flat_map_list_v2(map_aug_list2), name_list)
                    show_batch(list(y_pred1[...,0].numpy()[...,None]))
            
            grads = tape.gradient([loss1_update, loss2_update], model.trainable_weights)
            opt.apply_gradients((grad, var) for (grad, var) in zip(grads, model.trainable_weights) if grad is not None)
            
        else:
            X_train = tf.keras.applications.resnet50.preprocess_input(np.array(X_train, dtype = np.float32)[...,::-1])
            # [...,::-1]index to RGB so preprocess can reconvert to BGR and Zero center the data and don't divide by 255.0
            y_pred = model(X_train, training = train)
            
            y_pred0, y_pred1 = y_pred[0], y_pred[1]
            dim_grid1, dim_grid2 = y_pred0[0,...,0].shape, y_pred1[0,...,0].shape
            dim_grid1, dim_grid2 = (dim_grid1[0], dim_grid1[1]), (dim_grid2[0], dim_grid2[1])

            # flat_arr1, weight_arr1, _ = process_batch(batch, dim, dim_grid1)
            # flat_arr2, weight_arr2, _ = process_batch(batch, dim, dim_grid2)
            
            # loss1 =  loss_from_map_tensor(y_pred0[...,0], flat_arr1, weight_arr1)
            # loss2 =  loss_from_map_tensor(y_pred1[...,0], flat_arr2, weight_arr2)

            # pos_loss, neg_loss, dist_loss, cellOb_list = map_loss_from_batch_koi(y_pred[0], y_batch_koi, pt_index)
            # #loss_value = pos_loss + weight*neg_loss + 0.5*B.abs(B.sum([pos_loss, -neg_loss])) #MAKE SURE YOU CHANGE FOR BOTH VAL AND TRAIN
            map_list1, weight_list1, _ = process_batch_3D(batch, dim, dim_grid1)
            map_list2, weight_list2, _ = process_batch_3D(batch, dim, dim_grid2)
            
            act_list1 = act_list_3D(map_list1)
            act_list2 = act_list_3D(map_list2)
            
            pos1,neg1 = tensor_pos_neg_loss(y_pred0[...,0], map_list1, act_list1)
            pos2,neg2 = tensor_pos_neg_loss(y_pred1[...,0], map_list2, act_list2)
            
            loss1 = pos1 + neg1
            loss2 = pos2 + neg2
            
            loss_value = loss1 +loss2
            acc_value = loss2#tf.constant(0) loss2
            
        batch_loss_tracker.append(loss_value.numpy())
        batch_acc_tracker.append(acc_value)
        # if train and j==0:
        #     print(grads)
        j+=1
    #print('dim_grid1 = ', dim_grid1, 'dim_grid2 = ', dim_grid2)
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

def train_and_test_NEW(XY_train, XY_valid, model, dim, opt, batch_size, epochs, save_dir):

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
    number_of_epochs_to_break_val = 1000
    #train_flag, valid_flag = 0, 0
    five_saves_ago = 15
    start_saving = 15
    save_freq = 5
    save_names = []
    for i in range(epochs):
        #print("\nStart of epoch %d" % (i,))
        
        loss_per_epoch, acc_per_epoch= train_epoch_NEW(XY_train, model, dim, opt, batch_size, i, augments(), True)
        # loss_per_epoch, acc_per_epoch = train_epoch_NEW(XY_train, model, dim, None, batch_size, i, None, False)
        #compute loss on unaugmented training data for metrics, but train using augmented training data
        
        train_loss_tracker.append(loss_per_epoch)
        train_acc_tracker.append(acc_per_epoch)


        valid_loss_per_epoch, valid_acc_per_epoch = train_epoch_NEW(XY_valid, model, dim, None,  batch_size, i,None, False)

   
        valid_loss_tracker.append(valid_loss_per_epoch)
        valid_acc_tracker.append(valid_acc_per_epoch)

        sum_loss_per_epoch = np.sum([loss_per_epoch])
        sum_val_loss_per_epoch = np.sum([valid_loss_per_epoch])
        
        sum_acc_per_epoch = np.sum([acc_per_epoch])
        sum_val_acc_per_epoch = np.sum([valid_acc_per_epoch])
        
        min_train_loss, min_train_loss_counter = check_train_loss(min_train_loss_counter, sum_loss_per_epoch, min_train_loss)
        min_val_loss, min_val_loss_counter, old_name = check_val_loss(min_val_loss_counter, sum_val_loss_per_epoch, min_val_loss, old_name, i)
                   

        if not i%save_freq and i>=start_saving or i == epochs -1:
            f_name = save_and_print(i, model, save_dir, '', 0)
            save_names.append(f_name)
            if i - five_saves_ago >= 5*save_freq:
                remove_train_name = save_names.pop(0)
                remove_old_model(remove_train_name)
                five_saves_ago+=5
        
        B.set_value(opt.iterations, i)
        #B.set_value(opt.lr, opt._decayed_lr(tf.float32).numpy())
        #print('tr L:', loss_per_epoch,'A:',loss1_per_epoch,'+ ',loss2_per_epoch ,'  val L:', valid_loss_per_epoch, '+',valid_loss1_per_epoch,'+ ',valid_loss2_per_epoch, ' epoch: ', i, ' min_val: ', min_val_loss)
        print('tr L:', loss_per_epoch,'A:',acc_per_epoch, 'val L:', valid_loss_per_epoch, 'A:',valid_acc_per_epoch, 'epoch:', i, 'min_val:', min_val_loss)
        
        
        if check_for_break(min_train_loss_counter, number_of_epochs_to_break_train, min_val_loss_counter, number_of_epochs_to_break_val):
            break
        
        #break #FOR DEBUGGING
    
    return [train_loss_tracker, train_acc_tracker], [valid_loss_tracker,  valid_acc_tracker]



def save_and_print(i, model, save_dir, add_flag = '', verbose = 1):
    if verbose:
        print('-----saving weights------')
    number = adjust_number(i)
    full_name = os.path.join(save_dir, 'model_epoch_'+number+add_flag+'.hdf5')
    model.save_weights(full_name)
    return full_name

class file_writing():
    def __init__(self, file):
        self.file = file
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

def find_cent_on_im(contours):
    centers_on_image = []
    for c in contours:
    
        M = cv.moments(c)
        if  M["m00"] != 0: 
            cX = round(M["m10"] / M["m00"])
            cY = round(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
  
        centers_on_image.append([cX, cY])

    return centers_on_image

def find_cent_list(contours_on_image_list):
    return [find_cent_on_im(cntrs) for cntrs in contours_on_image_list]
    # centers_on_image_list = []

    # for cntrs in contours_on_image_list:
    #     centers_on_image = find_cent_on_im(cntrs)
    #     centers_on_image_list.append(centers_on_image) 

    # return centers_on_image_list

def draw_cent_on_im(im, centers_on_image, color = (1.,1.,1.)):

    if len(im.shape)<3:
        color = color[0]
    elif len(im.shape) == 3:
        if im.shape[2] == 1:
            color = color[0]
    im_copy = im.copy()        
    for c in centers_on_image:
        cX, cY = c[0],  c[1]
        
        im_copy[cY, cX] = color
    
    return im_copy

def draw_cent_on_im_list(centers_on_im_list, im_list = None, shape = None):
    drawn_cent_list = [] 
    for i in range(len(centers_on_im_list)):
        if im_list is not None:
            im = im_list[i].copy()
        else:
            im = np.zeros(shape)
            
        drawn = draw_cent_on_im(im, centers_on_im_list[i])
        drawn_cent_list.append(drawn)
    return drawn_cent_list

def draw_cent_on_im_v2(centers_on_image, im=None, shape = None, color = (1.,1.,1.)):
    if im is not None:
        im = im.copy()
    else:
        im = np.zeros(shape)
   
    if len(im.shape)<3:
        color = color[0]
    elif len(im.shape) == 3:
        if im.shape[2] == 1:
            color = color[0]
    im_copy = im.copy()        
    for c in centers_on_image:
        cX, cY = c[0],  c[1]
        
        im_copy[cY, cX] = color
    
    return im_copy

def TFmetrics(truth_map_i, pred_map_i):
    TPi = np.logical_and(truth_map_i, pred_map_i)*1.
    
    FPi = np.greater(pred_map_i,0.)*1. - TPi
    
    TNi = np.logical_and(np.logical_not(truth_map_i), np.logical_not(pred_map_i))*1.
    
    FNi = np.logical_not(pred_map_i)*1. - TNi
    
    
    TPi_no = np.sum(TPi)
    FPi_no = np.sum(FPi)
    
    TNi_no = np.sum(TNi)
    FNi_no = np.sum(FNi)
    
    return TPi_no, FPi_no, TNi_no, FNi_no

def Metrics_from_TF(TPi_no, FPi_no, TNi_no, FNi_no):
    if TPi_no == 0.:
        Prec = 0.
        Reca = 0.
        F1 = 0.
        MIoU = 0.
    else:
        Prec = TPi_no/(TPi_no + FPi_no)
        Reca = TPi_no/(TPi_no + FNi_no)
        F1 = 2.*Prec*Reca/(Prec+Reca)
        MIoU = TPi_no/(TPi_no + FPi_no + FNi_no)

    
    return Prec, Reca, F1, MIoU
def TF_Metrics_from_batch(truth_batch1, pred_batch1):
    TPi_list = []
    FPi_list = []
    TNi_list = []
    FNi_list = []

    for act_i, coi_i in zip(truth_batch1, pred_batch1):
        
        TPi_no, FPi_no, TNi_no, FNi_no = TFmetrics(act_i,coi_i) 
        
        TPi_list.append(TPi_no)
        FPi_list.append(FPi_no)
        TNi_list.append(TNi_no)
        FNi_list.append(FNi_no)

    return TPi_list, FPi_list, TNi_list, FNi_list


def Metrics_from_TF_batch(TPi_list, FPi_list, TNi_list, FNi_list):
    Prec_list = []
    Reca_list = []
    F1_list = []
    MIoU_list = []
    
    for TPi_no, FPi_no, TNi_no, FNi_no in zip(TPi_list, FPi_list, TNi_list, FNi_list):
        Prec, Reca, F1, MIoU = Metrics_from_TF(TPi_no, FPi_no, TNi_no, FNi_no)
        Prec_list.append(np.round(Prec, 2))
        Reca_list.append(np.round(Reca, 2))
        F1_list.append(np.round(F1, 2))
        MIoU_list.append(np.round(MIoU, 2))

    
    return Prec_list, Reca_list, F1_list, MIoU_list
if __name__ == '__main__':

    physical_devices = tf.config.list_physical_devices('GPU')

    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
 
    ROOT_DIR = Path(file.ROOT_DIR).parent

    folder = '1_224x224'
    data_path = os.path.join(ROOT_DIR, folder)
    folder_points = '1'
    file_points = "plant_centers_sifted_FINAL.json"
    points_path = os.path.join(ROOT_DIR, folder_points, file_points)
    Wo, Ho = 4056, 3040
    W, H = 448, 448
    shape = (H,W,3)
    dim = (W,H)
    scale_x, scale_y = W/Wo, H/Ho
    oimg_list = generate_data_list(data_path, points_path, scale_x, scale_y)
    X_train, X_test, _, _ =  train_test_split(oimg_list, oimg_list, test_size = 0.2, random_state = 42)
    X_test, X_valid, _, _ = train_test_split(X_test, X_test, test_size = 0.5, random_state = 42)
    
    project_folder = "project2024"
    swin_folder = "archive" 
    swin_pretrn_dir = os.path.join(ROOT_DIR, project_folder, swin_folder)
    swin_pretrn_file = "saved_model.pb"
    swin_pretrn_path = os.path.join(swin_pretrn_dir, swin_pretrn_file)
    base_model = tf.keras.models.load_model(swin_pretrn_dir)
    dummy_inputs = tf.ones((2, 224, 224, 3))
    base_output = base_model(dummy_inputs)
    print(base_model.summary(expand_nested=True))
    print(base_output)
    base_input = tf.keras.Input((224,224,3))
    


    from swin_transformers_tf_main.swins import SwinTransformer
    
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(4, 8, 16, 32),
    )
     
    swin_tiny_patch4_window7_224 = SwinTransformer(
        name="swin_tiny_patch4_window7_224", **cfg
    )
    print("Model instantiated, attempting predictions...")
    random_tensor = tf.random.normal((2, 224, 224, 3))
    outputs = swin_tiny_patch4_window7_224(random_tensor, training=False)
    
    swin_tiny_patch4_window7_224.summary(expand_nested = True)
    
    print(outputs.shape)
    
    print(swin_tiny_patch4_window7_224.count_params() / 1e6)
        
        
    base_model.output
    base_model = ResNet50(include_top=False, input_shape=shape,pooling='None',weights='imagenet')
    index =142
    base_model.trainable = False

    inputs = tf.keras.Input(shape=shape, name = "input_1")
    base_model2 = tf.keras.models.Model([base_model.inputs], [base_model.output, base_model.layers[index].output], name = 'base_model2')
    base_model2.trainable = False
    
    top_model = make_top_model_v2((int(H/32),int(W/32),2048), (int(H/16),int(W/16),1024))

    model = tf.keras.models.Model(base_model2.inputs, top_model(base_model2.output))
    #model.summary()
    
    #model = fconv_locate_deep_v2_w_upsampling(shape)#model_loaded #model instance

    model_name = 'LOADED'
    batch_size = 8
    epochs = 5000
    lr = 0.0001
    opt_dict = {'Adam': tf.keras.optimizers.Adam(lr= lr ), 'Adam_clip': tf.keras.optimizers.Adam(lr= lr, clipnorm = 0.5 ), 'sgd':tf.keras.optimizers.SGD(lr=lr, decay= 0.75*lr, momentum=0.9, nesterov=True) }
    opt_index = 'Adam'
    opt = opt_dict[opt_index]
    #print('optimizer = ', opt_index, ' ||  lr = ', lr)

    load_folder = "TEST_12_neg_5_dist_2p5_lo_res_base_old_map_frozen_lo_res_finetune_unfreeze_all_epochs_0200"
    load_dir = os.path.join(os.getcwd(), load_folder)
    #plot_model(model, show_shapes = True)
    
    print('------------------------------')
    print('')
    print(model_name)
    
    # for layer in model.layers:
    #     print(layer.trainable, ' name: ', layer.name)

    # for layer in model.layers[-1].layers:
    #     if layer.name in ['top_14', 'top_15', 'top_16', 'top_out_1']:
    #         layer.trainable = False
    #     print(layer.trainable, ' name: ', layer.name)
        
        
    model.load_weights(os.path.join(load_dir,"model_epoch_0116_min_val_USED.hdf5" ))
    


    time.sleep(1)
    

    thresh_value_list = np.arange(0.5,0.91,0.05)
    for thresh_value in thresh_value_list:
        batch_gen_test = generate_batch(X_test.copy(), 8, 0)
        save_folder = 'test_images'
        save_im_dir = os.path.join(load_dir, save_folder)
    
    
        print(thresh_value)
        TPi_list = []
        FPi_list = []
        TNi_list = []
        FNi_list = []
    
        Prec_list = []
        Reca_list = []
        F1_list = []
        MIoU_list = []
        for batch in batch_gen_test:
            X_test_batch = image_list_from_batch(batch, dim = None)
            X_test_names = name_list_from_batch(batch)
            X_test_batch_prep = tf.keras.applications.resnet50.preprocess_input(np.array(X_test_batch, dtype = np.float32)[...,::-1])
            y_pred = model(np.array(X_test_batch_prep))
            y_pred0, y_pred1 = y_pred[0], y_pred[1]
            y_test = y_pred0
            truth_map_list, _, _ = process_batch_3D(batch, dim, tuple(y_test[0,...,0].shape))
            pred_list = list(y_test[...,0].numpy()[...,None])
            
            act_list1 = act_list_3D(truth_map_list)
            flat_act_list1 = flat_map_list_v2(act_list1)
    
            #show_batch(blue_act_list1)
            flat_map_list1 = flat_map_list_v2(truth_map_list)
    
            pred_list_thresh = threshold_list(pred_list.copy(), thresh_value, 1.0)
                    
            pred_list_thresh = refine_thresh_list(pred_list.copy(), pred_list_thresh.copy())
            
            #=======================FOR METRICS START==================================
            truth_batch1 = flat_map_list1.copy()
            pred_batch1 = pred_list_thresh.copy()
            truth_act_batch1 = flat_act_list1.copy()
            
            #=======================FOR METRICS END====================================
            
            #=======================RESIZING START=============================
            
            flat_map_list1 = resize_list(flat_map_list1, (H,W))
            pred_list = resize_list(pred_list,(H,W))
            pred_list_thresh = resize_list(pred_list_thresh,(H,W))
            flat_act_list1 = resize_list(flat_act_list1, (H,W))
            
            #=======================RESIZING END=============================
            
            
            #show_batch(flat_act_list1)
            clr_act_list1 = colorMap_list(flat_act_list1.copy(), cv.COLORMAP_HOT)
            #show_batch(clr_act_list1)
            blue_act_list1 = zero_clr_channel_list(clr_act_list1.copy(), [1,2])
    

            #show_batch(colorMap_list(pred_list_thresh.copy(), cv.COLORMAP_HOT))
            
            
            
            #flat_map_list1_thresh = threshold_list(flat_map_list1, 0.0, 1.0) 
            #show_batch(pred_list_thresh)
            shape_grid2= pred_list[0].shape
            h_grid = shape_grid2[0]
    
            
            pred_thresh_clr = colorMap_list(pred_list_thresh, cv.COLORMAP_HOT)
            truth_clr = colorMap_list(flat_map_list1, cv.COLORMAP_OCEAN)
            blend_list = blend_pred_truth_list(pred_thresh_clr, truth_clr, 0.4, 0.6)
            blend_with_original = blend_pred_truth_list(X_test_batch, blend_list, 0.4, 0.6)
            cmb_blend_lists = cmb_3_images_list(X_test_batch, blend_with_original, blend_list)
            
            
            
            #show_batch(blend_with_original)
            #show_batch(cmb_blend_lists)
            
            blend_with_original_pred = blend_pred_truth_list(X_test_batch, pred_thresh_clr, 0.5, 0.5)
            cmb_blend_lists_pred = cmb_3_images_list(X_test_batch, blend_with_original_pred, pred_thresh_clr)
            cmb_pred_act_lists = cmb_3_images_list(cmb_blend_lists_pred, blue_act_list1, blue_act_list1)
            
            #show_batch(blend_with_original_pred)
            #show_batch(cmb_blend_lists_pred)
            
            
         
            #==============================METRICS START======================================================
            pred_cntrs_batch1 = get_cntrs_list(pred_batch1)
            pred_drawn_list1 = draw_cntrs_list(pred_batch1.copy(), pred_cntrs_batch1, thickness = -1, on_black = 1.)
            pred_drawn_expanded_list1 = draw_cntrs_exp_list(pred_batch1.copy(), pred_cntrs_batch1, thickness = -1)
            centers_on_im_list1 = find_cent_list(pred_cntrs_batch1)
            drawn_cent_on_im_list1 = draw_cent_on_im_list(centers_on_im_list1, None, pred_batch1[0].shape)
            flat_pred_act_list1 = act_from_pred_list(pred_batch1)
            #==============================METRICS END======================================================
            drawn_cent_resize_list1 = resize_list(drawn_cent_on_im_list1, (H,W))
            pred_act_resize_list1 = resize_list(flat_pred_act_list1, (H,W))
            #===========================SAVE IMAGE START============================================
            # show_batch(X_test_batch)
            # save_images_from_list(X_test_batch, X_test_names, save_im_dir, '_00O', 0)
            # save_images_from_list(colorMap_list(flat_map_list1.copy(), cv.COLORMAP_OCEAN), X_test_names, save_im_dir, '_01T', 0)
            # save_images_from_list(colorMap_list(pred_list.copy(), cv.COLORMAP_HOT), X_test_names, save_im_dir, '_02P', 0)
            
            # blue_act_list1 = zero_clr_channel_list(colorMap_list(flat_act_list1.copy(), cv.COLORMAP_OCEAN), [1,2])
            # save_images_from_list(blue_act_list1, X_test_names, save_im_dir, '_01T_act', 0)
            
            # red_pred_cent_list1 = zero_clr_channel_list(colorMap_list(drawn_cent_resize_list1.copy(), cv.COLORMAP_HOT), [0,1])
            # save_images_from_list(red_pred_cent_list1, X_test_names, save_im_dir, '_02P_cent', 0)
            
            # green_pred_act_list1 = zero_clr_channel_list(colorMap_list(pred_act_resize_list1.copy(), cv.COLORMAP_HOT), [0,2])
            # save_images_from_list(green_pred_act_list1, X_test_names, save_im_dir, '_02P_act', 0)
           
            # save_images_from_list(colorMap_list(pred_list_thresh.copy(), cv.COLORMAP_HOT), X_test_names, save_im_dir, '_03P', 0)
            # save_images_from_list(blend_list, X_test_names, save_im_dir, '_04B', 0)
            # #save_images_from_list(blend_with_original, X_test_names, save_im_dir, '_05B', 0)
            # save_images_from_list(cmb_pred_act_lists, X_test_names, save_im_dir, '_05B', 0)
            
            # cmb_actcent_lists = add_clr_im_list(blue_act_list1, red_pred_cent_list1) 
            # save_images_from_list(cmb_actcent_lists, X_test_names, save_im_dir, '_06C_cent', 0)
            
            # cmb_actact_lists = add_clr_im_list(blue_act_list1, green_pred_act_list1) 
            # save_images_from_list(cmb_actact_lists, X_test_names, save_im_dir, '_06C_act', 0)
            # show_batch(pred_list_thresh)
            pred_cntrs_to_draw = get_cntrs_list(pred_list_thresh)
            pred_cntrs_drawn = draw_cntrs_list(pred_list.copy(), pred_cntrs_to_draw, -1, 1,4,1)
            # show_batch(pred_cntrs_drawn)
            pred_cntrs_drawn_red = zero_clr_channel_list(colorMap_list(pred_cntrs_drawn.copy(), cv.COLORMAP_HOT), [0,1])
            # show_batch(pred_cntrs_drawn)
            
            # save_images_from_list(pred_cntrs_drawn_red, X_test_names, save_im_dir, '_02P_contour', 0)
            
            #===========================SAVE IMAGE END==============================================
            
            #===========================SHOW START=================================================
            show_batch(colorMap_list(flat_map_list1.copy(), cv.COLORMAP_OCEAN))
            show_batch(colorMap_list(pred_list.copy(), cv.COLORMAP_HOT))
            show_batch(colorMap_list(pred_list_thresh.copy(), cv.COLORMAP_HOT))
            
            show_batch(blend_list)
            show_batch(cmb_pred_act_lists)
    
            # show_batch(truth_batch1)            
            # show_batch(truth_act_batch1) 
            # show_batch(pred_batch1)
            # show_batch(pred_drawn_list1)
            # show_batch(drawn_cent_on_im_list1)
            # show_batch(red_pred_cent_list1) 
            # show_batch(green_pred_act_list1)
            #===========================SHOW END=================================================
            
            
            # TPi_batch, FPi_batch, TNi_batch, FNi_batch = TF_Metrics_from_batch(truth_batch1, pred_batch1); prefix = 'map'
            # TPi_batch, FPi_batch, TNi_batch, FNi_batch = TF_Metrics_from_batch(truth_act_batch1, drawn_cent_on_im_list1); prefix = 'cent'
            TPi_batch, FPi_batch, TNi_batch, FNi_batch = TF_Metrics_from_batch(truth_act_batch1, flat_pred_act_list1); prefix = 'act'         
            Prec_batch, Reca_batch, F1_batch, MIoU_batch = Metrics_from_TF_batch(TPi_batch, FPi_batch, TNi_batch, FNi_batch)
            
            TPi_list.extend(TPi_batch)
            FPi_list.extend(FPi_batch)
            TNi_list.extend(TNi_batch)
            FNi_list.extend(FNi_batch)
    
            Prec_list.extend(Prec_batch)
            Reca_list.extend(Reca_batch)
            F1_list.extend(F1_batch)
            MIoU_list.extend(MIoU_batch)
            #==============================METRICS END======================================================
           
            # cntrs_list = get_cntrs_list(pred_list_thresh_resized)
            # truth_cntrs_list = get_cntrs_list(flat_map_list1_thresh_resized)
            #break
            # show_batch(colorMap_list(pred_list_thresh_resized, cv.COLORMAP_HOT))
            
            # cmb_list = draw_cntrs_list(colorMap_list(resize_list(flat_map_list1.copy(), (H,W)), cv.COLORMAP_OCEAN), cntrs_list, colour= (0., 0., 1.), thickness =5, on_black = 0.)
            # #cmb_list = draw_cntrs_list(cmb_list, truth_cntrs_list, colour= (0., 0., 1.), thickness =5)
            # show_batch(cmb_list)
            # #show_batch(draw_cntrs_list(pred_list_thresh, cntrs_list))
            #break
        Prec_mean = np.mean(Prec_list)
        Reca_mean = np.mean(Reca_list)
        F1_mean = np.mean(F1_list)
        MIoU_mean = np.mean(MIoU_list)
        
            
        # with open(os.path.join(load_dir, prefix + 'LO_RES_metrics_thresh_0p'+ str(thresh_value*100)[:2] + '.csv'), 'w' ) as file:
        #     writer = csv.writer(file)
            
        #     writer.writerow(['Precision', Prec_mean])
        #     writer.writerow(['Recall', Reca_mean])
        #     writer.writerow(['F1 score', F1_mean])
        #     writer.writerow(['MIoU', MIoU_mean])

    print(Prec_mean)
    print(Reca_mean)
    print(F1_mean)
    print(MIoU_mean)
    
    for truth_i, pred_i, act_i, pred_cent_i, pred_act_i in zip(truth_batch1, pred_batch1, truth_act_batch1, drawn_cent_on_im_list1, flat_pred_act_list1):
        TPi_no, FPi_no, TNi_no, FNi_no = TFmetrics(truth_i, pred_i)
        Prec, Reca, F1, MIoU = Metrics_from_TF(TPi_no, FPi_no, TNi_no, FNi_no)
        print('Map Metrics')
        print("Prec ", np.round(Prec*100,2), ", Reca ", np.round(Reca*100,2)
              , ", F1 ", np.round(F1*100,2), ", MIoU ", np.round(MIoU*100,2))
        
        TPi_no, FPi_no, TNi_no, FNi_no = TFmetrics(act_i, pred_cent_i)
        Prec, Reca, F1, MIoU = Metrics_from_TF(TPi_no, FPi_no, TNi_no, FNi_no)
        print('Centroid Metrics')
        print("Prec ", np.round(Prec*100,2), ", Reca ", np.round(Reca*100,2)
              , ", F1 ", np.round(F1*100,2), ", MIoU ", np.round(MIoU*100,2))
        
        TPi_no, FPi_no, TNi_no, FNi_no = TFmetrics(act_i, pred_act_i)
        Prec, Reca, F1, MIoU = Metrics_from_TF(TPi_no, FPi_no, TNi_no, FNi_no)
        print('Activation Metrics')
        print("Prec ", np.round(Prec*100,2), ", Reca ", np.round(Reca*100,2)
              , ", F1 ", np.round(F1*100,2), ", MIoU ", np.round(MIoU*100,2))
        print("======================================================")







'''
show_batch(pred_batch1)
show_batch(flat_pred_act_list1) 
show_batch(truth_act_batch1)
show_batch(drawn_cent_on_im_list1)
show_batch(truth_act_batch1)

print(TPi_batch)
print(FPi_batch)
print(TNi_batch)
print(TNi_batch)

print(Prec_batch)
print(Reca_batch)
print(F1_batch)
print(MIoU_batch)

pred_cntrs_batch1 = get_cntrs_list(pred_batch1)
pred_drawn_list1 = draw_cntrs_list(pred_batch1.copy(), pred_cntrs_batch1, on_black = 1.)

centers_on_image_list = []

for cntrs in pred_cntrs_batch1:
    centers_on_image = []
    for c in cntrs:

        M = cv.moments(c)
        cX_i = int(M["m10"] / M["m00"])
        cY_i = int(M["m01"] / M["m00"])
        
        cX_r = round(M["m10"] / M["m00"])
        cY_r = round(M["m01"] / M["m00"])
        
        #print("cX_i = ", cX_i," cY_i = ", cY_i, "|cX_r = ", cX_r," cY_r = ", cY_r)
        centers_on_image.append([cX_r, cY_r])
    centers_on_image_list.append(centers_on_image)   

centroid_drawn_list1 = []
pred_drawn_copy = pred_drawn_list1.copy()
for im_i, coim in zip(pred_drawn_copy, centers_on_image_list):
    im_i_copy = im_i.copy()
    for c in coim:
        cX, cY = c[0],  c[1]
        
        im_i_copy[cY, cX] = 1.
           
    centroid_drawn_list1.append(im_i_copy)

centers_on_im_list1 = find_cent_list(pred_cntrs_batch1)

drawn_cent_on_im_list1 = draw_cent_on_im_list(centers_on_im_list1, None, pred_batch1[0].shape)

show_batch(pred_batch1)
show_batch(pred_drawn_list1)
show_batch(centroid_drawn_list1)
show_batch(truth_batch1)            
show_batch(truth_act_batch1)        
show_batch(drawn_cent_on_im_list1)






a = np.zeros((1,1))
a_list = [a, a, a]
b_list = []
for a_i in a_list.copy():
    a_i[0] = 2.
    
    b_list.append(a_i)
TPi_strict_list = []
FPi_strict_list = []
TNi_strict_list = []
FNi_strict_list = []

Prec_strict_list = []
Reca_strict_list = []
F1_strict_list = []
MIoU_strict_list = []

for act_i, coi_i in zip(truth_batch1, pred_batch1):#zip(truth_act_batch1, drawn_cent_on_im_list1): 
    TPi_strict = np.logical_and(act_i,coi_i)*1.
    
    FPi_strict = np.greater(coi_i,0)*1. - TPi_strict
    
    TNi_strict = np.logical_and(np.logical_not(act_i), np.logical_not(coi_i))*1.
    
    FNi_strict = np.logical_not(coi_i)*1. - TNi_strict
    
    
    TPi_no = np.sum(TPi_strict)
    FPi_no = np.sum(FPi_strict)
    
    TNi_no = np.sum(TNi_strict)
    FNi_no = np.sum(FNi_strict)
    
    TPi_strict_list.append(TPi_no)
    FPi_strict_list.append(FPi_no)
    TNi_strict_list.append(TNi_no)
    FNi_strict_list.append(FNi_no)
    
    Prec_strict = TPi_no/(TPi_no + FPi_no)
    Reca_strict = TPi_no/(TPi_no + FNi_no)
    
    
    if Prec_strict*Reca_strict == 0:
        F1_strict = 0
    else:
        F1_strict = 2*Prec_strict*Reca_strict/(Prec_strict+Reca_strict)
    
    MIoU_strict = TPi_no/(TPi_no + FPi_no + FNi_no)
    
    
    Prec_strict_list.append(np.round(Prec_strict, 2))
    Reca_strict_list.append(np.round(Reca_strict, 2))
    F1_strict_list.append(np.round(F1_strict, 2))
    MIoU_strict_list.append(np.round(MIoU_strict, 2))
    
print(TPi_strict_list)
print(FPi_strict_list)
print(TNi_strict_list)
print(TNi_strict_list)

print(Prec_strict_list)
print(Reca_strict_list)
print(F1_strict_list)
print(MIoU_strict_list)
    


TPi_list = []
FPi_list = []
TNi_list = []
FNi_list = []

Prec_list = []
Reca_list = []
F1_list = []
MIoU_list = []

for act_i, coi_i in zip(truth_batch1, pred_batch1):
    
    TPi_no, FPi_no, TNi_no, FNi_no = TFmetrics(act_i,coi_i) 
    Prec, Reca, F1, MIoU = Metrics_from_TF(TPi_no, FPi_no, TNi_no, FNi_no)
    
    TPi_list.append(TPi_no)
    FPi_list.append(FPi_no)
    TNi_list.append(TNi_no)
    FNi_list.append(FNi_no)
    
    Prec_list.append(np.round(Prec, 2))
    Reca_list.append(np.round(Reca, 2))
    F1_list.append(np.round(F1, 2))
    MIoU_list.append(np.round(MIoU, 2))
    
print(TPi_list)
print(FPi_list)
print(TNi_list)
print(TNi_list)

print(Prec_list)
print(Reca_list)
print(F1_list)
print(MIoU_list)




   
y = np.array([1, 0, 0])
p = np.array([1,0.5,0])

yandp = np.logical_and(y,p)*1.
np.sum(yandp)

ynandp =1-yandp

'''
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:16:39 2024

@author: i_bab
"""
import cv2 as cv
import numpy as np
import tensorflow as tf
from loss_functions import act_list_3D
from dataLoad import flat_map_list_v2

def threshold_list(img_list, thresh_val, max_val, flag = cv.THRESH_TOZERO):
    thresh_list = []
    for im in img_list:
        _,thresh = cv.threshold(im, thresh_val, max_val, flag)
        thresh_list.append(thresh[...,None])
    return thresh_list

def get_cntrs_list(img_list, ret_flag = cv.RETR_TREE, app_flag = cv.CHAIN_APPROX_SIMPLE, thresh_flag = cv.THRESH_BINARY):
    cntrs_list = []
    for im in img_list:
        _,thresh = cv.threshold(im, 0.01, 1., thresh_flag)
        contours, hierarchy = cv.findContours(np.uint8(thresh), ret_flag, app_flag)
        cntrs_list.append(contours)
    return cntrs_list

def refine_thresh(pred, thresh):
    #print('1', pred.shape)
    #print('2',thresh.shape)
    thresh = tf.where(thresh > 0., 1., 0.)
    #print('3',thresh.shape)
    thresh = cv.dilate(np.float32(thresh[...,0]), np.ones((3,3)))[...,None]
    #print('4',thresh.shape)
    
    return pred*thresh
    
def refine_thresh_list(pred_list, thresh_list):
    new_list = []
    for pred, thresh in zip(pred_list, thresh_list):
        new_list.append(refine_thresh(pred, thresh))
    return new_list
    
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

def expand_pred_list(pred_batch, pred_drawn_expanded_list):
    pred_exp_list = []      
    for pred, exp in zip(pred_batch, pred_drawn_expanded_list):
        pred_exp = pred*exp
        pred_exp_list.append(pred_exp)
 
    return pred_exp_list

def act_from_pred_list(pred_batch):
    pred_cntrs_batch = get_cntrs_list(pred_batch)
    pred_drawn_expanded_list = draw_cntrs_exp_list(pred_batch.copy(), pred_cntrs_batch, thickness = -1)
    pred_exp_list = expand_pred_list(pred_batch, pred_drawn_expanded_list)
    pred_act_list = act_list_3D(pred_exp_list)
    flat_pred_act_list = flat_map_list_v2(pred_act_list)
    return flat_pred_act_list

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
    centers_on_image_list = []

    for cntrs in contours_on_image_list:
        centers_on_image = find_cent_on_im(cntrs)
        centers_on_image_list.append(centers_on_image) 

    return centers_on_image_list

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

def printIfNan(iNo, iName='iNo'):
    if np.isnan(iNo):
        print(iName +": %s"%iNo)

def TFmetrics(truth_map_i, pred_map_i):
    TPi = np.logical_and(truth_map_i, pred_map_i)*1.
    
    FPi = np.greater(pred_map_i,0.)*1. - TPi
    
    TNi = np.logical_and(np.logical_not(truth_map_i), np.logical_not(pred_map_i))*1.
    
    FNi = np.logical_not(pred_map_i)*1. - TNi
    
    
    TPi_no = np.sum(TPi)
    FPi_no = np.sum(FPi)
    
    TNi_no = np.sum(TNi)
    FNi_no = np.sum(FNi)
    
    printIfNan(TPi_no, 'TPi_no')
    printIfNan(FPi_no, 'FPi_no')
    printIfNan(TNi_no, 'TNi_no')
    printIfNan(FNi_no, 'FNi_no')
    
    return TPi_no, FPi_no, TNi_no, FNi_no

def Metrics_from_TF(TPi_no, FPi_no, TNi_no, FNi_no):
    if TPi_no == 0.:
        Prec = 0.
        Reca = 0.
        F1 = 0.
        MIoU = 0.
    else:
        Prec = TPi_no/(TPi_no + FPi_no + 1e-15)
        Reca = TPi_no/(TPi_no + FNi_no + 1e-15)
        F1 = 2.*Prec*Reca/(Prec+Reca + 1e-15)
        MIoU = TPi_no/(TPi_no + FPi_no + FNi_no + 1e-15)

    printIfNan(Prec, 'Prec')
    printIfNan(Reca, 'Reca')
    printIfNan(F1, 'F1')
    printIfNan(MIoU, 'MIoU')
    
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


#%%



#find_cent_on_im, find_cent_list, draw_cent_on_im, draw_cent_on_im_list, TFmetrics, Metrics_from_TF, Metrics_from_TF_batch
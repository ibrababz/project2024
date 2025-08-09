# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 08:51:56 2022

@author: i_bab
"""

import tensorflow as tf
import random
import cv2 as cv    
from tensorflow.keras import backend as B 
import numpy as np
from itertools import permutations, product


    
def flatten_map_v2(im_map, inv = 0):
    max_map = np.max(im_map, axis = (0,1), keepdims=True) #get maximum of each channel
    min_map = np.min(im_map, axis = (0,1), keepdims=True)
    im_map = (im_map-min_map)/(max_map - min_map + 1e-15)
    if not inv:
        im_map = np.maximum.reduce(im_map, axis=2, keepdims = True)
    else:
        im_map = np.minimum.reduce(im_map, axis =2, keepdims = True)
    return im_map

def flat_map_list_v2(map_list, inv = 0):
    flat_list = []
    for map1 in map_list:
        flat = flatten_map_v2(map1, inv)
        flat_list.append(flat)
    return flat_list  
def act_list_3D(map_list):
    return [np.float32(np.where(mapi < np.max(mapi, axis = (0,1), keepdims=True), 0.,1.)) for mapi in map_list]
    # act_list = []
    
    # for mapi in map_list:
    #     max_map = np.max(mapi, axis = (0,1), keepdims=True)
    #     im_act = np.where(mapi < max_map, 0.,1.)
        
    #     act_list.append(np.float32(im_act))
    
    # return act_list

def act_3D(mapi):
    max_map = np.max(mapi, axis = (0,1), keepdims=True)
    im_act = np.where(mapi < max_map, 0.,1.)
    
    return np.float32(im_act)   

def map_2_grid(dim, dim_grid, bboi):
    
    
    W, H = dim[0], dim[1]
    n_w, n_h = dim_grid[0], dim_grid[1]
    xyxy = bboi.to_xyxy_array()
    x1, y1, x2, y2 = xyxy[0,0], xyxy[0,1], xyxy[0,2], xyxy[0,3]
    
    x_c, y_c = (x2+x1)/2.0, (y2+y1)/2.0
    w, h = x2 - x1, y2 - y1
    n_x_c, n_y_c = x_c*n_w/W, y_c*n_h/H
    
    if n_x_c - np.floor(n_x_c) <=0.00001:
        grid_centers_x = [int(n_x_c) - 1, int(n_x_c)]
    else:
        grid_centers_x = [int(n_x_c)]
        
    if n_y_c - np.floor(n_y_c) <=0.00001:
        grid_centers_y = [int(n_y_c) - 1, int(n_y_c)]
    else:
        grid_centers_y = [int(n_y_c)]
        
    grid_centers_ji = []
   
    for j in grid_centers_y:  
        for i in grid_centers_x:    
            grid_centers_ji.append([j, i])
 
    return grid_centers_ji, w/W, h/H

class cells_object:
    def __init__(self, grid_centers_x, grid_centers_y, x_c, y_c, w, h, dim, dim_grid):
        self.grid_centers_x, self.grid_centers_y = grid_centers_x, grid_centers_y
        self.w, self.h = w, h
        self.x_c, self.y_c = x_c, y_c
        self.dim = dim
        self.dim_grid = dim_grid
        self.W, self.H = self.dim[0], self.dim[1]
        self.n_w, self.n_h = self.dim_grid[0], self.dim_grid[1]
        self.grid_centers_ji = []
        for j in self.grid_centers_y:  
            for i in self.grid_centers_x:
                j, i = int(np.clip(j, 0, self.n_h-1)), int(np.clip(i, 0, self.n_w-1))
                self.grid_centers_ji.append([j, i])
        self.pos = None
        self.pred_map = None
        self.r_map = None
    @classmethod
    def map_2_grid(cls,dim, dim_grid, bboi):
        W, H = dim[0], dim[1]
        n_w, n_h = dim_grid[0], dim_grid[1]
        xyxy = bboi.to_xyxy_array()
        x1, y1, x2, y2 = xyxy[0,0], xyxy[0,1], xyxy[0,2], xyxy[0,3]
        
        x_c, y_c = (x2+x1)/2.0, (y2+y1)/2.0
        w, h = x2 - x1, y2 - y1
        n_x_c, n_y_c = x_c*n_w/W, y_c*n_h/H
        
        if n_x_c - np.floor(n_x_c) <=0.00001:
            grid_centers_x = [int(n_x_c) - 1, int(n_x_c)]
        else:
            grid_centers_x = [int(n_x_c)]
            
        if n_y_c - np.floor(n_y_c) <=0.00001:
            grid_centers_y = [int(n_y_c) - 1, int(n_y_c)]
        else:
            grid_centers_y = [int(n_y_c)]

     
        return cls(grid_centers_x, grid_centers_y, x_c, y_c, w, h, dim, dim_grid)
    
    @classmethod
    def map_2_grid_from_koi(cls,dim, dim_grid, koi, index = 2):
        W, H = dim[0], dim[1]
        n_w, n_h = dim_grid[0], dim_grid[1]
        xy = koi.to_xy_array()[index,:]
        x_c, y_c = xy[0], xy[1]
        
        #x_c, y_c = (x2+x1)/2.0, (y2+y1)/2.0
        w, h = np.nan, np.nan
        n_x_c, n_y_c = x_c*n_w/W, y_c*n_h/H
        
        if n_x_c - np.floor(n_x_c) <=0.00001:
            grid_centers_x = [int(n_x_c) - 1, int(n_x_c)]
        else:
            grid_centers_x = [int(n_x_c)]
            
        if n_y_c - np.floor(n_y_c) <=0.00001:
            grid_centers_y = [int(n_y_c) - 1, int(n_y_c)]
        else:
            grid_centers_y = [int(n_y_c)]
            
     
        return cls(grid_centers_x, grid_centers_y, x_c, y_c, w, h, dim, dim_grid)
    
    @classmethod
    def map_2_grid_from_xy(cls,dim, dim_grid, xy):
        W, H = dim[0], dim[1]
        n_w, n_h = dim_grid[0], dim_grid[1]
        x_c, y_c = xy[0], xy[1]
        
        #x_c, y_c = (x2+x1)/2.0, (y2+y1)/2.0
        w, h = np.nan, np.nan
        n_x_c, n_y_c = x_c*n_w/W, y_c*n_h/H
        
        if n_x_c - np.floor(n_x_c) <=0.00001:
            grid_centers_x = [int(n_x_c) - 1, int(n_x_c)]
        else:
            grid_centers_x = [int(n_x_c)]
            
        if n_y_c - np.floor(n_y_c) <=0.00001:
            grid_centers_y = [int(n_y_c) - 1, int(n_y_c)]
        else:
            grid_centers_y = [int(n_y_c)]
            
     
        return cls(grid_centers_x, grid_centers_y, x_c, y_c, w, h, dim, dim_grid)
    
    def load_radial_map(self):
        if len(self.grid_centers_x) == 1 and len(self.grid_centers_y) == 1:
            
            x_map, y_map = np.arange(0,self.n_w)[None,:], np.arange(0,self.n_h)[:,None]
            x_map -= self.grid_centers_x[0]
            y_map -= self.grid_centers_y[0]


        elif len(self.grid_centers_x) > 1 and len(self.grid_centers_y) == 1:
            
            x_map, y_map = np.arange(0,self.n_w)[None,:], np.arange(0,self.n_h)[:,None]
            x_map = np.concatenate(((x_map - self.grid_centers_x[0])[:,:self.grid_centers_x[1]], (x_map - self.grid_centers_x[1])[:,self.grid_centers_x[1]:]), axis = 1)
            y_map -= self.grid_centers_y[0]

        
        elif len(self.grid_centers_x) == 1 and len(self.grid_centers_y) > 1:
            x_map, y_map = np.arange(0,self.n_w)[None,:], np.arange(0,self.n_h)[:,None]
            x_map -= self.grid_centers_y[0]
            y_map = np.concatenate(((y_map - self.grid_centers_y[0])[:self.grid_centers_y[1],:], (y_map - self.grid_centers_y[1])[self.grid_centers_y[1]:,:]), axis = 0)


        else:
            x_map, y_map = np.arange(0,self.n_w)[None,:], np.arange(0,self.n_h)[:,None]
            x_map = np.concatenate(((x_map - self.grid_centers_x[0])[:,:self.grid_centers_x[1]], (x_map - self.grid_centers_x[1])[:,self.grid_centers_x[1]:]), axis = 1)
            y_map = np.concatenate(((y_map - self.grid_centers_y[0])[:self.grid_centers_y[1],:], (y_map - self.grid_centers_y[1])[self.grid_centers_y[1]:,:]), axis = 0)

        
        #r_map = np.floor(np.sqrt(x_map*x_map + y_map*y_map+1e-15))
        #take off floor
        x_map = np.zeros(self.dim_grid) + x_map
        y_map = np.zeros(self.dim_grid) + y_map
        r_map = np.sqrt(x_map*x_map + y_map*y_map+1e-15)
        max_x, max_y = self.n_w-1, self.n_h-1
        r_max = np.sqrt(max_x*max_x + max_y*max_y + 1e-15)
        r_min = np.min(r_map +1e-15)
        
        r_map= 1/(r_max-r_min)*(r_map-r_min)
        self.r_map = r_map
        self.r_map_inv = 1-r_map
        return self.r_map, self.r_map_inv
    
    def draw_grid_on_image(self, img):
        n_w, n_h = self.n_w, self.n_h
        W, H = self.W, self.H
        for j in range(1,n_h):
            p1 = (0, int(j*H/n_h))
            p2 = (W, int(j*H/n_h))
            cv.line(img, p1, p2, (0,255,0))
        for j in range(1,n_w):
            p1 = (int(j*W/n_w), 0)
            p2 = (int(j*W/n_w),H)
            cv.line(img, p1, p2, (0,255,0))
    def draw_cells_on_image(self, img,  grid_centers_ji = None, color = (255,0,255)):
        if grid_centers_ji is None:
            grid_centers_ji = self.grid_centers_ji
        for cell in grid_centers_ji:
            x1, y1 = (cell[1])*self.W/self.n_w, (cell[0])*self.H/self.n_h
            x2, y2 = (cell[1]+1)*self.W/self.n_w, (cell[0]+1)*self.H/self.n_h
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(img, (x1,y1), (x2,y2), color, -1)
        self.draw_grid_on_image(img)
    
    def set_from_predictions(self, tensor_pred_0_i):
        self.grid_centers_ji_pred = []
        pos = tensor_pred_0_i
        pos = pos[...,1]
        pos = pos/B.max(pos)
        self.pos = pos
        flat = B.reshape(pos, (self.dim_grid[1]*self.dim_grid[0])).numpy()
        indexes = np.argsort(flat)
        indexes = np.flip(indexes)
        #flat = flat[indexes]
        for k in indexes:
            #print(k)
            i = int(k%self.dim_grid[0])
            j = int(k/self.dim_grid[0])
            self.grid_centers_ji_pred.append([j,i])
        
        #print(self.grid_centers_ji_pred)
        #GET INDEXES OF TOP N Predictions
        #USE THEM TO SORT INSTEAD OF FOR LOOP
        #print(pos)
        #print(self.grid_centers_y)
        #print(self.grid_centers_x)
        # for j in range(pos.shape[1]):
        #     for i in range(pos.shape[0]):
        #         #print(pos[j,i].numpy())
        #         if pos[j,i].numpy() == 1.:
        #             self.grid_centers_ji_pred.append([j,i])
        
    def draw_pred_on_image(self, img, tensor_pred_0_i, top = 1):
        if self.pos == None:
            self.set_from_predictions(tensor_pred_0_i)
        #print(self.grid_centers_ji_pred)
        self.draw_cells_on_image(img, self.grid_centers_ji_pred[:top], color = (0,125,255))
        
    def get_pred_map(self):
        map_shape = (self.dim_grid[1], self.dim_grid[0], 3)
        pred_map= np.zeros(map_shape)+255.
        pos = self.pos.numpy()
        pred_map*=pos[...,np.newaxis]
        
        self.pred_map = cv.resize(np.uint8(pred_map), (self.dim[1], self.dim[0]), interpolation = cv.INTER_NEAREST)
        return self.pred_map
    
    def draw_radial_map(self, r_flag = 1):
        if self.r_map is None:
            _ = self.load_radial_map()
            return self.draw_radial_map(r_flag)
        else:
            if r_flag:
                img = np.uint8(self.r_map[...,None]*np.ones((self.n_h, self.n_w, 3))*255.)
            else:
                img = np.uint8(self.r_map_inv[...,None]*np.ones((self.n_h, self.n_w, 3))*255.)
            img = cv.resize(img, (self.dim[1], self.dim[0]), interpolation = cv.INTER_NEAREST)
            
            return img
        
        
    def set_scores_from_tensor(self, tensor_pred_1_i_scores):
        self.grid_centers_ji_scores = []
        scores = tensor_pred_1_i_scores
        
        scores = scores/B.max(scores)
        self.scores = scores
        flat = B.reshape(scores, (self.dim_grid[1]*self.dim_grid[0])).numpy()
        indexes = np.argsort(flat)
        indexes = np.flip(indexes)
        for k in indexes:
            #print(k)
            i = int(k%self.dim_grid[0])
            j = int(k/self.dim_grid[0])
            self.grid_centers_ji_scores.append([j,i])
        
    def draw_scores_on_image(self, img, tensor_pred_1_i_scores, top = 1):
        self.set_scores_from_tensor(tensor_pred_1_i_scores)
        #print(self.grid_centers_ji_pred)
        self.draw_cells_on_image(img, self.grid_centers_ji_scores[:top], color = (0,125,0))    
        
    def overlay_pred_map(self, img):
        if self.pred_map is not None:
            overlay = np.uint8(np.float32(img)*(self.pred_map/255.))
            return overlay
        else:
            self.get_pred_map()
            return self.overlay_pred_map(img)
        
def custom_bboi_loss(y_batch_bboi, y_pred_0, y_pred_1, idx = 0):
    batch_size, n_h, n_w, n_p = y_pred_1.shape[0], y_pred_1.shape[1], y_pred_1.shape[2], y_pred_1.shape[3]
    #n_p should be 5 predictions: 4 coordinates (x,y,w,h) + cell distance from BBOX center to filter out the bad ones
    #Activation map from y_pred_0 can be concatenated 
    #as an input for one of the previous convolutions
    #as well as being output with the final output
    labels = np.zeros((batch_size, 1, n_p-1))
    for i, bboi in zip(range(batch_size), y_batch_bboi):
        labels[i,...,:]= bboi.to_xyxy_array()
    #print(labels.shape)
    labels = np.ones((batch_size, n_h*n_w, n_p-1))*labels
    labels = tf.Variable(labels, dtype = y_pred_0.dtype)
    #labels = tf.constant(labels, dtype = y_pred_0.dtype)
    shape = bboi.shape
    W, H = shape[1], shape[0]
    
    k_ind = B.arange(n_h*n_w)*tf.ones((batch_size,n_h*n_w), dtype = tf.int32)
    i_ind = k_ind%n_w
    j_ind = tf.cast(k_ind/n_h, dtype = k_ind.dtype)
    x_ind = W/n_w*tf.cast(i_ind, dtype = labels.dtype)
    y_ind = H/n_h*tf.cast(j_ind, dtype = labels.dtype)

    #class activations from frozen layers
    class_score = tf.cast(tensor_batch_per_sample_thresh_by_idx(y_pred_0[...,1,None], idx), dtype = labels.dtype)
    #print(class_score[...,None])
    class_act = tf.where(class_score>0., 1., 0.)
    
    x_cp, y_cp = B.reshape(y_pred_1[...,0], (batch_size, n_h*n_w)), B.reshape(y_pred_1[...,1], (batch_size, n_h*n_w))

    x_ca = (tf.cast(x_cp, dtype = class_act.dtype)*tf.cast(W/n_w, dtype = class_act.dtype) + x_ind)
    y_ca = (tf.cast(y_cp, dtype = class_act.dtype)*tf.cast(H/n_h, dtype = class_act.dtype) + y_ind)
    # print('max x_ca', tf.math.reduce_max(x_ca, axis =(0,1)))
    # print('min x_ca', tf.math.reduce_min(x_ca, axis =(0,1)))
    #print(B.concatenate((x_ca[...,None], W - x_ca[...,None])).shape)
    prior_w = 2*B.min(B.concatenate((x_ca[...,None], W - x_ca[...,None])), axis = 2)
    prior_h = 2*B.min(B.concatenate((y_ca[...,None], H - y_ca[...,None])), axis = 2)
    
    # print('max prior_w', tf.math.reduce_max(prior_w, axis =(0,1)))
    # print('min prior_w', tf.math.reduce_min(prior_w, axis =(0,1)))

    w_p, h_p = B.reshape(y_pred_1[...,2], (batch_size, n_h*n_w)), B.reshape(y_pred_1[...,3], (batch_size, n_h*n_w))
    
    
    # w_a = tf.cast(w_p, dtype = class_act.dtype)*tf.cast(W, dtype = class_act.dtype) 
    # h_a = tf.cast(h_p, dtype = class_act.dtype)*tf.cast(H, dtype = class_act.dtype)
    #print(prior_w.shape)
    #print(B.exp(w_p).shape)
    w_a = tf.cast(prior_w, dtype = class_act.dtype)*tf.cast(w_p, dtype = class_act.dtype) 
    #using B.exp(w_p) as in yoloV2 allows it to become bigger than prior which we don't want
    h_a = tf.cast(prior_h, dtype = class_act.dtype)*tf.cast(h_p, dtype = class_act.dtype) 
    
    x1_p, x2_p = (x_ca - w_a/2.),(x_ca + w_a/2.)
    y1_p, y2_p = (y_ca - h_a/2.),(y_ca + h_a/2.)
    
    # print('max prior_w', tf.math.reduce_max(prior_w, axis =(0,1)))
    # print('min prior_w', tf.math.reduce_min(prior_w, axis =(0,1)))
    
    pred = B.concatenate((x1_p[...,None], y1_p[...,None], x2_p[...,None], y2_p[...,None])) 
    pred = class_act[...,None]*pred
    
    # for i in range(len(class_act)):
    #     print(tf.reshape(class_act[i], (7,7)))
        
    #pred dimensions should now be (batch_size, (n_w*n_h), 4)
    #labels is now expanded to (batch_size, (n_w*n_h), 4)
    x_c, y_c =(labels[...,2]+labels[...,0])/2., (labels[...,3]+labels[...,1])/2.
    
    x_off, y_off =tf.cast(x_c, dtype = x_ind.dtype) - x_ind, tf.cast(y_c, dtype = y_ind.dtype)- y_ind
    xy_off = B.concatenate((x_off[...,None],y_off[...,None]))
    d_score = tf.sqrt(tf.reduce_sum(xy_off*xy_off, axis =2, keepdims = True)+1e-6)
    xy_off_norm = d_score/B.max(d_score, axis = 1, keepdims = True)
    xy_off_score = 1.- xy_off_norm


    labels_zeroed = tf.cast(class_act[...,None], labels.dtype)*labels

    labels_x12 = B.concatenate((labels_zeroed[...,0, None], labels_zeroed[...,2, None]))
    labels_y12 = B.concatenate((labels_zeroed[...,1, None], labels_zeroed[...,3, None]))

    pred_x34 = B.concatenate((pred[...,0, None], pred[...,2, None]))
    pred_y34 = B.concatenate((pred[...,1, None], pred[...,3, None]))

    x1234 = B.concatenate((labels_x12, pred_x34))
    y1234 = B.concatenate((labels_y12, pred_y34))

    Wtot = B.max(x1234, axis = 2, keepdims = True)-B.min(x1234, axis = 2, keepdims = True)
    Htot = B.max(y1234, axis = 2, keepdims = True)-B.min(y1234, axis = 2, keepdims = True)
    # print("==============================")
    # print('max Wtot', tf.math.reduce_max(Wtot, axis =(0,1)))
    # print('min Wtot', tf.math.reduce_min(Wtot, axis =(0,1)))
    
    C_score = tf.sqrt(Wtot*Wtot + Htot*Htot+1e-6)
    
    # print('max C_score', tf.math.reduce_max(C_score, axis =(0,1)))
    # print('min C_score', tf.math.reduce_min(C_score, axis =(0,1)))
    W1 = (labels_x12[...,1]-labels_x12[...,0])[...,None]
    #print(W1)
    W2 = (pred_x34[...,1]-pred_x34[...,0])[...,None]
    H1 = (labels_y12[...,1]-labels_y12[...,0])[...,None]
    H2 = (pred_y34[...,1]-pred_y34[...,0])[...,None]
    # print('max W1', tf.math.reduce_max(W1, axis =(0,1,2)))
    # print('min W1', tf.math.reduce_min(W1, axis =(0,1,2)))
    
    # print('max W2', tf.math.reduce_max(W2, axis =(0,1,2)))
    # print('min W2', tf.math.reduce_min(W2, axis =(0,1,2)))
    WI = W1 + W2 - Wtot
    HI = H1 + H2 - Htot

    WI, HI = B.clip(WI, 0., np.inf), B.clip(HI, 0., np.inf)

    I = HI*WI
    
    # print('max I', tf.math.reduce_max(I, axis =(0,1,2)))
    # print('min I', tf.math.reduce_min(I, axis =(0,1,2)))
    #print(I)
    A1, A2 = W1*H1, W2*H2
    U = A1 + A2 - I
    # print('max U', tf.math.reduce_max(U, axis =(0,1,2)))
    # print('min U', tf.math.reduce_min(U, axis =(0,1,2)))
    
    IOU = I/(U+1e-6)
    # print('max IOU', tf.math.reduce_max(IOU, axis =(0,1,2)))
    # print('min IOU', tf.math.reduce_min(IOU, axis =(0,1,2)))
    
    score_pred = tf.cast(B.reshape(y_pred_1[...,4], (batch_size, n_h*n_w, 1)), dtype = class_act.dtype)
    # print('max score pred', tf.math.reduce_max(score_pred, axis =(0,1,2)))
    # print('min score pred', tf.math.reduce_min(score_pred, axis =(0,1,2)))
    #print(IOU)
    #IOU_loss_map = tf.cast(tf.where(IOU>1e-15, (1.-IOU), 0.), dtype= IOU.dtype) #1-IOU where IOU is not zero
    #print("===============================")
    #print('max d_score', tf.math.reduce_max(d_score, axis =(0,1)))
    #print('min d_score', tf.math.reduce_min(d_score, axis =(0,1)))
    score_true = (IOU - d_score*d_score/(C_score*C_score))
    
    #print('max score true', tf.math.reduce_max(score_true, axis =(0,1,2)))
    #print('min score true', tf.math.reduce_min(score_true, axis =(0,1,2)))
    #score_true = IOU*class_score[...,None]
    #print(score_true)
    #premultiplying truth by class_score means no need to remultiply by activations
    score_pos_diff = (score_true - score_pred)*class_act[...,None] 
    score_pos_sqrt_diff = tf.sqrt(score_pos_diff*score_pos_diff+1e-6)
    #print(tf.reduce_sum(score_pos_sqrt_diff, axis = 1))
    score_pos_loss = tf.reduce_sum(score_pos_sqrt_diff, axis = 1)/(tf.reduce_sum(class_act[...,None], axis = 1)+1e-6)
    score_pos_batch_loss = B.mean(score_pos_loss)
    
    score_neg_diff = 0. - score_pred*(1. - class_act[...,None]) 
    # we just want to compare it with how close it is to zero no IOU score or anything
    #just masking the positive score values to zero is necessary
    #we later divide by the count of only the zero (neg) score values
    #print('max neg diff', tf.math.reduce_max(score_neg_diff, axis =(0,1,2)))
    #print('min neg diff',tf.math.reduce_min(score_neg_diff, axis =(0,1,2)))
    score_neg_sqrt_diff = tf.sqrt(score_neg_diff*score_neg_diff+1e-6)
    #print('max sqrt neg', tf.math.reduce_max(score_neg_sqrt_diff, axis =(0,1,2)))
    #print('min sqrt neg',tf.math.reduce_min(score_neg_sqrt_diff, axis =(0,1,2)))
    score_neg_loss = tf.reduce_sum(score_neg_sqrt_diff, axis = 1)/(tf.reduce_sum(1. - class_act[...,None], axis = 1)+1e-6)
    score_neg_batch_loss = B.mean(score_neg_loss)
    #print((x_ca - x_c).shape)
    dx, dy  = (x_ca - x_c)/((idx+1)*W/n_w), (y_ca - y_c)/((idx+1)*H/n_h)
    #dx, dy  = (x_ca - x_c), (y_ca - y_c)
    #predicted minus labeled normalized over max diff which is approximated as the spread of activated grid cells
    xy_distance = (tf.sqrt(dx*dx + dy*dy+1e-6)[...,None]*class_act[...,None])
    xy_distance_weighted = xy_off_score*xy_distance
    xy_loss = tf.reduce_sum(xy_distance_weighted, axis = 1)/(tf.reduce_sum(class_act[...,None], axis = 1)+1e-6)
    xy_loss = B.mean(xy_loss)
    
    dw, dh = (tf.sqrt(W1+1e-6) - tf.sqrt(W2+1e-6))/(idx+1), (tf.sqrt(H1+1e-6) - tf.sqrt(H2+1e-6))/(idx+1)
    #dw, dh = (tf.sqrt(W1+1e-6) - tf.sqrt(W2+1e-6)), (tf.sqrt(H1+1e-6) - tf.sqrt(H2+1e-6))
    #dw, dh = W1 - W2, H1 - H2
    wh_distance = (tf.sqrt(dw*dw + dh*dh+1e-6)*class_act[...,None])
    wh_distance_weighted = xy_off_score*wh_distance
    wh_loss = tf.reduce_sum(wh_distance_weighted, axis = 1)/(tf.reduce_sum(class_act[...,None], axis = 1)+1e-6)
    wh_loss = B.mean(wh_loss)
    #xy_regression_loss = 
    #print((tf.reduce_sum(class_act[...,None], axis = 1)+1e-15))
    #IOU_loss_map = (IOU_act  - IOU) #1 - IOU where there is IOU or zero
    #IOU_map = class_act[...,None]
    #IOU_loss = tf.reduce_sum(IOU_loss, axis = 1)/(tf.reduce_sum(IOU_act, axis = 1)+1e-15) #per sample average
    #IOU_loss = B.mean(IOU_loss, axis = 0)#for whole batch
    
    # print(dist_pred.dtype)
    # print(dist_score.dtype)
    #dist_loss = tf.sqrt((dist_score-dist_pred)*(dist_score-dist_pred))
    #dist_loss = class_act[...,None]*tf.sqrt((dist_score-dist_pred)*(dist_score-dist_pred))
    #print(dist_loss.shape)
    # print(B.max(dist_loss))
    # dist_loss =  tf.reduce_sum(dist_loss, axis = 1)/(tf.reduce_sum(class_act[...,None], axis = 1)+1e-15)
    # dist_loss = B.mean(dist_loss, axis=0)#for whole batch
    #print(score_pos_batch_loss.numpy(), score_neg_batch_loss.numpy(), xy_loss.numpy(), wh_loss.numpy())
    return score_pos_batch_loss, score_neg_batch_loss, xy_loss, wh_loss

def tensor_batch_per_sample_thresh_by_idx(y_pred_0, idx):
    t_shape = y_pred_0.shape
    batch_size, n_h, n_w = t_shape[0], t_shape[1], t_shape[2]
    flat = B.reshape(y_pred_0, (batch_size, n_w*n_h))
    sort_ind = tf.experimental.numpy.flip(tf.argsort(flat), axis = 1)
    flat_sort = tf.experimental.numpy.take_along_axis(flat, sort_ind, axis = 1)
    #zeroed = tf.where( flat < flat_sort[:,idx][:,None], 0., 1.)
    zeroed = tf.where( flat < flat_sort[:,idx][:,None], 0., flat)
    # if normalize:
    #     zeroed = tf.math.divide(zeroed, tf.math.reduce_max(zeroed, axis = 1, keepdims = True))
    return zeroed

def tensor_sample_thresh_by_idx(y_pred_0_i, idx):
    t_shape = y_pred_0_i.shape
    n_h, n_w = t_shape[0], t_shape[1]
    flat = B.reshape(y_pred_0_i, (n_w*n_h))
    sort_ind = tf.experimental.numpy.flip(tf.argsort(flat), axis = 0)
    flat_sort = tf.experimental.numpy.take_along_axis(flat, sort_ind, axis = 0)
    #zeroed = tf.where( flat < flat_sort[:,idx][:,None], 0., 1.)
    zeroed = tf.where( flat < flat_sort[idx], 0., flat)
    # if normalize:
    #     zeroed = tf.math.divide(zeroed, tf.math.reduce_max(zeroed, axis = 1, keepdims = True))
    return zeroed

def sample_thresh_by_idx(y_pred_0_i, idx):
    t_shape = y_pred_0_i.shape
    n_h, n_w = t_shape[0], t_shape[1]
    flat = np.reshape(y_pred_0_i, (n_w*n_h))
    sort_ind = np.flip(np.argsort(flat), axis = 0)
    flat_sort = np.take_along_axis(flat, sort_ind, axis = 0)
    #zeroed = tf.where( flat < flat_sort[:,idx][:,None], 0., 1.)
    zeroed = np.where( flat < flat_sort[idx], 0., flat)
    # if normalize:
    #     zeroed = tf.math.divide(zeroed, tf.math.reduce_max(zeroed, axis = 1, keepdims = True))
    return zeroed


def expand_tensor_map(pred, size):
    #pred is the tensor map reshaped from a 2D (H,W) tensor to a 1D (H*W) tensor
    shape = pred.shape
    skel = tf.ones((size, shape[0]))
    expanded = skel*tf.cast(pred, skel.dtype)
    return expanded

def expand_zero_tensor_map(pred, idx, act = False):
    pred_zeroed = tensor_sample_thresh_by_idx(pred, idx)
    pred_zeroed = tf.cast(pred_zeroed, dtype = tf.float32)

    meat = expand_tensor_map(pred_zeroed, idx+1)

    sort_ind = tf.experimental.numpy.flip(tf.argsort(pred_zeroed), axis = 0)
    pred_zeroed_sort = tf.experimental.numpy.take_along_axis(pred_zeroed, sort_ind, axis = 0)
    act_vect = pred_zeroed_sort[:idx+1,None]
    if act:
        expanded = tf.where(meat != act_vect, 0., 1.)
    else:
        expanded = tf.where(meat != act_vect, 0., meat)

    return expanded



def permutations_v2(iterable, r=None):
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    for indices in product(range(n), repeat=r):
        if len(set(indices)) == r:
            yield list(pool[i] for i in indices)

def my_tf_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier
def dist_lossi(mapi, predi, permutate = 0):
    idx = mapi.shape[2]-1
    # for i in range(mapi.shape[2]):
    #     show_batch(flat_map_list_v2([mapi[...,i,None]]))
    expanded = expand_zero_tensor_map(predi,idx, True)
    #print('pred_expanded')
    #print(expanded.numpy())
    acti = mapi

    acti_T= tf.transpose(acti, (2,0,1))
    acti_shape = tf.shape(acti_T)
    acti_re_3D = B.reshape(acti_T, (acti_shape[0],1,acti_shape[1]*acti_shape[2]))

    #acti_re_3D = acti_re[:,None,:]
    #print('weight maps')
    #print(np.round(acti_re_3D, 2))
    loss_map = acti_re_3D*expanded
    #print('multiplied')
    #print(np.round(loss_map, 2))
    #print('loss_map_reduced')
    loss_map_red = tf.reduce_sum(loss_map, 2)
    #print(np.round(loss_map_red, 2))
    mean_loss_list = []
    if permutate:
        perm_list = list(permutations_v2(np.arange(mapi.shape[2]), mapi.shape[2]))
        #print('type of perm_list = ', type(perm_list))
        random.shuffle(perm_list)
        perm_list = perm_list[:mapi.shape[2]]
    else:
        perm_list = [list(np.arange(mapi.shape[2]))]
    
    for perm in perm_list:
        #print('perm =', perm)
        deleted = []
        min_list = []
        for i in perm:
            #print('loss map red [',i,']')
            #print(np.round(loss_map_red_copy[i], 2))
            sort_i = tf.argsort(loss_map_red[i], 0)
    
            for j in range(len(loss_map_red[i])):
                # if sort_i[j] not in deleted and loss_map_red[i][sort_i[j]] != loss_map_red[i][sort_i[-1]]:
                if sort_i[j] not in deleted and loss_map_red[i][sort_i[j]] != 1.:
                    deleted.append(sort_i[j].numpy())
                    min_list.append(loss_map_red[i][sort_i[j]])
                    break
                elif loss_map_red[i][sort_i[j]] == 1.:
                    min_list.append(loss_map_red[i][sort_i[j]])
                    break
    
        #print('min_list')
        #print(np.round(min_list, 2))
        #print('deleted indexes')
        #print(deleted)
        #print('mean_loss_per_permutation')
        mean_loss = tf.reduce_mean(min_list)
        mean_loss_list.append(mean_loss)
        #print(mean_loss.numpy())
    return tf.reduce_min(mean_loss_list)



def dist_loss_from_list(weight_list, pred, permutate = 0):
    loss_list =[]
    
    for weighti, i in zip(weight_list, range(len(pred))):
        loss_list.append(dist_lossi(weighti, pred[i], permutate))
        # print('min_loss_per_pred')
        # print(loss_list[i].numpy())
        
    return tf.reduce_mean(loss_list)
# x1, y1, x2, y2 = 30., 30., 98., 98.

# batch_size = 2 
# bbox_tensor = tf.Variable(tf.zeros((batch_size, 1, 4), dtype = tf.float64))
# for i in range(batch_size):
#     bbox_tensor[i,...,:].assign(tf.constant([[x1+i*5.,y1+i*5.,x2+i*5.,y2+i*10.]], dtype = tf.float64))
    
# x_c, y_c =(bbox_tensor[...,2]+bbox_tensor[...,0])/2., (bbox_tensor[...,3]+bbox_tensor[...,1])/2.
# xy_c = B.concatenate((x_c,y_c))
# w, h = bbox_tensor[...,2]-bbox_tensor[...,0],bbox_tensor[...,3]-bbox_tensor[...,1]
# wh = B.concatenate((w,h))
# n_w, n_h = 3, 3        
# W, H = 224, 224

# k_ind = B.arange(n_h*n_w)*tf.ones((batch_size,n_h*n_w), dtype = tf.int32)

# i_ind = k_ind%n_w
# j_ind = tf.cast(k_ind/n_h, dtype = k_ind.dtype)
# x_ind = W/n_w*tf.cast(i_ind, dtype = tf.float64)
# y_ind = H/n_h*tf.cast(j_ind, dtype = tf.float64)

# x_off, y_off =tf.cast(x_c, dtype = x_ind.dtype) - x_ind, tf.cast(y_c, dtype = y_ind.dtype)- y_ind
# xy_off = B.concatenate((x_off[...,None],y_off[...,None]))
# xy_off_norm = xy_off - B.min(tf.reshape(xy_off,(batch_size, n_w*n_h,2)), axis = 1)[:,None,:]
# xy_off_norm = tf.sqrt(tf.reduce_sum(xy_off_norm*xy_off_norm, axis =2, keepdims = True))
# xy_off_norm = xy_off_norm/B.max(xy_off_norm, axis = 1, keepdims = True)



# from tensorflow.keras import backend as B
# import tensorflow as tf


# #pred_0 = tf.random.uniform((batch_size, n_h, n_w,1),0.,1.)
# pred_0 = tf.experimental.numpy.random.randint(0,9,(batch_size, n_h, n_w,1))
# flat = B.reshape(pred_0, (batch_size, n_w*n_h))
# sort_ind = tf.experimental.numpy.flip(tf.argsort(flat), axis = 1)
# flat_sort = tf.experimental.numpy.take_along_axis(flat, sort_ind, axis = 1)
# zeroed = tf.where( flat < flat_sort[:,1][:,None], 0., 1.)
# #zeroed = tf.where( flat < flat_sort[:,1][:,None], 0, flat)



# print(pred_0)
# print(flat)
# print(sort_ind)
# print(flat_sort)
# print(zeroed)
# print(B.reshape(pred_0, (batch_size, n_w*n_h)))
# print(tensor_batch_per_sample_thresh_by_idx(pred_0, 1, 1))
# n_p = 5
# y_pred_1 = tf.cast(tf.random.uniform((batch_size, n_h, n_w, n_p),0.25, 0.75),dtype = tf.float64)
# #flat_1 = B.reshape(y_pred_1, (batch_size, ))
# #:[n_w*n_h:batch_size*n_w*n_h:n_w*n_h]]
# # pos = y_pred[0][i,...,1]
# # pos /=np.max(pos)       
# class_act = tensor_batch_per_sample_thresh_by_idx(pred_0, 1, 1)

# x_cp, y_cp = B.reshape(y_pred_1[...,0], (batch_size, n_h*n_w)), B.reshape(y_pred_1[...,1], (batch_size, n_h*n_w))

# x_ca = (tf.cast(x_cp, dtype = class_act.dtype)*tf.cast(W/n_w, dtype = class_act.dtype) + x_ind)
# y_ca = (tf.cast(y_cp, dtype = class_act.dtype)*tf.cast(H/n_h, dtype = class_act.dtype) + y_ind)


# w_p, h_p = B.reshape(y_pred_1[...,2], (batch_size, n_h*n_w)), B.reshape(y_pred_1[...,3], (batch_size, n_h*n_w))
# w_a = tf.cast(w_p, dtype = class_act.dtype)*tf.cast(W, dtype = class_act.dtype) 
# h_a = tf.cast(h_p, dtype = class_act.dtype)*tf.cast(H, dtype = class_act.dtype)
# x1, x2 = (x_ca - w_a/2.),(x_ca + w_a/2.)
# y1, y2 = (y_ca - h_a/2.),(y_ca + h_a/2.)

# pred = B.concatenate((x1[...,None], y1[...,None], x2[...,None], y2[...,None])) 
# pred = class_act[...,None]*pred

# labels  = bbox_tensor*tf.Variable(tf.ones((batch_size, n_h*n_w, n_p-1), dtype = tf.float64))


# x_c, y_c =(labels[...,2]+labels[...,0])/2., (labels[...,3]+labels[...,1])/2. 

# x_off, y_off =tf.cast(x_c, dtype = x_ind.dtype) - x_ind, tf.cast(y_c, dtype = y_ind.dtype)- y_ind
# xy_off = B.concatenate((x_off[...,None],y_off[...,None]))
# #xy_off_norm = xy_off - B.min(tf.reshape(xy_off,(batch_size, n_w*n_h,2)), axis = 1)[:,None,:]
# xy_off_norm = tf.sqrt(tf.reduce_sum(xy_off*xy_off, axis =2, keepdims = True))
# xy_off_norm = xy_off_norm/B.max(xy_off_norm, axis = 1, keepdims = True)
# dist_score = 1- xy_off_norm

# labels_zeroed = tf.cast(class_act[...,None], labels.dtype)*labels

# labels_x12 = B.concatenate((labels_zeroed[...,0, None], labels_zeroed[...,2, None]))
# labels_y12 = B.concatenate((labels_zeroed[...,1, None], labels_zeroed[...,3, None]))

# pred_x34 = B.concatenate((pred[...,0, None], pred[...,2, None]))
# pred_y34 = B.concatenate((pred[...,1, None], pred[...,3, None]))

# x1234 = B.concatenate((labels_x12, pred_x34))
# y1234 = B.concatenate((labels_y12, pred_y34))

# Wtot = B.max(x1234, axis = 2, keepdims = True)-B.min(x1234, axis = 2, keepdims = True)
# Htot = B.max(y1234, axis = 2, keepdims = True)-B.min(y1234, axis = 2, keepdims = True)
# W1 = (labels_x12[...,1]-labels_x12[...,0])[...,None]
# W2 =  (pred_x34[...,1]-pred_x34[...,0])[...,None]
# H1 = (labels_y12[...,1]-labels_y12[...,0])[...,None]
# H2 =  (pred_y34[...,1]-pred_y34[...,0])[...,None]

# WI = W1 + W2 - Wtot
# HI = H1 + H2 - Htot

# WI, HI = B.clip(WI, 0., np.inf), B.clip(HI, 0., np.inf)

# I = HI*WI
# A1, A2 = W1*H1, W2*H2
# U = A1 + A2 - I
# IOU = I/(U+1e-15)

# dim_grid = (7,7)   
# n_h, n_w = 7, 7
# grid_centers_x = [3, 4]
# grid_centers_y = [3, 4]
# x_map, y_map = B.arange(0.,n_w)[None,:], B.arange(0.,n_h)[:,None]
# x_map = B.concatenate(((x_map - grid_centers_x[0])[:,:grid_centers_x[1]], (x_map - grid_centers_x[1])[:,grid_centers_x[1]:]), axis = 1)
# y_map = B.concatenate(((y_map - grid_centers_y[0])[:grid_centers_y[1],:], (y_map - grid_centers_y[1])[grid_centers_y[1]:,:]), axis = 0)
# x_map = B.zeros(dim_grid) + x_map
# y_map = B.zeros(dim_grid) + y_map
# r_map = tf.floor(B.sqrt(x_map*x_map + y_map*y_map))
# r_map /=B.max(r_map)
# r_inv = 1-r_map


def class_loss_from_one_tensor(pred, bboi):
    #pred is an input tensor that represents one sample from the batch 
    #(h,w, class_predictions) They are the output of of a softmax activation
    #class_predictions they will be a probabiltiy distribtuion varying between [0,1] an [1,0]
    t_shape = pred.shape
    #print('tensor shape = ', t_shape)
    dim_grid = (t_shape[1], t_shape[0]) #shape is in HWC whereas DIM is in WH
    dim = (bboi.shape[1], bboi.shape[0])
    cells_ji, w, h = map_2_grid(dim, dim_grid, bboi)
    c = np.zeros((dim_grid[1], dim_grid[0], 1))
    #wh = np.random.uniform(0 , 1, (dim_grid[1], dim_grid[0], 2))
    for index_ji in cells_ji:
        c[index_ji[0], index_ji[1]] = 1.0
        #wh[index_ji[0], index_ji[1]] = np.array([w,h])
    #c[...,0]
    c = tf.constant(c)
    d = 1-c
    #d[...,0]
    e = B.concatenate((d,c), axis =2)
    
    q = B.reshape(e, (dim_grid[0]*dim_grid[1],2)) 
    l =  B.reshape(pred, (dim_grid[0]*dim_grid[1],2))
    
    q_list = list(q)
    l_list = list(l)
    
    one_labels = []
    one_pred = []

    adjust_index = 0
    for index_ji in cells_ji:
        one_labels.append(q_list.pop(index_ji[0]*dim_grid[0]+index_ji[1] - adjust_index)) #index in list is width*j + i - amount_of_indexes_popped
        one_pred.append(l_list.pop(index_ji[0]*dim_grid[0]+index_ji[1]- adjust_index))
        adjust_index +=1
        
    loss_positive = tf.keras.losses.BinaryCrossentropy()(one_labels, one_pred)  
    loss_negative = tf.keras.losses.BinaryCrossentropy()(q_list,l_list)
    #print('loss_pos = ', loss_positive,' loss_neg = ', loss_negative )
    return loss_positive, loss_negative

def class_loss_from_batch(batch, batch_bboi, weight = 1.0):
    #print('calculating batch loss')
    #assuming batch is a list of tensors
    batch_loss_pos = tf.constant(0.0)
    batch_loss_neg = tf.constant(0.0)
    i=0.0
    #print('entering for loop')
    for i in range(len(batch)):
        loss_pos, loss_neg = class_loss_from_one_tensor(batch[i], batch_bboi[i])
        batch_loss_pos+=loss_pos
        batch_loss_neg+=(weight*loss_neg)
        i+=1.0
    batch_loss_pos /=i #get average
    batch_loss_neg /=i
    return batch_loss_pos, batch_loss_neg

def class_loss_from_batch_bboi(batch, batch_bboi):
    #batch is in a tensor of dimensions (batch_size, H, W, C)
    #batch_bboi is a list of len batch_size
    t_shape = batch.shape
    batch_size = t_shape[0]
    dim_grid = (t_shape[2], t_shape[1])
    shape = batch_bboi[0].shape
    dim = (shape[1], shape[0])
    
    c = np.zeros((batch_size, dim_grid[1], dim_grid[0], 1))
    cells_list = []
    
    for bboi in batch_bboi:
        cells_ob = cells_object.map_2_grid(dim, dim_grid, bboi)
        # for cell in cells_ji:
        #     print(cell)
        cells_list.append(cells_ob) 
    
    for i, cells_ob in zip(range(batch_size), cells_list):
        
         for index_ji in cells_ob.grid_centers_ji:
             c[i,index_ji[0], index_ji[1]] = 1   
    d = 1-c
    
    e = B.concatenate((d,c), axis = 3)    
    
    g = B.reshape(c[...,0], (batch_size*dim_grid[0]*dim_grid[1],1))
    
    l =  B.reshape(batch[...,1], (batch_size*dim_grid[0]*dim_grid[1],1))
    g = tf.cast(g, l.dtype)
  
   
    loss = tf.reduce_sum(tf.clip_by_value((-g*tf.math.log(g*l+ 1e-15)), 0.0, 15.))/tf.reduce_sum(g)
    
    
    f = B.reshape(d[...,0], (batch_size*dim_grid[0]*dim_grid[1],1))
    f = tf.cast(f, l.dtype)
    
    loss2 = tf.reduce_sum(tf.clip_by_value((-f*tf.math.log(f*(1-l)+ 1e-15)), 0.0, 15.))/tf.reduce_sum(f)
    # gl = B.concatenate((g,l,f,l), axis = 1)   
    # print(gl[:dim_grid[0]*dim_grid[1]])
    
    h = B.reshape(e, (batch_size*dim_grid[0]*dim_grid[1],2))
    k =  B.reshape(batch, (batch_size*dim_grid[0]*dim_grid[1],2))
    
    loss3 = tf.keras.losses.BinaryCrossentropy()(h,k)

    return loss, loss2, loss3, cells_list

def class_loss_from_batch_koi(batch, batch_koi, index = 2):
    #batch is in a tensor of dimensions (batch_size, H, W, C)
    #batch_bboi is a list of len batch_size
    t_shape = batch.shape
    batch_size = t_shape[0]
    dim_grid = (t_shape[2], t_shape[1])
    shape = batch_koi[0].shape
    dim = (shape[1], shape[0])
    
    c = np.zeros((batch_size, dim_grid[1], dim_grid[0], 1))
    cells_list = []
    cell_pos_weights_list = []
    cell_neg_weights_list = []
    for koi in batch_koi:
        cells_ob = cells_object.map_2_grid_from_koi(dim, dim_grid, koi, index)
        neg_weights, pos_weights = cells_ob.load_radial_map()
        cell_pos_weights_list.append(pos_weights)
        cell_neg_weights_list.append(neg_weights)
        # for cell in cells_ji:
        #     print(cell)
        cells_list.append(cells_ob) 
    
    pos_weights_batch = tf.stack(cell_pos_weights_list)
    neg_weights_batch = tf.stack(cell_neg_weights_list)
    
    for i, cells_ob in zip(range(batch_size), cells_list):
        
         for index_ji in cells_ob.grid_centers_ji:
             c[i,index_ji[0], index_ji[1]] = 1    
    d = 1-c
    
    e = B.concatenate((d,c), axis = 3)    
    
    g = B.reshape(c[...,0], (batch_size*dim_grid[0]*dim_grid[1],1))
    gweights = B.reshape(pos_weights_batch, (batch_size*dim_grid[0]*dim_grid[1],1))
    
    l =  B.reshape(batch[...,1], (batch_size*dim_grid[0]*dim_grid[1],1))
    #positive predictions
    g = tf.cast(g, l.dtype)
    gweights = tf.cast(gweights, l.dtype)
   
    loss = tf.reduce_sum(tf.clip_by_value((-g*gweights*tf.math.log(g*l+ 1e-15)), 0.0, 15.))/tf.reduce_sum(g)
    
   
    
    f = B.reshape(d[...,0], (batch_size*dim_grid[0]*dim_grid[1],1))
    fweights = B.reshape(neg_weights_batch, (batch_size*dim_grid[0]*dim_grid[1],1))
    f = tf.cast(f, l.dtype)
    fweights = tf.cast(fweights, l.dtype)
    
    loss2 = tf.reduce_sum(tf.clip_by_value((-f*fweights*tf.math.log(f*(1-l)+ 1e-15)), 0.0, 15.))/tf.reduce_sum(f)
    # gl = B.concatenate((g,l,f,l), axis = 1)   
    # print(gl[:dim_grid[0]*dim_grid[1]])
    
    l_2d = B.reshape(batch[...,1], (batch_size, dim_grid[1]*dim_grid[0]))
    l_2d_max_only = tf.where(l_2d>=tf.reduce_max(l_2d, axis=1, keepdims = True), l_2d, 0.)

    l_2d_max_only_flat = B.reshape(l_2d_max_only, (batch_size*dim_grid[1]*dim_grid[0], 1))
    
    dist_weights = fweights
    dist_loss_of_max = tf.reduce_sum(l_2d_max_only_flat*dist_weights, axis = 0)/batch_size
    
    # h = B.reshape(e, (batch_size*dim_grid[0]*dim_grid[1],2))
    # k =  B.reshape(batch, (batch_size*dim_grid[0]*dim_grid[1],2))
    
    loss3 = dist_loss_of_max

    return loss, loss2, loss3, cells_list



def map_loss_from_batch_koi(batch, batch_koi, index = 2):
    #batch is in a tensor of dimensions (batch_size, H, W, C)
    #batch_bboi is a list of len batch_size
    t_shape = batch.shape
    batch_size = t_shape[0]
    dim_grid = (t_shape[2], t_shape[1])
    shape = batch_koi[0].shape
    dim = (shape[1], shape[0])
    
    c = np.zeros((batch_size, dim_grid[1], dim_grid[0], 1))
    cells_list = []
    cell_pos_weights_list = []
    cell_neg_weights_list = []
    for koi in batch_koi:
        cells_ob = cells_object.map_2_grid_from_koi(dim, dim_grid, koi, index)
        neg_weights, pos_weights = cells_ob.load_radial_map()
        cell_pos_weights_list.append(pos_weights)
        cell_neg_weights_list.append(neg_weights)
        # for cell in cells_ji:
        #     print(cell)
        cells_list.append(cells_ob) 
    
    pos_weights_batch = tf.stack(cell_pos_weights_list)
    neg_weights_batch = tf.stack(cell_neg_weights_list)
    
    for i, cells_ob in zip(range(batch_size), cells_list):
        
         for index_ji in cells_ob.grid_centers_ji:
             c[i,index_ji[0], index_ji[1]] = 1   
    d = 1-c
    
    e = B.concatenate((d,c), axis = 3)    
    
    g = B.reshape(c[...,0], (batch_size*dim_grid[0]*dim_grid[1],1))
    gweights = B.reshape(pos_weights_batch, (batch_size*dim_grid[0]*dim_grid[1],1))
    
    l =  B.reshape(batch[...,1], (batch_size*dim_grid[0]*dim_grid[1],1))
    #positive predictions
    g = tf.cast(g, l.dtype)
    gweights = tf.cast(gweights, l.dtype)
   
    loss = tf.reduce_sum(tf.clip_by_value((-g*gweights*tf.math.log(g*l+ 1e-15)), 0.0, 15.))/tf.reduce_sum(g)
    
 
    
    
    f = B.reshape(d[...,0], (batch_size*dim_grid[0]*dim_grid[1],1))
    fweights = B.reshape(neg_weights_batch, (batch_size*dim_grid[0]*dim_grid[1],1))
    f = tf.cast(f, l.dtype)
    fweights = tf.cast(fweights, l.dtype)
    
    #idx = 5
    #max_map = tensor_batch_per_sample_thresh_by_idx(batch[...,1,None], idx)#map of top 5 activations per batch
    
    #max_map_loss = tf.reduce_sum(B.reshape(max_map,(batch_size*dim_grid[0]*dim_grid[1],1))*fweights, axis = 0)/(batch_size*idx+1e-15)
    # print(max_map_loss)
    loss2 = tf.reduce_sum(tf.clip_by_value((-f*fweights*tf.math.log(f*(1-l)+ 1e-15)), 0.0, 15.))/tf.reduce_sum(f)
    # gl = B.concatenate((g,l,f,l), axis = 1)   
    # print(gl[:dim_grid[0]*dim_grid[1]])
    
    l_2d = B.reshape(batch[...,1], (batch_size, dim_grid[1]*dim_grid[0]))
    l_2d_max_only = tf.where(l_2d>=tf.reduce_max(l_2d, axis=1, keepdims = True), l_2d, 0.)

    l_2d_max_only_flat = B.reshape(l_2d_max_only, (batch_size*dim_grid[1]*dim_grid[0], 1))
    
    dist_weights = fweights
    dist_loss_of_max = tf.reduce_sum(l_2d_max_only_flat*dist_weights, axis = 0)/batch_size
    
    # h = B.reshape(e, (batch_size*dim_grid[0]*dim_grid[1],2))
    # k =  B.reshape(batch, (batch_size*dim_grid[0]*dim_grid[1],2))
    
    loss3 = tf.reduce_mean(gweights*tf.sqrt((l-gweights)*(l-gweights)+1e-15), axis = 0)
    #can add dist_loss_of_max to loss3
    return loss, loss2, 10*loss3, cells_list


def loss_from_map_tensor(pred, flat_arr, weight_arr):
    pred_shape = pred.shape
    batch_size = pred_shape[0]
    w_pred, h_pred =  pred_shape[2], pred_shape[1]
    
    #pred =  np.reshape(pred[...,1], (batch_size*w_pred*h_pred,1)) #B.reshape
    pred =  B.reshape(pred, (batch_size*w_pred*h_pred,1))         #B.reshape
    
    weights = tf.cast(weight_arr, pred.dtype)#tf.cast
    weights =  B.reshape(weights, (batch_size*w_pred*h_pred,1))#B.reshape
    
    truth = tf.cast(flat_arr, pred.dtype)#tf.cast
    truth =  B.reshape(truth, (batch_size*w_pred*h_pred,1))#B.reshape
    
    #loss = tf.reduce_mean(weights*tf.sqrt((pred-truth)*(pred-truth)+1e-15), axis = 0) #tf.reduce_mean
    loss = tf.reduce_mean(tf.sqrt((pred-truth)*(pred-truth)+1e-15), axis = 0) #tf.reduce_mean

    return loss


def getVectors(pred, map_list, act_list = None):
    if act_list is None:
        act_list = act_list_3D(map_list)
    shape_pred = pred.shape
    batch_size = shape_pred[0]
    vect_pred = B.reshape(pred, (batch_size*shape_pred[1]*shape_pred[2],1))
    flat_map_list1 = flat_map_list_v2(map_list)
    flat_act_list1 = flat_map_list_v2(act_list)
    flat_map_arr1 = np.array(flat_map_list1)
    flat_act_arr1 = np.array(flat_act_list1)
    
    shape_map = np.shape(flat_map_arr1)
    shape_act = np.shape(flat_act_arr1)
    
    #print("shape_map = ", shape_map)
    #print("shape_act = ", shape_act)
    vect_map1 = B.reshape(flat_map_arr1, (shape_map[0]*shape_map[1]*shape_map[2],shape_map[3]))
    vect_act1 = B.reshape(flat_act_arr1, (shape_act[0]*shape_act[1]*shape_act[2],shape_act[3]))

    
    # vect_map1_inv = 1 - vect_map1
    # vect_act1_inv = 1 - vect_act1
    
    return vect_pred, vect_map1, vect_act1
    

def tensorPosNegLoss(vect_pred, vect_map1, vect_act1, vect_map1_inv, vect_act1_inv):
    # print(vect_pred.shape, vect_map1.shape, vect_act1.shape)
    neg_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1_inv*vect_map1_inv*tf.math.log(vect_act1_inv*(1-vect_pred) + 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1_inv)
    pos_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1*tf.math.log(vect_act1*vect_pred+ 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1)
    return pos_loss, neg_loss

def tensorDiceLoss(vect_pred, vect_map1, vect_act1, vect_map1_inv, vect_act1_inv):
    dice_pos = 1 - 2.*tf.reduce_sum(vect_act1*vect_pred)/(tf.reduce_sum(vect_act1*vect_act1) + tf.reduce_sum(vect_pred*vect_pred)+1.e-15)
    dice_neg = 1 - 2.*tf.reduce_sum(vect_act1_inv*(1.-vect_pred))/(tf.reduce_sum(vect_act1_inv* vect_act1_inv) + tf.reduce_sum((1.-vect_pred)*(1.-vect_pred))+1.e-15)
    dice_loss = (dice_pos+dice_neg)/2
    return dice_loss    
    
def tensor_pos_neg_loss(pred, map_list, act_list = None):
    
    if act_list is None:
        act_list = act_list_3D(map_list)
    shape_pred = pred.shape
    batch_size = shape_pred[0]
    vect_pred = B.reshape(pred, (batch_size*shape_pred[1]*shape_pred[2],1))
    flat_map_list1 = flat_map_list_v2(map_list)
    flat_act_list1 = flat_map_list_v2(act_list)
    flat_map_arr1 = np.array(flat_map_list1)
    flat_act_arr1 = np.array(flat_act_list1)
    
    shape_map = np.shape(flat_map_arr1)
    shape_act = np.shape(flat_act_arr1)
    
    #print("shape_map = ", shape_map)
    #print("shape_act = ", shape_act)
    vect_map1 = B.reshape(flat_map_arr1, (shape_map[0]*shape_map[1]*shape_map[2],shape_map[3]))
    vect_act1 = B.reshape(flat_act_arr1, (shape_act[0]*shape_act[1]*shape_act[2],shape_act[3]))

    vect_map1_inv = 1 - vect_map1
    vect_act1_inv = 1 - vect_act1
    #pos_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1*vect_map1*tf.math.log(vect_act1*vect_pred+ 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1)
    neg_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1_inv*vect_map1_inv*tf.math.log(vect_act1_inv*(1-vect_pred) + 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1_inv)
    pos_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1*tf.math.log(vect_act1*vect_pred+ 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1)
    # neg_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1_inv*tf.math.log(vect_act1_inv*(1-vect_pred) + 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1_inv)

    return pos_loss, neg_loss


def tensor_map_loss(pred, map_list, act_list = None):
    
    if act_list is None:
        act_list = act_list_3D(map_list)
    shape_pred = pred.shape
    batch_size = shape_pred[0]
    vect_pred = B.reshape(pred, (batch_size*shape_pred[1]*shape_pred[2],1))
    flat_map_list1 = flat_map_list_v2(map_list)
    flat_act_list1 = flat_map_list_v2(act_list)
    flat_map_arr1 = np.array(flat_map_list1)
    flat_act_arr1 = np.array(flat_act_list1)
    
    shape_map = np.shape(flat_map_arr1)
    shape_act = np.shape(flat_act_arr1)
    
    #print("shape_map = ", shape_map)
    #print("shape_act = ", shape_act)
    vect_map1 = B.reshape(flat_map_arr1, (shape_map[0]*shape_map[1]*shape_map[2],shape_map[3]))
    vect_act1 = B.reshape(flat_act_arr1, (shape_act[0]*shape_act[1]*shape_act[2],shape_act[3]))

    map_loss = tf.sqrt(tf.reduce_mean(vect_map1*(vect_act1-vect_pred)*(vect_act1-vect_pred), axis = 0)+1e-15)
    return map_loss


def tensor_dice_loss(pred, map_list, act_list = None):
    
    if act_list is None:
        act_list = act_list_3D(map_list)
    shape_pred = pred.shape
    batch_size = shape_pred[0]
    vect_pred = B.reshape(pred, (batch_size*shape_pred[1]*shape_pred[2],1))
    flat_map_list1 = flat_map_list_v2(map_list)
    flat_act_list1 = flat_map_list_v2(act_list)
    flat_map_arr1 = np.array(flat_map_list1)
    flat_act_arr1 = np.array(flat_act_list1)
    
    shape_map = np.shape(flat_map_arr1)
    shape_act = np.shape(flat_act_arr1)
    
    #print("shape_map = ", shape_map)
    #print("shape_act = ", shape_act)
    vect_map1 = B.reshape(flat_map_arr1, (shape_map[0]*shape_map[1]*shape_map[2],shape_map[3]))
    vect_act1 = B.reshape(flat_act_arr1, (shape_act[0]*shape_act[1]*shape_act[2],shape_act[3]))

    
    vect_map1_inv = 1 - vect_map1
    vect_act1_inv = 1 - vect_act1
    dice_pos = 1 - 2.*tf.reduce_sum(vect_act1*vect_pred)/(tf.reduce_sum(vect_act1*vect_act1) + tf.reduce_sum(vect_pred*vect_pred)+1.e-15)
    dice_neg = 1 - 2.*tf.reduce_sum(vect_act1_inv*(1.-vect_pred))/(tf.reduce_sum(vect_act1_inv* vect_act1_inv) + tf.reduce_sum((1.-vect_pred)*(1.-vect_pred))+1.e-15)
    dice_loss = (dice_pos+dice_neg)/2
    #pos_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1*vect_map1*tf.math.log(vect_act1*vect_pred+ 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1)
    #neg_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1_inv*vect_map1_inv*tf.math.log(vect_act1_inv*(1-vect_pred) + 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1_inv)
    #pos_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1*tf.math.log(vect_act1*vect_pred+ 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1)
    #neg_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1_inv*tf.math.log(vect_act1_inv*(1-vect_pred) + 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1_inv)

    return dice_loss


def BCE_loss_from_map_tensor(pred, flat_arr, weight_arr):
    pred_shape = pred.shape
    batch_size = pred_shape[0]
    w_pred, h_pred =  pred_shape[2], pred_shape[1]
    
    #pred =  np.reshape(pred[...,1], (batch_size*w_pred*h_pred,1)) #B.reshape
    pred =  B.reshape(pred, (batch_size*w_pred*h_pred,1))         #B.reshape
    
    weights = tf.cast(weight_arr, pred.dtype)#tf.cast
    weights =  B.reshape(weights, (batch_size*w_pred*h_pred,1))#B.reshape
    
    truth = tf.cast(flat_arr, pred.dtype)#tf.cast
    truth =  B.reshape(truth, (batch_size*w_pred*h_pred,1))#B.reshape
    
    #loss = tf.reduce_mean(weights*tf.sqrt((pred-truth)*(pred-truth)+1e-15), axis = 0) #tf.reduce_mean
    loss = tf.reduce_mean(tf.sqrt((pred-truth)*(pred-truth)+1e-15), axis = 0) #tf.reduce_mean

    return loss

# batch_size  = 2


# dim_grid = (3,3)
# wh = np.random.uniform(0 , 1, (batch_size, dim_grid[1], dim_grid[0]))
# wh=tf.constant(wh)
# wh_flat = tf.reshape(wh, (batch_size, dim_grid[0]*dim_grid[1]))
# wh_flat_max_only = tf.where(wh_flat>=tf.reduce_max(wh_flat, axis=1, keepdims = True), wh_flat, 0.)
# wh_max_only = tf.reshape(wh_flat_max_only, (batch_size, dim_grid[1], dim_grid[0]))

# wh_flat_flat = tf.reshape(wh_flat_max_only, (batch_size*dim_grid[1]*dim_grid[0], 1))
# tf.reduce_sum(wh_flat_flat, axis = 0)/batch_size
# input_tensor = tf.constant(np.ones((4,2,3)))
# input_tensor.shape
# dim = (224,224)
# shape = (dim[1], dim[0], 3)
# batch_size  = 2

# bboi1 = BoundingBoxesOnImage.from_xyxy_array(np.array([[100, 4,156,60]]), shape)
# bboi2 = BoundingBoxesOnImage.from_xyxy_array(np.array([[4,36,68,220]]), shape)
# bboi_list = [bboi1, bboi2]
# dim_grid = (7,7)

# c = np.zeros((batch_size, dim_grid[1], dim_grid[0], 1))
# cells_list = []
# wh_list = []
# for bboi in bboi_list:
#     cells_ji, w, h = map_2_grid(dim, dim_grid, bboi)
#     for cell in cells_ji:
#         print(cell)
#     print(w, h)
#     cells_list.append(cells_ji)
#     wh_list.append([w,h])



# wh = np.random.uniform(0 , 1, (batch_size, dim_grid[1], dim_grid[0], 2))
# for i, cells_ji in zip(range(batch_size), cells_list):
#     for index_ji in cells_ji:
#         c[i,index_ji[0], index_ji[1]] = 1
#         wh[i,index_ji[0], index_ji[1]] = np.array([w,h])

# c[...,0]
# d = 1-c
# d[...,0]

# e = B.concatenate((d,c), axis =3)


# f, g = B.reshape(e[...,0], (batch_size*dim_grid[0]*dim_grid[1],1)), B.reshape(e[...,1], (batch_size*dim_grid[0]*dim_grid[1],1))

# g = B.reshape(c[...,0], (batch_size*dim_grid[0]*dim_grid[1],1))

# losses= []
# losses_wh = []
# #pred = e.copy() #for when you set everything else to zero
# pred = np.random.uniform(0,1, (batch_size, dim_grid[1], dim_grid[0], 2))
# #pred = tf.constant(pred)
# pred_wh = wh.copy()
# for i, cells_ji in zip(range(batch_size),cells_list) :
#     l1, l2 = 0.3, 0.7
#     w1, h1 = 0.2, 0.9
#     for index_ji in cells_ji:
#         pred[i, index_ji[0], index_ji[1]] = np.array([l1, l2])
#         pred_wh[i, index_ji[0], index_ji[1]] = np.array([w1, h1])
#         loss = tf.keras.losses.BinaryCrossentropy()(tf.constant([0.0,1.0]), tf.constant([l1,l2])).numpy()
#         loss_wh = tf.keras.metrics.RootMeanSquaredError()(tf.constant([w,h]), tf.constant([w1,h1])).numpy()
#         #loss_wh = mse_custom(tf.constant([w,h], dtype = tf.double), tf.constant([w1,h1], dtype = tf.double))
#         losses.append(loss)
#         losses_wh.append(loss_wh)
#         print('[', l1,',', l2, '] loss:', loss)
#         #print('[', w1,',', h1, '] loss_wh:', loss_wh)
#         l1-=0.05
#         l2+=0.05
#         w1+=0.02
#         h1-=0.02
# print('mean =', np.mean(losses))


  
# #print('mean_wh =', np.mean(losses_wh))  
# # print(e[...,0])
# # print(pred[...,0])
# # print(e[...,1])
# # print(pred[...,1])
# pred = tf.Variable(pred)


# loss, loss2, loss3, cells_ob = class_loss_from_batch_bboi(pred, bboi_list)
# print(loss.numpy(),' ', loss2.numpy(),' ', loss3.numpy())


# q = B.reshape(e, (batch_size*dim_grid[0]*dim_grid[1],2))
# q_wh = np.reshape(wh, ((batch_size*dim_grid[0]*dim_grid[1],2)))

# l =  B.reshape(pred[...,1], (batch_size*dim_grid[0]*dim_grid[1],1))
# l_all =B.reshape(pred, (batch_size*dim_grid[0]*dim_grid[1],2))
# l_wh = np.reshape(pred_wh, (batch_size*dim_grid[0]*dim_grid[1],2))

# # m= np.concatenate((q,l), axis = -1)
# # q_wh_list = list(q[...,[1]]*q_wh)
# # l_wh_list = list(q[...,[1]]*l_wh)
# # q_list = list(q)
# # l_list = list(l)


# test = g*l

# tf.reduce_sum(tf.clip_by_value((-g*tf.math.log(g*l+ 1e-15)), 0.0, 15.))/tf.reduce_sum(g)

# tf.math.log(test+1e-15)

# print(-g*tf.math.log(test[...,1, tf.newaxis]+ 1e-15))
# print(tf.keras.losses.BinaryCrossentropy()(q,l_all))


# ROOT_DIR = file.ROOT_DIR   
# batch_size = 2
# file_points = "keypoints_bbox.json"
# file_points_0 = "keypoints_0.json"
# folder = '1_segments_2'
# folder_0 = '0_segments'
# dim = (224,224)
# shape = (dim[1], dim[0], 3) #shape = (dim[1], dim[0], 3)
# shape2 = shape
# n_weights = 11

# load_path = os.path.join(ROOT_DIR, 'fully_convolutional_epochs_0500', 'model_epoch_0345_min_val_keep.hdf5')
# #model_loaded = fconv_class(shape)
# #model_loaded.load_weights(load_path)


# oimg_list, total_weight = dl.generate_data_list(ROOT_DIR, folder, file_points, class_label = 1, stop = 0) #total weight since some original images have more subimages than others
# oimg_list_0, total_weight_0 = dl.generate_data_list(ROOT_DIR, folder_0, file_points_0,class_label = 0, stop = 0, add_unlabel = 1) #add unlabeled 0 images

# oimg_pair_list, remain, remain_0 = dl.generate_original_image_data_pair_list(oimg_list, oimg_list_0)

# load_dir = ROOT_DIR

# XY_train_paths = dl.load_kptarr_path_list(load_dir, 'train_images.txt')
# XY_valid_paths = dl.load_kptarr_path_list(load_dir, 'valid_images.txt')
# XY_test_paths = dl.load_kptarr_path_list(load_dir, 'test_images.txt')

# list_all_kptarr = dl.original_img_data_2_kptarr(oimg_list, 0)
# XY_train, list_rest_kptarr= dl.split_kptarr_list_by_file_names(list_all_kptarr[:1328], XY_train_paths)
# XY_valid, list_rest_kptarr = dl.split_kptarr_list_by_file_names(list_rest_kptarr, XY_valid_paths)
# XY_test, list_last_kptarr = dl.split_kptarr_list_by_file_names(list_rest_kptarr, XY_test_paths)

# batch_generator_01 = generate_batch(XY_test[:30], batch_size, seed=0)

# for batch in batch_generator_01:
# #for batch, batch_0 in zip(batch_generator_1, batch_generator_0):
#     #print('batch ', j)
#     #print(len(batch))
    
#     #X_batch_01, y_batch_01, names_01 = split_and_augment_batches_01(ROOT_DIR, folder_0, batch, batch_0, dim, seq)
#     X_batch_01, y_batch_01, y_batch_koi, y_batch_bboi, names_01 = split_aug_shuffle_batches_01(ROOT_DIR, batch, dim, augments())
#     X_train_norm = [X_a/255.0 for X_a in X_batch_01]
#     X_draw = draw_batch(X_batch_01, y_batch_koi, y_batch_bboi, names_01, color1 = (255,0,0), color2 =(0,0,255))
#     cllOb_list = []
#     for bboi in y_batch_bboi:
#         cllOb_list.append(cells_object.map_2_grid(dim, (7,7), bboi))
#     for X, cllOb in zip(X_draw, cllOb_list):
#         cllOb.draw_cells_on_image(X)
#     cllOb_list_2 = []
#     for koi in y_batch_koi:
#         cllOb_list_2.append(cells_object.map_2_grid_from_koi(dim, (7,7), koi))
#     for X, cllOb in zip(X_draw, cllOb_list_2):
#         cllOb.draw_cells_on_image(X, color = (0,255,255))
    
#     show_batch(X_draw, names_01)
    



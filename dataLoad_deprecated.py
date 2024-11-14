# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:09:53 2022

@author: i_bab
"""


import cv2 as cv
import numpy as np
from itertools import permutations, product
import os
import json
import file
import random
import glob
from helper import show_wait
from augment_utils import inverse_transform_matrix, augments
import matplotlib.pyplot as plt
import csv
from pathlib import Path
#imgaug
from sklearn.model_selection import train_test_split
from imgaug.augmentables import KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from loss_functions import cells_object



def split_at_index(some_list, some_index, seed = 0):
    if seed:
        random.Random(seed).shuffle(some_list)    
    return some_list[:some_index], some_list[some_index:]

def get_percent_1_to_0_of_pair_list(oimg_pair_list):
    w, w0, w1 = get_pair_weights(oimg_pair_list)
    percent_1_to_0 = w1/w
    return percent_1_to_0


def adjust_01_to_half(train_01, r_0, r_1, tol=0.01, seed = 0):
    percent_1_to_0 = get_percent_1_to_0_of_pair_list(train_01)
    
    if percent_1_to_0 > 0.5+tol:
        train_01.append(r_0.pop(0))
    elif percent_1_to_0 < 0.5-tol:
        train_01.append(r_1.pop(0))
    else:
        guess = random.Random(seed).randint(0,1)
        if guess:
            train_01.append(r_0.pop(0))
        else:
            train_01.append(r_1.pop(0))
            
def check_remain_size(r_1, r_0):
    n_r1, n_r0 = len(r_1), len(r_0)
    if not n_r1 or not n_r0:
        break_flag = 1
    else:
        break_flag = 0
    return break_flag

            
            
def train_test_valid_split(o_01, r_1, r_0, train_percent=0.75, test_valid_percent = 0.5, seed = 4):
    #ASSUMING SPLIT SIZES ARE valid <= test <= train
    index = int(train_percent*len(o_01))
    train_01, test_01 = split_at_index(o_01.copy(), index, seed)
    random.Random(seed).shuffle(r_1)
    random.Random(seed).shuffle(r_0)
    index1 = int(test_valid_percent*len(test_01))
    test_01, valid_01 = split_at_index(test_01, index1)
    
    
    percent_1_to_0 = get_percent_1_to_0_of_pair_list(train_01)
    print('train ', percent_1_to_0)
    percent_1_to_0_test = get_percent_1_to_0_of_pair_list(test_01)
    percent_1_to_0_val = get_percent_1_to_0_of_pair_list(valid_01)
    print('test ', percent_1_to_0_test)
    print('valid ', percent_1_to_0_val)
    
    train_to_test_valid_ratio = int(train_percent/(1-train_percent))
    
    test_to_valid_ratio = int(test_valid_percent/(1-test_valid_percent))
    
    break_flag = check_remain_size(r_1, r_0)
    
    i = 0
    while not break_flag:
        
        percent_1_to_0 = get_percent_1_to_0_of_pair_list(valid_01)
        #print('valid ', percent_1_to_0)
        adjust_01_to_half(valid_01, r_0, r_1, seed = i)
        i+=1
        break_flag = check_remain_size(r_1, r_0)
        #print('adjusted valid')
        if break_flag:
            break
        else:
            for j in range(test_to_valid_ratio):
                if break_flag:
                    break
                percent_1_to_0 = get_percent_1_to_0_of_pair_list(test_01)
                #print('test ',percent_1_to_0)
                adjust_01_to_half(test_01, r_0, r_1, seed = i)
                i+=1
                break_flag = check_remain_size(r_1, r_0)
                #print('adjusted test')
                if break_flag:
                        break
                else:
                    for i in range(train_to_test_valid_ratio):
                        percent_1_to_0 = get_percent_1_to_0_of_pair_list(train_01)
                        #print('train ',percent_1_to_0)
                        adjust_01_to_half(train_01, r_0, r_1, seed = i)
                        seed +=1
                        break_flag = check_remain_size(r_1, r_0)
                        #print('adjusted train')
                        if break_flag:
                                break 

    percent_1_to_0 = get_percent_1_to_0_of_pair_list(train_01)
    print('train ', percent_1_to_0)
    percent_1_to_0_test = get_percent_1_to_0_of_pair_list(test_01)
    percent_1_to_0_val = get_percent_1_to_0_of_pair_list(valid_01)
    print('test ', percent_1_to_0_test)
    print('valid ', percent_1_to_0_val)
    return train_01, test_01, valid_01

def generate_original_image_data_pair_list(oimg_list, oimg_list_0):
    oimg_pair_list = []
    copy1 = oimg_list.copy()
    copy2 = oimg_list_0.copy()
    remain = []
    remain_0 = []
    j_pop = -1
    for i in range(len(copy1)):
        name = copy1[i].name
        
        if j_pop >= 0:
            pass
            
        for j in range(len(copy2)):
            name2 = copy2[j].name
            if name == name2:
                oimg_pair_list.append(original_image_data_pair(copy1[i], copy2.pop(j)))
                j_pop = j
                i_append = -1
                break
            else:
                i_append = i
                j_pop = -1  
        if i_append>=0:
            remain.append(original_image_data_pair(copy1[i_append], None))
    
    for copy in copy2:
        remain_0.append(original_image_data_pair(None, copy))
        
    return oimg_pair_list, remain, remain_0

class original_image_data_pair():
    def __init__(self, oimg = None, oimg_0 = None):
        
        if oimg is not None and oimg_0 is not None:
            if oimg.name != oimg_0.name or oimg.path != oimg_0.path:
                print('NOT FROM SAME ORIGINAL IMAGE')
            else:    
                self.name = oimg.name
                self.path = oimg.path
                self.oimg_list = [oimg, oimg_0]
                self.subimg_list = oimg.subimg_list + oimg_0.subimg_list
                self.weight_1, self.weight_0 = oimg.weight, oimg_0.weight
                self.kptarr_list = oimg.kptarr_list + oimg_0.kptarr_list
                self.labels = [1]*len(oimg.kptarr_list) + [0]*len(oimg_0.kptarr_list)
        elif oimg is not None:
            self.name = oimg.name
            self.path = oimg.path
            self.oimg_list = [oimg]
            self.subimg_list = oimg.subimg_list
            self.weight_0 = 0
            self.weight_1 = oimg.weight
            self.kptarr_list = oimg.kptarr_list
            self.labels = [1]*len(oimg.kptarr_list)
        else:
            self.name = oimg_0.name
            self.path = oimg_0.path
            self.oimg_list = [oimg_0]
            self.subimg_list = oimg_0.subimg_list
            self.weight_0 = oimg_0.weight
            self.weight_1 = 0
            self.kptarr_list = oimg_0.kptarr_list
            self.labels = [0]*len(oimg_0.kptarr_list)
            
        self.weight = self.weight_1 + self.weight_0
        
        def __str__(self):
            return self.name
   
class original_image_data():
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(self.path) #works for windows, on linux returns full path
        self.subimg_list =[]
        self.kptarr_list =[]
        self.weight = 0
    
    def assignkptarr(self, kpt_arr):
        self.kptarr_list.append(kpt_arr)
        self.subimg_list.append(kpt_arr.name)
        self.weight +=1 #important for accurate data split
    
        
    def __str__(self):
        return self.name

class Original_image():
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(self.path) #works for windows, on linux returns full path
        self.region_list = []
        self.type_list = []
        self.weight = 0
        self.comp_map = None
        self.comp_pn = None
        
    def set_region_from_attr(self, region_attr, scale_x = 1., scale_y = 1.):
        rtype = region_attr['name']

        if rtype == 'point':
            region = Point.from_attr(region_attr, scale_x, scale_y) 
        elif rtype == 'rect':
            pass
        elif rtype == 'circle':
            region = Circle.from_attr(region_attr, scale_x, scale_y) 
        elif rtype == 'ellipse':
            region = Ellipse.from_attr(region_attr, scale_x, scale_y)
        
        self.region_list.append(region)
        self.type_list.append(str(region))
        self.weight+=1
        
    def set_comp_map(self, dim, dim_grid):
        map_list = []
        map_3d_list = []
        pn_list = []
        pn_3d_list =[]
        for region in self.region_list:
            region.set_map(dim, dim_grid)
            grad_map = region.get_map()
            map_list.append(grad_map)
            map_3d_list.append(grad_map[:,:,None])
            pn = region.get_pn()
            pn_list.append(pn)
            pn_3d_list.append(pn[:,:,None])
            
            
        self.comp_map = np.maximum.reduce(map_list)
        self.comp_map_3d = np.concatenate(map_3d_list, 2)
        self.comp_pn = np.maximum.reduce(pn_list)
        self.comp_pn_3d = np.concatenate(pn_3d_list, 2)
        
    
    def get_comp_map(self):
        return np.float32(self.comp_map)
    def get_comp_map_3d(self):
        return np.float32(self.comp_map_3d)
    
    def get_comp_pn(self):
        return np.float32(self.comp_pn)
    
    def get_comp_pn_3d(self):
        return np.float32(self.comp_pn_3d)
    
    def load_image(self, dim = None, flag = cv.IMREAD_COLOR):

        im = cv.imread(self.path, flag)
        
        if dim is not None:
            im = cv.resize(im, dim)  
 
        return im 
        
class Region_obj():
    def __init__(self, region_name):
        self.rtype = region_name
    
    @classmethod    
    def from_attr(cls, region_attr):
        rtype = region_attr['name']
        return cls(rtype)

    def __repr__(self):
        return self.rtype
    
    
class Point(Region_obj):
    def __init__(self, rtype, x, y):
        super().__init__(rtype)
        self.x, self.y = x, y
    
    @classmethod
    def from_attr(cls, region_attr):
        rtype = region_attr['name']
        x, y = region_attr['cx'], region_attr['cy']
        return cls(rtype, x, y)  

class Ellipse(Point):
    def __init__(self, rtype, x, y, rx, ry, theta):
        super().__init__(rtype, x, y)
        self.rx, self.ry = rx, ry
        self.theta = theta
        
        
    @classmethod
    def from_attr(cls, region_attr, scale_x = 1., scale_y = 1.):
        rtype = region_attr['name']
        x, y = region_attr['cx'], region_attr['cy']
        rx, ry = region_attr['rx'], region_attr['ry']
        theta =  region_attr['theta']
        
        x = scale_x*x
        y = scale_y*y
        
        #here we scale rx and ry proportionately to how far they x components and y components
        #we have to divide by the unscaled sum components to normalize
        rx = (np.abs(np.cos(theta)*scale_x) + np.abs(np.sin(theta)*scale_y))/(np.abs(np.cos(theta)) + np.abs(np.sin(theta)))*rx
        ry = (np.abs(np.sin(theta)*scale_x) + np.abs(np.cos(theta)*scale_y))/(np.abs(np.cos(theta)) + np.abs(np.sin(theta)))*ry
        
        #here we find the new angle by transforming a unit circle coordinate and computing the new angle
        tscale = np.array([[scale_x,0.], [0., scale_y]])
        angle_components = np.array([[np.cos(theta)], [np.sin(theta)]])
        scaled_components = tscale@angle_components
        theta = np.arctan2(scaled_components[1,0], scaled_components[0,0] )
        return cls(rtype, x, y, rx, ry, theta)
    
    def set_map(self, dim, dim_grid):
        x,y = self.x, self.y
        rx, ry = self.rx, self.ry
        theta =  self.theta    

        rx_nrm, ry_nrm = rx/dim[0], ry/dim[1]
        r_min = np.min([rx_nrm, ry_nrm])
        r_base= 0.04 

        if r_min < r_base:
            factor = r_base/r_min
            rx_nrm *=factor
            ry_nrm *=factor
        #Define origin center (center grid)    
        cobj1 = cells_object.map_2_grid_from_xy(dim, dim_grid, (dim[0]/2,dim[1]/2))
        _, r_map1 = cobj1.load_radial_map()
        center = cobj1.grid_centers_ji[0]
        o_x, o_y = center[1], center[0]
        #Define region center
        cobj2 = cells_object.map_2_grid_from_xy(dim, dim_grid, (x,y))
        _, r_map_alt = cobj2.load_radial_map()
        center2 = cobj2.grid_centers_ji[0]
        x_grid, y_grid = center2[1], center2[0]

        #Perform image transformations to generate final transformation matrix
        t12 = np.array([[1.,0.,-o_x], [0., 1., -o_y], [0., 0., 1.]]) #if we only translate origin without reflecting y-axis
        scale_matrix = np.array([[2*rx_nrm , 0.0, 0.], [0.0, 2*ry_nrm, 0.],[0., 0., 1.]])
        
        #flip = np.array([[1.,0.,0], [0., 1., 0], [0., 0., 1.]])
        rot_mat = np.array([[np.cos(theta),-np.sin(theta),0], [np.sin(theta), np.cos(theta), 0], [0., 0., 1.]])
        tx, ty = x_grid-o_x, y_grid-o_y
        trans_mat = np.array([[1.,0.,tx], [0., 1., ty], [0., 0., 1.]])
        t21 = inverse_transform_matrix(t12)
        #Neglect bottom row for openCV function
        M = (t21@trans_mat@rot_mat@scale_matrix@t12)[:2,:]
        #Apply transformations
        r_map1_warped = cv.warpAffine(r_map1, M, dim_grid)
        
        #scale = 30
        #show_wait(r_map1_warped,scale, interpolation = cv.INTER_NEAREST)
        self.x_grid, self.y_grid = x_grid, y_grid
        posneg = np.zeros(dim_grid)
        posneg[y_grid,  x_grid] = 1.
        self.pn = posneg
        self.map = r_map1_warped
        #self.map = r_map_alt
        
    def get_map(self):
        return self.map
    
    def get_pn(self):
        return self.pn
    
    # def get_vrtx(self):
        
    #     vertex = 

class Circle(Ellipse):
    def __init__(self, rtype, x, y, rx, ry):
        super().__init__(rtype, x, y, rx, ry, 0)
    @classmethod
    def from_attr(cls, region_attr, scale_x = 1., scale_y = 1.): 
        rtype = region_attr['name']
        x, y = region_attr['cx'], region_attr['cy']
        rx, ry = region_attr['r'], region_attr['r']
        return cls(rtype, scale_x*x, scale_y*y, scale_x*rx, scale_y*ry)


class bbox(Point):
    pass
    
    
class keypointarray():
    def __init__(self, n_pts ,path, label):
        self.n_pts= n_pts
        self.array = np.zeros((n_pts, 2))
        self.path = path
        self.name= os.path.basename(self.path) #works for windows, on linux returns full path
        self.original = self.name[:6]
        self.h, self.w = None, None
        self.koi = None
        self.bbox = np.zeros((4))
        self.bboi = None
        self.new_array = np.zeros((n_pts, 2))
        self.new_bbox = np.zeros((4))
        self.new_h, self.new_w = None, None 
        self.label = label
        self.bbox_center_added = 0
    
    def assign_kpts(self, i, xi, yi):
        if i < self.n_pts:
            self.array[i,0] = xi
            self.array[i,1] = yi
        else:
            temp = np.zeros((self.n_pts+1, 2))
            temp[:self.n_pts, :2]= self.array
            temp[i, 0] = xi
            temp[i, 1] = yi
            self.array = temp
            self.n_pts+=1
            
            
    def assign_bbox(self, x, y, w, h):
        self.bbox[0], self.bbox[1] = x, y
        self.bbox[2], self.bbox[3] = w, h
        
    def add_bbox_center_as_kpt(self):
        x, y = self.bbox[0], self.bbox[1]
        w, h = self.bbox[2], self.bbox[3]
        x1, y1 = x,y
        x2, y2 = x+w, y+h
        x_c, y_c = int((x2+x1)/2.0), int((y2+y1)/2.0)
        pt  = np.array([[x_c, y_c]])
        
        
        self.array = np.concatenate((self.array, pt), axis = 0)
        self.new_array = np.concatenate((self.new_array, np.array([[0., 0.]])), axis = 0)
        self.bbox_center_added = 1
            
    def __str__(self):
        return self.name
    
    def loadimage(self, dim, flag = cv.IMREAD_COLOR):
        im = cv.imread(self.path, flag)
        
        self.h, self.w = np.shape(im)[0], np.shape(im)[1]
        
        if np.sum(self.bbox) == np.inf:
            self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3] = 0, 0, self.w, self.h
        if not self.bbox_center_added:
            self.add_bbox_center_as_kpt()
        
        if dim[0] != self.w or dim[1] != self.h:
            scale_w, scale_h = dim[0]/self.w, dim[1]/self.h
            
            self.new_array = np.concatenate((self.array[:,0,None] *scale_w,self.array[:,1, None] *scale_h), 1)
            # self.new_array[:,0] = self.array[:,0] *scale_w
            # self.new_array[:,1] = self.array[:,1] *scale_h
            
            self.new_bbox[0] = self.bbox[0] *scale_w
            self.new_bbox[1] = self.bbox[1] *scale_h
            self.new_bbox[2] = self.bbox[2] *scale_w
            self.new_bbox[3] = self.bbox[3] *scale_h
        
        else:
            self.new_array = np.concatenate((self.array[:,0, None],self.array[:,1, None]), 1)
            # self.new_array[:,0] = self.array[:,0]
            # self.new_array[:,1] = self.array[:,1]
            
            self.new_bbox[0] = self.bbox[0]
            self.new_bbox[1] = self.bbox[1]
            self.new_bbox[2] = self.bbox[2]
            self.new_bbox[3] = self.bbox[3]
        
        im = cv.resize(im, dim)  
        self.new_h, self.new_w = dim[1], dim[0]
        
        
        if len(np.shape(im))>2:
            shape = (self.new_h, self.new_w, np.shape(im)[2])
        else:
            shape = (self.new_h, self.new_w)
        
        self.koi = KeypointsOnImage.from_xy_array(self.new_array, shape)
        self.bboi = BoundingBoxesOnImage([BoundingBox(self.new_bbox[0], self.new_bbox[1], self.new_bbox[0]+self.new_bbox[2], self.new_bbox[1]+self.new_bbox[3])], shape)
        
        return im


def save_resized(ROOT_DIR, src_folder, dst_folder, size_src = None, size_dst = None):
    src_path = os.path.join(ROOT_DIR, src_folder)
    dst_path = os.path.join(ROOT_DIR, dst_folder)
    
    for filename in os.listdir(src_path):
        if filename[-3:] == 'bmp':
            im_path = os.path.join(src_path, filename)
            #print(im_path,'   ', filename)
            im = cv.imread(im_path)
            #print(im.shape)
            write_path = os.path.join(dst_path, filename)
            #print(write_path,'   ', filename)
            im = cv.resize(im, size_dst)
            cv.imwrite(write_path, im)
            
  
        #cv.imread()
# ROOT_DIR = Path(file.ROOT_DIR).parent        
# src_folder= '1'
# dst_folder = '1_224x224'
# size_src = (4056,3040)
# size_dst = (224,224)
# save_resized(ROOT_DIR, src_folder, dst_folder, size_src, size_dst)
    


def generate_data_list(data_path, file_points_path, scale_x = 1., scale_y = 1.):
    #the file_points uses the orignal image dimensions whereas the image folder
    #can use downsampled versions for which the scale_x,y must be adjusted
    with open(os.path.join(data_path, file_points_path), 'r') as f:
        via = json.load(f)    
    
    oimg_list = []    
    for fid in via['_via_img_metadata']:
        
        file_name = via['_via_img_metadata'][fid]['filename']
        file_path = os.path.join(data_path, file_name)

        if not os.path.isfile(file_path):
            print('File not found! %s' %(file_path))
            continue

        oim = Original_image(file_path)
        for region in via['_via_img_metadata'][fid]['regions']:
            oim.set_region_from_attr(region['shape_attributes'], scale_x, scale_y)

        oimg_list.append(oim)

    return oimg_list

def get_pair_weights(oimg_pair_list):
    w0, w1 = 0, 0
    for pair in oimg_pair_list:
        w0 += pair.weight_0
        w1 += pair.weight_1
    total_w = w0+w1
    return total_w, w0, w1
    
    
def shuffle_and_split_original_img_data(oimg_list, total_weight, train_percent = 0.8, tol = 0.001, seed = 4):
    if seed:
        random.Random(seed).shuffle(oimg_list)
    XY_train, XY_test = oimg_list[0:int(train_percent*len(oimg_list))], oimg_list[int(train_percent*len(oimg_list)):]
    #len(XY_train)
    #len(XY_test)
    weight = 0
    for oimg_data in XY_train:
        weight+=oimg_data.weight

    diff = weight/total_weight - train_percent
    #print('original diff is ', diff)
    #print('orignal length of XY_train', len(XY_train), ' and length of XY_test', len(XY_test) )
    #print(diff)
    #print(tol)
    #i=0
    while np.abs(diff)>tol:
        
        if(diff<0):
            weight += XY_test[0].weight
            XY_train.append(XY_test.pop(0))
            diff = weight/total_weight - train_percent
            #print('adding element to XY_train')
            #print(diff)
        elif(diff>0):
            weight -= XY_train[-1].weight
            XY_test.append(XY_train.pop(-1))
            diff = weight/total_weight - train_percent
            #print('removing element from XY_train')
            #print(diff)
        
        #print('length of XY_train', len(XY_train), 'length of XY_test', len(XY_test) )
        # if i > 100:
        #     break
        # i+=1
    return XY_train, XY_test

def adjust_number(number_string, length = 4):
    if type(number_string) is not str:
        number_string = str(number_string)
    if len(number_string)<length:
           return adjust_number('0'+number_string)
    else:
        return number_string

def original_img_data_2_kptarr(XY_train_oimg, seed = 4):
    XY_train =[]
    for oim_data in XY_train_oimg: #split oimg data objects into there subimage components
        for kptarr in oim_data.kptarr_list:
            XY_train.append(kptarr)
    if seed:
        random.Random(seed).shuffle(XY_train)
    return XY_train

def make_clr(grey_im):
    shape = np.shape(grey_im)
    if len(shape)>2:
        if shape[2] == 1:
            return np.concatenate((grey_im, grey_im, grey_im), axis = 2)
        elif shape[2] == 3:
            return grey_im
        else:
            print('error')
    elif len(shape) ==2:
        return make_clr(grey_im[...,None])
        
    
class generate_batch():
    
    def __init__(self, XY_train, batch_size, seed = 4):
        #print('constructing class')
        if seed:
            random.Random(seed).shuffle(XY_train)
        self.XY_train = XY_train
        #print('type of XY_train = ', type(self.XY_train))
        self.batch_size = batch_size
        self.steps = int(np.ceil(len(XY_train)/batch_size))
        self.extra_step_size = len(XY_train)%batch_size
        self.index = 0
    def __next__(self):
        #calls next item (batch) in sequence
        if self.index < self.steps:
            if self.index == self.steps-1 and self.extra_step_size:
                batch = self.XY_train[self.index*self.batch_size:self.index*self.batch_size + self.extra_step_size]
                #print('last batch, batch index = ', self.index)
            else:
                #print('normal batch, batch index = ', self.index)
                batch = self.XY_train[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index +=1
            return batch
        else:
            raise StopIteration
    
    def __iter__(self):
        #returns iterator object
        return self

    

    
def split_XY(batch_of_kptarr, dim, classed = 0, flag = cv.IMREAD_COLOR):
    X_batch = []
    y_batch_koi = []
    y_batch_bboi = []
    batch_names = []
    y_batch_label = []
    for kptarr in batch_of_kptarr:
        X_batch.append(kptarr.loadimage(dim, flag))#note call this before the labels (y_batch) because this scales both images and kpts 
        y_batch_koi.append(kptarr.koi)  #and initializes 'Keypoints on Image' object from kpts
        y_batch_bboi.append(kptarr.bboi)
        batch_names.append(kptarr.path)
        y_batch_label.append(kptarr.label)
    if classed:
        return X_batch, y_batch_koi, y_batch_bboi, y_batch_label, batch_names 
    else:
        return X_batch, y_batch_koi, y_batch_bboi, batch_names   

def show_batch(X_batch, batch_names = None, size = 40.):
    #print('ShowingBatch')    
    _, axs = plt.subplots(1, len(X_batch), figsize=(40, 40))
    if len(X_batch)>1:
        axs = axs.flatten()
    else:
        axs = [axs]
    if batch_names is not None:
        for img, name, ax in zip(X_batch, batch_names, axs):
            h,w = np.shape(img)[0], np.shape(img)[1]
            img = img[...,::-1]
            ax.text(w/3, h/6, name[-13:-4], size = size, weight = 'bold', color = 'r')
            ax.imshow(img)
    else:
        for img, ax in zip(X_batch, axs):
            h,w = np.shape(img)[0], np.shape(img)[1]
            img = img[...,::-1]
            ax.text(w/3, h/6, '', size = size, weight = 'bold', color = 'r')
            ax.imshow(img)

    plt.show()
    
    
def show_batch_v2(im_batch, batch_names = None):
    #print('ShowingBatch')    
    _, axs = plt.subplots(1, len(im_batch), figsize=(40, 40))
    if len(im_batch)>1:
        axs = axs.flatten()
    else:
        axs = [axs]
    if batch_names is not None:
        for img, name, ax in zip(im_batch, batch_names, axs):
            h,w = np.shape(img)[0], np.shape(img)[1]
            img = img[...,::-1]
            ax.text(w/3, h/6, name[-13:-4], size = 70., weight = 'bold', color = 'r')
            ax.imshow(img)
    else:
        for img, ax in zip(im_batch, axs):
            h,w = np.shape(img)[0], np.shape(img)[1]
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            ax.text(w/3, h/6, '', size = 70., weight = 'bold', color = 'r')
            ax.imshow(img)

    plt.show()
    
   
def draw_batch(X_batch, y_batch_koi, y_batch_bboi, batch_names, color1 = (255,0,0), color2 =(0,0,255)):
    #print('drawingBatch')    
    X_draw = []
    for img, pts, bboi, name in zip(X_batch, y_batch_koi, y_batch_bboi, batch_names):
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #bbox = bboi.clip_out_of_image()
        #print(name)
        #print(bboi.to_xyxy_array())
        img = bboi.draw_on_image(img, color1, size = 5)
        img = pts.draw_on_image(img, color2, size=5)
        
        X_draw.append(img)
    return X_draw

# def cvt_pred(y_pred, shape):
#     #y_pred is a batch_size X 2 tensor
#     #y_pred[0] is a the first element which itself is batch_size X 4
#     #y_pred[1] is a the second element which itself is batch_size X 2
#     dim = (shape[1], shape[0])
#     bbox_crds = y_pred[0]/np.array([dim[0], dim[1], dim[0], dim[1]])
#     bboi_list = []
#     for i in range(len(bbox_crds)):
#         bboi_list.append(BoundingBoxesOnImage.from_xyxy_array(bbox_crds[i], shape))
#     return bboi_list
    #cos_sin = y_pred[1]
    #bbox_list = [bbox/np.array([dim[0], dim[1], dim[0], dim[1]]) for bbox in bbox_norm_list]
    
    
    #cos_list, sin_list = [cos, sin in zip(cos_sin_list[0])]
    
def load_batch_from_file(ROOT_DIR, folder_0, batch_names, dim):
    X_batch = []
    full_names = []
    for name in batch_names:
        file_path = os.path.join(ROOT_DIR, folder_0, name)
        #print(file_path)
        img = cv.imread(file_path)
        img = cv.resize(img, dim)
        X_batch.append(img)
        full_names.append(file_path)
    return X_batch, full_names
def load_class_0(ROOT_DIR, folder_0):
    XY_0_names = os.listdir(os.path.join(ROOT_DIR, folder_0))
    return XY_0_names


def mix_and_augment_batches_01(ROOT_DIR, folder_0, batch, batch_0, dim, seq = None, seed = 4):
    X_batch, _, _, batch_names = split_XY(batch, dim)
    y_batch = [1]*len(batch)
    X_batch_0, batch_names_0 = load_batch_from_file(ROOT_DIR, folder_0, batch_0, dim)
    y_batch_0 = [0]*len(batch_names_0)
    
    X_batch_01 = X_batch_0 + X_batch
    if seq is not None:
        X_batch_01 = seq(images = X_batch_01)
    
    y_batch_01 = y_batch_0 + y_batch
    names_01 =   batch_names_0 + batch_names
    if seed:
        random.Random(seed).shuffle(X_batch_01)
        random.Random(seed).shuffle(y_batch_01)
        random.Random(seed).shuffle(names_01)
    
    return X_batch_01, y_batch_01, names_01

def split_aug_shuffle_batches_01(ROOT_DIR, batch_of_kptarr, dim, seq = None, seed = 4):
    
    X_batch_01, y_batch_koi, y_batch_bboi, y_batch_01, names_01 = split_XY(batch_of_kptarr, dim, classed = 1)
    y_batch_bboi_clip = []
    if seq is not None:
        X_batch_01, y_batch_koi, y_batch_bboi = seq(images = X_batch_01, keypoints = y_batch_koi, bounding_boxes = y_batch_bboi)
        
    for bboi in y_batch_bboi:
        y_batch_bboi_clip.append(bboi.clip_out_of_image())
        
    if seed:
        random.Random(seed).shuffle(X_batch_01)
        random.Random(seed).shuffle(y_batch_01)
        random.Random(seed).shuffle(y_batch_koi)
        random.Random(seed).shuffle(y_batch_bboi_clip)
        random.Random(seed).shuffle(names_01)
    
    return X_batch_01, y_batch_01, y_batch_koi, y_batch_bboi_clip, names_01


def save_kptarr_list(XY_train, save_dir, file_name):
    with open(os.path.join(save_dir, file_name), 'w', newline ='') as f:
        writer = csv.writer(f)
        for kpt_arr in XY_train:
            row = [kpt_arr.path]
            writer.writerow(row)

def load_kptarr_path_list(load_dir, file_name):
    names = []
    with open(os.path.join(load_dir, file_name), newline ='') as f:
        reader = csv.reader(f)
        for row in reader:
            names.extend(row)
    return names

def split_kptarr_list_by_file_names(list_all_kptarr, XY_train_paths):
    XY_train = []
    for path in XY_train_paths:
        n = len(list_all_kptarr)
        i = 0
        while n and i < n:
            if path == list_all_kptarr[i].path:
                XY_train.append(list_all_kptarr.pop(i))
                n = len(list_all_kptarr)
            else:
                i+=1
    return XY_train, list_all_kptarr
def scale_map(im_map):
    max_map = np.max(im_map, axis = (0,1), keepdims=True) #get maximum of each channel
    min_map = np.min(im_map, axis = (0,1), keepdims=True)
    im_map = (im_map-min_map)/(max_map - min_map + 1e-15)

    return im_map

def scale_map_list_3D(map_list):
    scaled_map_list = []
    for mapi in map_list:
        scaled = scale_map(mapi)
        scaled_map_list.append(scaled)
    return scaled_map_list
    

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

def flatten_map(im_map, invert = 1):
    #print(np.shape(im_map))
    max_map = np.max(im_map, axis = (0,1), keepdims=True) #get maximum of each channel
    min_map = np.min(im_map, axis = (0,1), keepdims=True)
    im_act = im_map #for activations later
    im_map = (im_map-min_map)/(max_map - min_map + 1e-15)
    im_map = np.maximum.reduce(im_map, axis=2)
    if invert:
        im_map = 1 - im_map

    im_act = np.where(im_act < max_map, 0.,1.) #set max element of each channel to 1 and all else to zero.
    im_act = np.maximum.reduce(im_act, axis=2)
    im_act= np.float32(im_act) 
    
    return im_map, im_act
def image_list_from_batch(batch, dim = None):
    im_list =[]
    for oim in batch:
        im_list.append(oim.load_image(dim)) #ADD [...,::-1] TO MAKE IT RGB 
    return im_list

def name_list_from_batch(batch):
    name_list = []
    for oim in batch:
        name_list.append(oim.name)
    return name_list

def save_images_from_list(im_list, name_list, save_dir, add = '', BGR = 1):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for im, name in zip(im_list, name_list):
        new_name = name[:-4] + add + name[-4:]
        save_path = os.path.join(save_dir, new_name) 
        if not BGR:
            im = im[::-1]
        cv.imwrite(save_path, im)


    
from imgaug.augmentables.heatmaps import HeatmapsOnImage
def map_list_from_batch(batch, dim, dim_grid):
    map_list = []
    for oim in batch:
        oim.set_comp_map(dim, dim_grid)
        map_list.append(oim.get_comp_map_3d())
    return map_list

def map2heatmap_list(map_list, dim):
    heatmap_list = []
    for mapi in map_list:
        heatmap_list.append(HeatmapsOnImage(mapi, dim))
    return heatmap_list

def heatmap2map_list(heatmap_list):
    map_list = []
    for heatmap in heatmap_list:
        map_list.append(heatmap.get_arr())
    return map_list

def flat_map_list(map_list):
    flat_list = []
    for map1 in map_list:
        flat, _ = flatten_map(map1)
        flat_list.append(flat)
  
    return flat_list            

def loss_from_map(pred, flat_arr, weight_arr):
    
    pred_shape = pred.shape
    batch_size =pred_shape[0]
    w_pred, h_pred =  pred_shape[2], pred_shape[1]
    
    #pred =  np.reshape(pred[...,1], (batch_size*w_pred*h_pred,1)) #B.reshape
    pred =  np.reshape(pred, (batch_size*w_pred*h_pred,1))         #B.reshape
    
    weights = np.ndarray.astype(weight_arr, pred.dtype)#tf.cast
    weights =  np.reshape(weights, (batch_size*w_pred*h_pred,1))#B.reshape
    
    truth = np.ndarray.astype(flat_arr, pred.dtype)#tf.cast
    truth =  np.reshape(truth, (batch_size*w_pred*h_pred,1))#B.reshape
    
    loss = np.mean(weights*np.sqrt((pred-truth)*(pred-truth)+1e-15), axis = 0) #tf.reduce_mean
    return loss

# def loss_from_map_tensor(pred, flat_arr, weight_arr):
#     pred_shape = pred.shape
#     w_pred, h_pred =  pred_shape[2], pred_shape[1]
    
#     #pred =  np.reshape(pred[...,1], (batch_size*w_pred*h_pred,1)) #B.reshape
#     pred =  B.reshape(pred, (batch_size*w_pred*h_pred,1))         #B.reshape
    
#     weights = tf.cast(weight_arr, pred.dtype)#tf.cast
#     weights =  B.reshape(weights, (batch_size*w_pred*h_pred,1))#B.reshape
    
#     truth = tf.cast(flat_arr, pred.dtype)#tf.cast
#     truth =  B.reshape(truth, (batch_size*w_pred*h_pred,1))#B.reshape
    
#     loss = tf.reduce_mean(weights*np.sqrt((pred-truth)*(pred-truth)+1e-15), axis = 0) #tf.reduce_mean
#     return loss
def process_batch(batch, dim, dim_grid, seq = None, lo_val = 0.1):
    map_list = map_list_from_batch(batch, dim, dim_grid)
    if seq is not None:
        _, map_aug_list = seq(images = map_list, heatmaps = map_list)
    else:
        map_aug_list = map_list

    flat_list1 = flat_map_list(map_aug_list)
    flat_arr1 = np.array(flat_list1)

    weight_arr1=  (1.-lo_val)*(1-flat_arr1) + lo_val #scale so that black cell weights of inverse map are not zero

    return flat_arr1, weight_arr1, seq

def weight_list_3D(map_list, lo_val = 0.0):
    weight_list = []
    for mapi in map_list:
        weight_map=  (1.-lo_val)*(1-mapi) + lo_val #scale so that black cell weights of inverse map are not zero
        weight_list.append(weight_map)
    return weight_list

def process_batch_3D(batch, dim, dim_grid, seq = None, lo_val = 0.0):
    map_list = map_list_from_batch(batch, dim, dim_grid)
    if seq is not None:
        heatmap_aug_list = seq(heatmaps = map2heatmap_list(map_list, dim))
        map_aug_list = heatmap2map_list(heatmap_aug_list)
    else:
        map_aug_list = map_list
    
    map_aug_list = scale_map_list_3D(map_aug_list)
    weight_list = weight_list_3D(map_aug_list, lo_val)
        
    return map_aug_list, weight_list, seq


    
def act_list_3D(map_list):
    act_list = []
    
    for mapi in map_list:
        max_map = np.max(mapi, axis = (0,1), keepdims=True)
        im_act = np.where(mapi < max_map, 0.,1.)
        
        act_list.append(np.float32(im_act))
    
    return act_list

###_______________NEWEST CODE START__________________________________############

####_______________SAVE 5 BEST START__________________________________############
'''last_n = 3
min_val_loss_list = [np.inf]
min_name_list = ['inf']
all_losses = []
for i in range(10):
    name = str(i)
    val_loss =  np.random.rand()
    all_losses.append(val_loss)
    print(val_loss)
    
    #np.argsort(min_val_loss_list)
    max_idx_sorted = np.flip(np.argsort(min_val_loss_list))
    max_idx = max_idx_sorted[0]
    max_min_val_loss = min_val_loss_list[max_idx]
    if val_loss <= max_min_val_loss:
        min_val_loss_list.append(val_loss)
        min_name_list.append(name)
        if len(min_val_loss_list)  > last_n:
            min_val_loss_list.pop(max_idx)
            delete_name = min_name_list.pop(max_idx)
            print('deleting file name = ', delete_name)

print(min_val_loss_list)
all_losses_sorted = all_losses.copy()
all_losses_sorted.sort()
print(all_losses_sorted)'''
####_______________SAVE 5 BEST END__________________________________############

####_______________INIT DATA START__________________________________############
'''
import tensorflow as tf
from loss_functions import tensor_sample_thresh_by_idx
from tensorflow.keras import backend as B
ROOT_DIR = Path(file.ROOT_DIR).parent

folder = '1_448x448'
data_path = os.path.join(ROOT_DIR, folder)
folder_points = '1'
file_points = "plant_centers_sifted_FINAL.json"
points_path = os.path.join(ROOT_DIR, folder_points, file_points)
Wo, Ho = 4056, 3040
W, H = 448, 448
scale_x, scale_y = W/Wo, H/Ho
oimg_list = generate_data_list(data_path, points_path, scale_x, scale_y)
X_train, X_test, _, _ =  train_test_split(oimg_list, oimg_list, test_size = 0.4, random_state = 42)
X_test, X_valid, _, _ = train_test_split(X_test, X_test, test_size = 0.5, random_state = 42)

dim = (H,W)


seq1 = augments()
#seq1.localize_random_state_()
# seq2 = seq1.deepcopy()
batch_size = 4
batch_gen = generate_batch(X_train, batch_size)
scale = 15
    
dim_grid1, dim_grid2 = (3,3),(28,28)

H_grid, W_grid = dim_grid1[0], dim_grid1[1]
#pred = np.random.rand(batch_size,H_grid, W_grid)
pred= tf.random.uniform((batch_size,H_grid, W_grid))
pred_numpy = pred.numpy()
'''
####_______________INIT DATA END__________________________________############
###-------------------------------------BLOCK 3 DEBUGGING CODE START----------------------------####
'''
flag =0
i= 0
for batch in batch_gen:
    
    if i+1:
        im_list = image_list_from_batch(batch, dim)
        name_list = name_list_from_batch(batch)
        
        im_aug_list = im_list
        #predict data
    
        map_list1, weight_list1, _ = process_batch_3D(batch, dim, dim_grid1, None)
        map_list2, weight_list2, _ = process_batch_3D(batch, dim, dim_grid2, None)
        
        act_list1 = act_list_3D(map_list1)
        act_list2 = act_list_3D(map_list2)
    
        show_batch(im_aug_list, name_list)
        # show_batch(flat_map_list_v2(map_list1), name_list)
        show_batch(flat_map_list_v2(map_list2.copy()), name_list)
        
        temp = seq1.deepcopy()
             
        #predict data
        map_list1, weight_list1, _ = process_batch_3D(batch, dim, dim_grid1, temp.deepcopy())
        map_list2, weight_list2, _ = process_batch_3D(batch, dim, dim_grid2, temp.deepcopy())
        act_list1 = act_list_3D(map_list1)
        act_list2 = act_list_3D(map_list2)
        
        im_aug_list = seq1(images = im_list)
        #break
        #weight_list2 = temp.deepcopy()(images = weight_list2)
        #map_list2 = temp.deepcopy()(images = map_list2)
        
        
        for mapi in map_list1:
            if mapi.shape[2] ==4:
                show_batch(im_aug_list, name_list)

                show_batch(flat_map_list_v2(map_list1), name_list)
                show_batch(flat_map_list_v2(map_list2.copy()), name_list)
                show_batch(flat_map_list_v2(weight_list2.copy(), 1), name_list)
                show_batch(flat_map_list_v2(act_list2.copy()), name_list)
                
                
                
                flag = 1
                break
        if flag:
            break
        
            
    i+=1
for mapi in map_list2:
    break
flat_arr2 = np.array(flat_map_list_v2(map_list2.copy()))
flat_arr2.shape
# for mapi, acti, weighti in zip(map_list1,act_list1,weight_list1):
#     act_layers = len(mapi[0,0])
#     if act_layers ==2:
#         flag = 1
#         break
#     for i in range(len(mapi[0,0])):
#         show_wait(np.concatenate(flat_weight_list2, axis = 1), scale, '28x28', 1, 1, 0, 0, interpolation = cv.INTER_NEAREST)
'''
###-------------------------------------BLOCK 3 DEBUGGING CODE END------------------------------####
'''
  
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

'''

#--------POSNEG LOSS CALCULATIONS BEGIN----------------------#
'''shape_pred = np.shape(pred)
vect_pred = np.reshape(pred, (shape_pred[0]*shape_pred[1],1))
flat_map_list1 = flat_map_list_v2(map_list1)
flat_act_list1 = flat_map_list_v2(act_list1)
for i in range(len(flat_map_list1)):
     show_wait(np.concatenate([flat_map_list1[i], flat_act_list1[i]], axis = 1), scale, '28x28', 1, 1, 0, 0, interpolation = cv.INTER_NEAREST)
flat_mapi = flat_map_list1[i]
flat_acti = flat_act_list1[i]
shape_mapi = np.shape(flat_mapi)
shape_acti = np.shape(flat_acti)
vect_mapi = np.reshape(flat_mapi, (shape_mapi[0]*shape_mapi[1],1))
vect_acti = np.reshape(flat_acti, (shape_acti[0]*shape_acti[1],1))

vect_mapi_inv = 1 - vect_mapi
vect_acti_inv = 1 - vect_acti
pos_loss = np.sum(np.clip((-vect_acti*vect_mapi*np.log(vect_acti*vect_pred+ 1e-15)), 0.0, 15.))/np.sum(vect_acti)
neg_loss = np.sum(np.clip((-vect_acti_inv*vect_mapi_inv*np.log(vect_acti_inv*vect_pred + 1e-15)), 0.0, 15.))/np.sum(vect_acti_inv)

def pos_neg_loss(pred, map_list, act_list = None):
    
    if act_list is None:
        act_list = act_list_3D(map_list)
     
    shape_pred = pred.shape
    batch_size = shape_pred[0]
    vect_pred = np.reshape(pred, (batch_size*shape_pred[1]*shape_pred[2],1))
    flat_map_list1 = flat_map_list_v2(map_list)
    flat_act_list1 = flat_map_list_v2(act_list)
    # for i in range(len(flat_map_list1)):
    #      show_wait(np.concatenate([flat_map_list1[i], flat_act_list1[i]], axis = 1), scale, '28x28', 1, 1, 0, 0, interpolation = cv.INTER_NEAREST)

    flat_map_arr1 = np.asarray(flat_map_list1)
    flat_act_arr1 = np.asarray(flat_act_list1)
    
    shape_map = np.shape(flat_map_arr1)
    shape_act = np.shape(flat_act_arr1)
    
    vect_map1 = np.reshape(flat_map_arr1, (shape_map[0]*shape_map[1]*shape_map[2],1) )
    vect_act1 = np.reshape(flat_act_arr1, (shape_act[0]*shape_act[1]*shape_act[2],1) )

    vect_map1_inv = 1 - vect_map1
    vect_act1_inv = 1 - vect_act1

    # flat_mapi = flat_map_list1[i]
    # flat_acti = flat_act_list1[i]

    # shape_mapi = np.shape(flat_mapi)
    # shape_acti = np.shape(flat_acti)
    # vect_mapi = np.reshape(flat_mapi, (shape_mapi[0]*shape_mapi[1],1))
    # vect_acti = np.reshape(flat_acti, (shape_acti[0]*shape_acti[1],1))

    # vect_mapi_inv = 1 - vect_mapi
    # vect_acti_inv = 1 - vect_acti

    # pos_loss = np.sum(np.clip((-vect_acti*vect_mapi*np.log(vect_acti*vect_pred+ 1e-15)), 0.0, 15.))/np.sum(vect_acti)
    # neg_loss = np.sum(np.clip((-vect_acti_inv*vect_mapi_inv*np.log(vect_acti_inv*vect_pred + 1e-15)), 0.0, 15.))/np.sum(vect_acti_inv)

    pos_loss = np.sum(np.clip((-vect_act1*vect_map1*np.log(vect_act1*vect_pred+ 1e-15)), 0.0, 15.))/np.sum(vect_act1)
    neg_loss = np.sum(np.clip((-vect_act1_inv*vect_map1_inv*np.log(vect_act1_inv*(1-vect_pred) + 1e-15)), 0.0, 15.))/np.sum(vect_act1_inv)

    return pos_loss, neg_loss


pos_neg_loss(pred_numpy, map_list1, act_list1)
    
def tensor_pos_neg_loss(pred, map_list, act_list = None):
    
    if act_list is None:
        act_list = act_list_3D(map_list)
    shape_pred = pred.shape
    batch_size = shape_pred[0]
    vect_pred = B.reshape(pred, (batch_size*shape_pred[1]*shape_pred[2],1))
    flat_map_list1 = flat_map_list_v2(map_list)
    flat_act_list1 = flat_map_list_v2(act_list)
    flat_map_arr1 = np.asarray(flat_map_list1)
    flat_act_arr1 = np.asarray(flat_act_list1)
    
    shape_map = np.shape(flat_map_arr1)
    shape_act = np.shape(flat_act_arr1)
    
    vect_map1 = B.reshape(flat_map_arr1, (shape_map[0]*shape_map[1]*shape_map[2],1))
    vect_act1 = B.reshape(flat_act_arr1, (shape_act[0]*shape_act[1]*shape_act[2],1))

    vect_map1_inv = 1 - vect_map1
    vect_act1_inv = 1 - vect_act1
    pos_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1*vect_map1*tf.math.log(vect_act1*vect_pred+ 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1)
    neg_loss = tf.reduce_sum(tf.clip_by_value((-vect_act1_inv*vect_map1_inv*tf.math.log(vect_act1_inv*(1-vect_pred) + 1e-15)), 0.0, 15.))/tf.reduce_sum(vect_act1_inv)

    return pos_loss, neg_loss
tensor_pos_neg_loss(pred, map_list1, act_list1)  
'''

#--------POSNEG LOSS CALCULATIONS END----------------------#

#------DiSTANCE LOSS CALCULATIONS BEGIN----------------------#
'''
#np.random.rand(dim_grid1[0], dim_grid1[1])
mapi = weight_list1[1]
idx = mapi.shape[2]-1
for i in range(mapi.shape[2]):
    show_batch(flat_map_list_v2([mapi[...,i,None]]))

expanded = expand_zero_tensor_map(pred[0],idx, True)
print('pred_expanded')
print(expanded.numpy())
acti = mapi

acti_T= np.transpose(acti, (2,0,1))
acti_shape = np.shape(acti_T)
acti_re = np.reshape(acti_T, (acti_shape[0],acti_shape[1]*acti_shape[2]))

x = np.ones((mapi.shape[0],mapi.shape[1]))[...,None]
x = np.concatenate([x,2*x], 2)

y = np.transpose(x, (2,0,1))
shape_y = y.shape 
y = np.reshape(y,(shape_y[0], shape_y[1]*shape_y[2]) )
acti_re_3D = acti_re[:,None,:]
print('weight maps')
print(np.round(acti_re_3D, 2))
loss_map = acti_re_3D*expanded.numpy()
print('multiplied')
print(np.round(loss_map, 2))
print('loss_map_reduced')
loss_map_red = np.sum(loss_map, 2)
print(np.round(loss_map_red, 2))

deleted = []
min_list = []
for i in range(mapi.shape[2]):
    print('loss map red [',i,']')
    print(np.round(loss_map_red[i], 2))
    sort_i = np.argsort(loss_map_red[i], 0)

    for j in range(len(loss_map_red[i])):
        if sort_i[j] not in deleted and loss_map_red[i][sort_i[j]] != 1.:
            deleted.append(sort_i[j])
            min_list.append(loss_map_red[i][sort_i[j]])
            loss_map_red[i][sort_i[j]] = 1.
            break
        elif loss_map_red[i][sort_i[j]] == 1:
            min_list.append(loss_map_red[i][sort_i[j]])
            break

print('min_list')
print(np.round(min_list, 2))
print('deleted indexes')
print(deleted)
print('mean_loss')
mean_loss = np.mean(min_list)
print(mean_loss)
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
        perm_list = permutations_v2(np.arange(mapi.shape[2]), mapi.shape[2])
    else:
        perm_list = [list(np.arange(mapi.shape[2]))]
    
    for perm in perm_list:
        #print('perm =', perm)
        loss_map_red_copy = loss_map_red.numpy()
        deleted = []
        min_list = []
        for i in perm:
            #print('loss map red [',i,']')
            #print(np.round(loss_map_red_copy[i], 2))
            sort_i = tf.argsort(loss_map_red_copy[i], 0)
    
            for j in range(len(loss_map_red_copy[i])):
                # if sort_i[j] not in deleted and loss_map_red[i][sort_i[j]] != loss_map_red[i][sort_i[-1]]:
                if sort_i[j] not in deleted and loss_map_red_copy[i][sort_i[j]] != 1.:
                    deleted.append(sort_i[j].numpy())
                    min_list.append(loss_map_red_copy[i][sort_i[j]])
                    loss_map_red_copy[i][sort_i[j]] = 1.
                    break
                elif loss_map_red_copy[i][sort_i[j]] == 1.:
                    min_list.append(loss_map_red_copy[i][sort_i[j]])
                    break
    
        #print('min_list')
        #print(np.round(min_list, 2))
        #print('deleted indexes')
        #print(deleted)
        print('mean_loss_per_permutation')
        mean_loss = tf.reduce_mean(min_list)
        mean_loss_list.append(mean_loss)
        print(mean_loss.numpy())
    return tf.reduce_min(mean_loss_list)

#print(dist_lossi(mapi, pred[0],1))

def dist_loss_from_list(weight_list, pred, permutate = 0):
    loss_list =[]
    
    for weighti, i in zip(weight_list, range(len(pred))):
        loss_list.append(dist_lossi(weighti, pred[i], permutate))
        print('min_loss_per_pred')
        print(loss_list[i].numpy())
        
    return tf.reduce_mean(loss_list)
print("mean loss of all predictions")
print(dist_loss_from_list(weight_list1,pred, 1))


b = np.arange(0,3)
            
list(permutations_v2(b,3))
'''
#------DiSTANCE LOSS CALCULATIONS END----------------------#    
'''   
# pred_zeroed = tensor_sample_thresh_by_idx(pred, idx).numpy()
# #pred_zeroed = np.reshape(pred_zeroed, (H_grid, W_grid))

# skel = np.ones((idx+1, H_grid* W_grid))
# meat = skel*pred_zeroed

# sort_ind = tf.experimental.numpy.flip(tf.argsort(pred_zeroed), axis = 0)
# pred_zeroed_sort = tf.experimental.numpy.take_along_axis(pred_zeroed, sort_ind, axis = 0)
# act_vect = pred_zeroed_sort[:idx+1,None]
# expanded = tf.where(meat != act_vect, 0., meat)
# #zeroed = tf.where( flat < flat_sort[:,idx][:,None], 0., 1.)

'''
###______________DiSTANCE LOSS CALCULATIONS BEGIN__________________############



###_______________NEW1 CODE START__________________________________############

"""
for batch in batch_gen:

    im_list = image_list_from_batch(batch, dim)
    
    temp = seq1.deepcopy()
    im_aug_list = seq1(images = im_list)
    
    #predict data
    
    flat_arr1, weight_arr1, _ = process_batch(batch, dim, dim_grid1, temp.deepcopy())
    flat_arr2, weight_arr2, seq1 = process_batch(batch, dim, dim_grid2, temp.deepcopy())
    

    for i in range(len(flat_arr1)):
        flat1, flat2 = flat_arr1[i], flat_arr2[i]
        weight1, weight2 = weight_arr1[i], weight_arr2[i]

        show_wait(np.concatenate([flat1, weight1], axis = 1), scale, '28x28', 0, 1, 0, 0, interpolation = cv.INTER_NEAREST)
        show_wait(np.concatenate([flat2, weight2], axis = 1), scale, '14x14', 1, 1, 0, 300, interpolation = cv.INTER_NEAREST)
    
    
    pred1 = flat_arr1[...,None]
    pred2 = flat_arr2[...,None]
    
    pred1[0,...,0].shape
    # pred_shape = pred.shape
    # w_pred, h_pred =  pred_shape[2], pred_shape[1]
    
    # #pred =  np.reshape(pred[...,1], (batch_size*w_pred*h_pred,1)) #B.reshape
    # pred =  np.reshape(pred, (batch_size*w_pred*h_pred,1))         #B.reshape
    
    # weights1 = np.ndarray.astype(weight_arr1, pred.dtype)#tf.cast
    # weights1 =  np.reshape(weights1, (batch_size*w_pred*h_pred,1))#B.reshape
    
    # truth1 = np.ndarray.astype(flat_arr1, pred.dtype)#tf.cast
    # truth1 =  np.reshape(truth1, (batch_size*w_pred*h_pred,1))#B.reshape
    
    # loss = np.mean(weights1*np.sqrt((pred-truth1)*(pred-truth1)+1e-15), axis = 0) #tf.reduce_mean
    # print(loss)
    



    print(loss_from_map(pred1, flat_arr1, weight_arr1))
    print(loss_from_map(pred2, flat_arr2, weight_arr2))
    break
    
dim_grid_list = [(7,7), (14,14), (28,28)]

for oim in oimg_list[285:286]:
    for dim_grid in dim_grid_list:
        oim.set_comp_map(dim, dim_grid)
        show_wait(oim.get_comp_map(), scale, interpolation = cv.INTER_NEAREST)
        a = oim.get_comp_pn_3d()
        print(a.shape)


start = 285
end = 301
i = start


batch_img_a = []
batch_map_a = []
batch_pn_a = []
 

    
for oim in batch:
    print(oim.name)
    oim.set_comp_map(dim, dim_grid)
    im = oim.load_image(dim)/255.0
    im_b = oim.get_comp_map_3d()
    im_a, im_c = seq1(image = im, heatmaps = im_b[None,:,:])
    im_c =im_c[0,:,:]
    
    im_d = im_c
    max_d = np.max(im_d, axis = (0,1), keepdims=True)
    min_d = np.min(im_d, axis = (0,1), keepdims=True)
    im_d = (im_d-min_d)/(max_d - min_d + 1e-15)
    im_d = np.maximum.reduce(im_d, axis=2)
    im_d = 1 - im_d
    max_c = np.max(im_c, axis = (0,1), keepdims=True) #get maximum of each channel
    im_c = np. where(im_c < max_c, 0.,1.) #set max element of each channel to 1 and all else to zero.
    im_c = np.maximum.reduce(im_c, axis=2)
    im_c= np.float32(im_c)
   
    v_divider = np.zeros((dim_grid[0], int(0.05*dim_grid[1]),3 ), dtype = np.uint8)
    im_map_aug =  cv.resize(cv.cvtColor(im_d, cv.COLOR_GRAY2BGR), dim, interpolation = cv.INTER_NEAREST)
    im_pn_aug = cv.resize(cv.cvtColor(im_c, cv.COLOR_GRAY2BGR), dim, interpolation = cv.INTER_NEAREST)
    im_pn = cv.resize(cv.cvtColor(oim.get_comp_pn(), cv.COLOR_GRAY2BGR), dim, interpolation = cv.INTER_NEAREST)
    im_comp = cv.resize(cv.cvtColor(1.-oim.get_comp_map(), cv.COLOR_GRAY2BGR), dim, interpolation = cv.INTER_NEAREST)
    im_or = np.hstack((im, im_comp, im_pn))
    im_aug = np.hstack((im_a, im_map_aug,im_pn_aug))
    h_divider = np.zeros(( int(0.08*dim_grid[1]),2*dim_grid[0]+np.shape(v_divider)[1], 3), dtype = np.uint8)
    
    batch_img_a.append(im_a)
    batch_map_a.append(im_d)
    batch_pn_a.append(im_c)
    
    show_wait(im_or)
    show_wait(im_aug)



for oim in oimg_list:
    oim.set_comp_map(dim, dim_grid)"""

###_______________NEW1 CODE END__________________________________############

'''
x, y = e_1.x, e_1.y

rx, ry = e_1.rx, e_1.ry
angle = e_1.theta*180/np.pi

dim_grid = (21,21)
x_grid, y_grid = int(x/dim[0]*dim_grid[0]), int(y/dim[1]*dim_grid[1])

rx_nrm, ry_nrm = rx/dim[0], ry/dim[1]

r_min = np.min([rx_nrm, ry_nrm])
r_base= 0.15

if r_min < r_base:
    factor = r_base/r_min
    rx_nrm *=factor
    ry_nrm *=factor
    
rx_grid, ry_grid = int(np.ceil(rx_nrm*dim_grid[0])), int(np.ceil(ry_nrm*dim_grid[1]))


cobj1 = cells_object.map_2_grid_from_xy(dim, dim_grid, (x,y))
_, r_map1 = cobj1.load_radial_map()
center = cobj1.grid_centers_ji[0]
c_x, c_y = center[1], center[0]

show_wait(r_map1, scale, interpolation = cv.INTER_NEAREST)

t12 = np.array([[1.,0.,-c_x], [0., -1., c_y], [0., 0., 1.]])
scale_matrix = np.array([[rx_nrm , 0.0, 0.], [0.0, ry_nrm, 0.],[0., 0., 1.]])
#scale_matrix = np.array([[0.5 , 0.0, 0.], [0.0, 0.5, 0.],[0., 0., 1.]])
t21 = inverse_transform_matrix(t12)

M1 = t21@scale_matrix@t12

M = M1[:2,:]

r_map1_warped = cv.warpAffine(r_map1, M, dim_grid)
show_wait(r_map1_warped,scale, interpolation = cv.INTER_NEAREST)

rot_mat = cv.getRotationMatrix2D((c_x,c_y), 180 - angle, 1)

r_map1_rot = cv.warpAffine(r_map1_warped, rot_mat, dim_grid)
show_wait(r_map1_rot,scale, interpolation = cv.INTER_NEAREST)
'''

'''
# ellipse_image = cv.ellipse(np.zeros(dim_grid, np.uint8), (c_x,c_y), (rx_grid, ry_grid), angle, 0, 360, 255, -1)
# show_wait(ellipse_image,scale, interpolation = cv.INTER_NEAREST)

# map_rot = ellipse_image/255.0*r_map1_rot
# show_wait(map_rot,scale, interpolation = cv.INTER_NEAREST)

# trans_vect = np.array([x_grid,  y_grid]) - np.array([c_y, c_x])

# trans_matrix = np.array([[1, 0.0, trans_vect[0]], [0.0, 1, trans_vect[1]],[0., 0., 1.]])

# M = trans_matrix

# M= M[:2,:]
# map_final = cv.warpAffine(map_rot,M, dim_grid)
# show_wait(map_final,scale, interpolation = cv.INTER_NEAREST)

# e_1 = oimg_list[258].region_list[0]

# dim = (4056,3040)
# x, y = e_1.x, e_1.y

# rx, ry = e_1.rx, e_1.ry
# angle = e_1.theta*180/np.pi

# dim_grid = (21,21)
# x_grid, y_grid = int(x/dim[0]*dim_grid[0]), int(y/dim[1]*dim_grid[1])

# rx_nrm, ry_nrm = rx/dim[0], ry/dim[1]

# rx_grid, ry_grid = int(rx_nrm*dim_grid[0]), int(ry_nrm*dim_grid[1])
# scale = 50

# cobj1 = cells_object.map_2_grid_from_xy(dim, dim_grid, (dim[0]/2, dim[1]/2))
# _, r_map1 = cobj1.load_radial_map()
# center = cobj1.grid_centers_ji[0]
# c_x, c_y = center[1], center[0]

# show_wait(r_map1, scale, interpolation = cv.INTER_NEAREST)

# t12 = np.array([[1.,0.,-c_y], [0., -1., c_x], [0., 0., 1.]])
# scale_matrix = np.array([[2*rx_nrm , 0.0, 0], [0.0, 2*ry_nrm, 0],[0., 0., 1.]])
# t21 = inverse_transform_matrix(t12)

# M1 = t21@scale_matrix@t12

# M = M1[:2,:]

# r_map1_warped = cv.warpAffine(r_map1, M, dim_grid)
# show_wait(r_map1_warped,scale, interpolation = cv.INTER_NEAREST)

# rot_mat = cv.getRotationMatrix2D((c_x,c_y), -angle, 1)

# r_map1_rot = cv.warpAffine(r_map1_warped, rot_mat, dim_grid)
# show_wait(r_map1_rot,scale, interpolation = cv.INTER_NEAREST)

# ellipse_image = cv.ellipse(np.zeros(dim_grid, np.uint8), (c_x,c_y), (rx_grid, ry_grid), angle, 0, 360, 255, -1)
# show_wait(ellipse_image,scale, interpolation = cv.INTER_NEAREST)

# map_rot = ellipse_image/255.0*r_map1_rot
# show_wait(map_rot,scale, interpolation = cv.INTER_NEAREST)


# trans_vect = np.array([x_grid,  y_grid]) - np.array([c_y, c_x])

# trans_matrix = np.array([[1, 0.0, trans_vect[0]], [0.0, 1, trans_vect[1]],[0., 0., 1.]])

# M = trans_matrix

# M= M[:2,:]
# map_final = cv.warpAffine(map_rot,M, dim_grid)
# show_wait(map_final,scale, interpolation = cv.INTER_NEAREST)
'''



"""
XY_train_oimg, XY_test_oimg = shuffle_and_split_original_img_data(oimg_list, total_weight)

dim = (100,100)     
XY_train = original_img_data_2_kptarr(XY_train_oimg)
XY_test = original_img_data_2_kptarr(XY_test_oimg)
train_length = len(XY_train)
test_length = len(XY_test)





folder_0 = '0_segments'
XY_0_names = load_class_0(ROOT_DIR, folder_0)

random.Random(4).shuffle(XY_0_names)
XY_train_0_names = XY_0_names[:train_length]
XY_test_0_names = XY_0_names[train_length:test_length]


    

#calculate mean and std once for whole X_train dataset
#X_train, y_train_koi, y_train_bboi, names = split_XY(XY_train, dim)

batch_size = 8
for j in range(4): 
    batch_generator_1 = generate_batch(XY_train.copy(), int(batch_size/2), 0)
    batch_generator_0 = generate_batch(XY_train_0_names.copy(), int(batch_size/2), 0)
    i = 0
    seq = augments()
    for batch, batch_0 in zip(batch_generator_1, batch_generator_0):
        if i < 5:
            X_batch_01, y_batch_01, names_01 = mix_and_augment_batches_01(batch, batch_0, seq)
            print('batch_length = ', len(names_01))
            show_batch(X_batch_01, names_01)
        i+=1
    #print ('iterations =', i)
    
"""    

"""

   
_, X_mean, X_std = std_im(X_train)
_, y_theta = koi_list2hesse(y_train, names)
y_norm, y_mean, y_std = std_vector(y_theta)

del(X_train)
del(y_train)



XY_test = original_img_data_2_kptarr(XY_test_oimg)     

"""


# batch_size = 8
# for j in range(4): 
#     batch_generator = generate_batch(XY_train.copy(), batch_size, 0)
#     i = 0
#     seq = augments()
#     for batch in batch_generator:
        
#         if i < 1:
            
#             X_batch, y_batch_koi, y_batch_bboi, batch_names = split_XY(batch, dim)
#             # for bboi, name in zip(y_batch_bboi, batch_names):
#             #     print(bboi.to_xyxy_array(), ' || ', name[-13:])
#             X_batch_a, y_batch_a_koi, y_batch_a_bboi = seq(images = X_batch, keypoints = y_batch_koi, bounding_boxes = y_batch_bboi)
#             _, y_batch_theta = koi_list2hesse(y_batch_a_koi, batch_names)
            
            
#             _, axs2 = plt.subplots(1, len(X_batch), figsize=(40, 40))
#             _, axs = plt.subplots(1, len(X_batch), figsize=(40, 40))
#             axs = axs.flatten()
#             axs2 = axs2.flatten()
            
#             for img, pts, bbox, names, ax in zip(X_batch, y_batch_koi, y_batch_bboi, batch_names, axs2):
#                 bbox = bbox.remove_out_of_image().clip_out_of_image()
#                 img = bbox.draw_on_image(img, color = (255,0,0), size = 2)
#                 ax.text(0,0, names[-13:])
#                 ax.imshow(pts.draw_on_image(img, color = (0,0,255), size=2))    
            
#             for img, pts, bbox, names, ax in zip(X_batch_a, y_batch_a_koi, y_batch_a_bboi, batch_names, axs):
#                 bbox = bbox.clip_out_of_image()
#                 img = bbox.draw_on_image(img, color = (0,0,255), size = 2)
#                 ax.text(0,0, names[-13:])
#                 ax.imshow(pts.draw_on_image(img, color = (255,0,0), size=2))
#             plt.show()
#         else:
#             break
#         i+=1
        
#     print ('iterations =', i)
    
    # for bboi in y_batch_bboi:
    #     print(bboi.to_xyxy_array())



# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 18:51:15 2022

@author: i_bab
"""
from __future__ import print_function
import argparse
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


import os
import file
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

from helper import show_wait, process
def estructurant(radius):
    kernel = np.zeros((2*radius+1, 2*radius+1) ,np.uint8)
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    kernel[0,radius-1:kernel.shape[1]-radius+1] = 1
    kernel[kernel.shape[0]-1,radius-1:kernel.shape[1]-radius+1]= 1
    kernel[radius-1:kernel.shape[0]-radius+1,0] = 1
    kernel[radius-1:kernel.shape[0]-radius+1,kernel.shape[1]-1] = 1
    return kernel

def getThreshHsv(img = None, low = (33, 0, 0), high = (109, 255, 255)):
    frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, low, high)
    frame_threshold = cv.cvtColor(frame_threshold, cv.COLOR_GRAY2BGR)
    frame_threshold = np.bitwise_and(img,frame_threshold)
    return frame_threshold

def getThresh(frame_threshold = None, clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)), k_dim = (3, 50, 50)):
    cl1 = np.clip(clahe.apply(cv.cvtColor(frame_threshold, cv.COLOR_BGR2GRAY)).astype(np.uint16), 0, 255).astype(np.uint8)
        
    #decreasing sigmas decreases noise
    #POSSIBLE GROWTH MAP from big to small
    #%area decrease within BoundingBox of big
    blur = cv.bilateralFilter(cl1, k_dim[0], k_dim[1], k_dim[2])   
    # Otsu's thresholding
    ret2,th2 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    return th2   

'''
#ROOT_DIR = file.ROOT_DIR

ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))

#divide images and masks by 4

#go to path

# data_path = 'data2019\\1\\train1'
# dest_folder = '1_m_greater'
# dest_folder2 = '1_m_lesser'

data_path = 'data2019\\1\\train1'
dest_folder = 'data2019\\1\\train1_masks_closed'
#dest_folder2 = '1_m_lesser'


img_path = os.path.join(ROOT_DIR, data_path)
dest_path = os.path.join(ROOT_DIR, dest_folder)
#dest_path2 = os.path.join(ROOT_DIR, dest_folder2)
img_list = os.listdir(img_path)
max_value = 255
max_value_H = 360//2
low_H = 33
low_S = 0
low_V = 0
high_H = 109
high_S = max_value
high_V = max_value
n_div = 4


for i in range(30):#range(len(img_list)):#range(100) :
    if img_list[i][-3:] not in ['jpg', 'bmp', 'png']:
        print("Error skipped this .'"+ img_list[i][-3:] + "' file")
    else:
        img =  cv.imread(os.path.join(img_path,img_list[i]))
        h, w = np.shape(img)[0], np.shape(img)[1]
        #img = cv.resize(img, (w//4, h//4))
        
        
        frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        frame_threshold = cv.cvtColor(frame_threshold, cv.COLOR_GRAY2BGR)
        frame_threshold = np.bitwise_and(img,frame_threshold)

        clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        cl1 = clahe.apply(cv.cvtColor(frame_threshold, cv.COLOR_BGR2GRAY))

        blur = cv.bilateralFilter(cl1, 3, 50, 50) #decreasing sigmas decereases noise
                                                    #POSSIBLE GROWTH MAP from big to small
                                      #%area decrease within BoundingBox of big
        kw = 3    

        
        # Otsu's thresholding
        ret2,th2 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        #show_wait(th2)
        th2_clr= np.stack([th2,th2,th2], axis = 2)
        show_wait(np.concatenate([img,th2_clr], axis = 1))
        # cv.imwrite(os.path.join(dest_path,img_list[i][:-4]+''+img_list[i][-4:]),th2_clr)
        erode = th2
        kernel = np.ones((kw,kw))
        se = disk(2)
        for j in range(1):
        #     #MOD OPEN 1
        #     erode = cv.erode(erode,kernel)
        #     erode = cv.dilate(erode, kernel) 
            #MOD CLOSE 1
            erode = cv.dilate(erode, kernel)
            erode = cv.erode(erode,kernel) #or se
        erode_clr= np.stack([erode,erode,erode], axis = 2)
        #cv.imwrite(os.path.join(dest_path,img_list[i][:-4]+''+img_list[i][-4:]),erode_clr)
        # #show_wait(erode)
        
        # erode_clr= np.stack([erode,erode,erode], axis = 2)
        # show_wait(np.concatenate([img,erode_clr], axis = 1))
        # break
        # se = disk(3)
        # for j in range(1):
        #     erode = cv.erode(erode,se)
        # se = disk(2)
        # #img2 = cv.resize(erode, (w, h))
        # #cv.imwrite(os.path.join(dest_path,img_list[i][:-4]+'b'+img_list[i][-4:]), img2)
        # for j in range(1):
        #     erode = cv.erode(erode,se)
        # #img2 = cv.resize(erode, (w, h))  
        # img2 = erode #MIGHT AS WELL SAVE THEM AT 1/4 OF THE SIZE 
        # #cv.imwrite(os.path.join(dest_path,img_list[i][:-4]+''+img_list[i][-4:]), img2)#USED FOR GREATER MASK
        # for j in range(3):
        #     erode = cv.erode(erode,se)
        # #img2 = cv.resize(erode, (w, h)) 
        # img2 = erode #MIGHT AS WELL SAVE THEM AT 1/4 OF THE SIZE 
        # #cv.imwrite(os.path.join(dest_path2,img_list[i][:-4]+''+img_list[i][-4:]), img2)#USED FOR LESSER MASK

        # if i%10 ==0:
        #     print('i = ' , i )
            
        
        # #show_wait(erode)
'''      


        
        
        
        
        
        
        
        
        
        
        
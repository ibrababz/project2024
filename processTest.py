# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:01:55 2024

@author: i_bab
"""
import keyboard
import cv2 as cv
import numpy as np
import os
import file
from helper import show_wait
from binarizationHSV import getThresh, getThreshHsv
from adjustContrast import claheHsv


ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))
weed_folder = 'data2019\\0\\train1_0_grass_dark'
weed_path = os.path.join(ROOT_DIR, weed_folder)

dest_folder = 'data2019\\0\\train1_0_grass_dark_contrast'
# dest_folder = 'data2019\\1\\train1_contrastv2'
dest_path = os.path.join(ROOT_DIR, dest_folder)
if not os.path.exists(dest_path):
    os.makedirs(dest_path, exist_ok = True)
img_list = os.listdir(weed_path)

for img in img_list:
    if img[-3:] not in ['jpg', 'bmp', 'png']:
        print("Error skipped this .'"+ img[-3:] + "' file")
    else:
        weed = cv.imread(os.path.join(weed_path,img))
        #show_wait(weed)
        clip_limit = 1
        weed_new = claheHsv(weed, clip_limit)
        cv.imwrite(os.path.join(dest_path,img[:-4]+''+img[-4:]),weed_new)
        # show_wait(weed_new)
        # h_upper = int(255);
        # h_lower = int(0);
        # # key = 'null'
        # # while key != 'x':
        # #     thresh_hsv = getThreshHsv(weed_new.copy(), (int(h_lower),0,0), (int(h_upper), 255, 255))
        # #     show_wait(thresh_hsv)
        # #     key = keyboard.read_key()
        # #     if key == 'w':
        # #         h_lower+=1
        # #     elif key =='s':
        # #         h_lower-=1
        # #     elif key =='up':
        # #         h_upper+=1
        # #     elif key =='down':
        # #         h_upper-=1
    
        # #     h_lower = np.clip(h_lower, 0, 255)
        # #     h_upper = np.clip(h_upper, 0, 255)
        # #     h_lower = np.clip(h_lower, 0 ,h_upper)
        # #     h_upper = np.clip(h_upper, h_lower, 255)
            
        # #     print('(h_lower, h_upper) = (', h_lower,',', h_upper, ')')
        
        # thresh_hsv = getThreshHsv(weed_new.copy(), (int(h_lower),0,0), (int(h_upper), 255, 255))
        # thresh = getThresh(thresh_hsv)
        # # show_wait(thresh)
        # thresh_bgr= np.stack([thresh,thresh,thresh], axis = 2)
        # cv.imwrite(os.path.join(dest_path,img[:-4]+''+img[-4:]),thresh)
    
   

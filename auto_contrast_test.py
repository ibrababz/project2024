# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 22:15:23 2024

@author: i_bab
"""

import cv2 as cv
import numpy as np
import os
import file
from helper import show_wait

from automatic_brightness_and_contrast import automatic_brightness_and_contrast

ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))
src_folder = 'data2019\\0\\train1_0_grass_dark'
src_path = os.path.join(ROOT_DIR, src_folder)
src_im_list = os.listdir(src_path)

tar_folder = 'data2019\\0\\train1_0_grass_dark'
tar_path = os.path.join(ROOT_DIR, tar_folder)
tar_im_list = os.listdir(tar_path)

# dest_folder = 'data2019\\0\\train1_0_grass_dark_contrast_color_transfer'
# dest_path = os.path.join(ROOT_DIR, dest_folder)
# if not os.path.exists(dest_path):
#     os.makedirs(dest_path, exist_ok = True)

# for img in img_list:
#     if img[-3:] not in ['jpg', 'bmp', 'png']:
#         print("Error skipped this .'"+ img[-3:] + "' file")
#     else:
    
src_im = cv.imread(os.path.join(src_path,src_im_list[0]))

show_wait(src_im)

auto_im, _, _ = automatic_brightness_and_contrast(src_im)

show_wait(auto_im)

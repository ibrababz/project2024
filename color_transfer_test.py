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
from color_transfer import color_transfer
from automatic_brightness_and_contrast import automatic_brightness_and_contrast
import colortrans
from RandomSampler import RandSampler

ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))
src_folder = 'data2019\\0\\train1_0_grass'
src_path = os.path.join(ROOT_DIR, src_folder)
src_im_list = os.listdir(src_path)

tar_folder = 'data2019\\1\\train1'
tar_path = os.path.join(ROOT_DIR, tar_folder)
tar_im_list = os.listdir(tar_path)

dest_folder = 'data2019\\0\\train1_0_grass_dark_contrast_color_transfer'
dest_path = os.path.join(ROOT_DIR, dest_folder)
if not os.path.exists(dest_path):
    os.makedirs(dest_path, exist_ok = True)

# for img in img_list:
#     if img[-3:] not in ['jpg', 'bmp', 'png']:
#         print("Error skipped this .'"+ img[-3:] + "' file")
#     else:
    
src_im = cv.imread(os.path.join(src_path,src_im_list[1]))
tar_im = cv.imread(os.path.join(tar_path,tar_im_list[0]))

# show_wait(src_im)
# show_wait(tar_im)

# show_wait(color_transfer(src_im, tar_im, True, True))
# show_wait(color_transfer(src_im, tar_im, True, False))

# from automatic_brightness_and_contrast import automatic_brightness_and_contrast    

# tar_im_auto, _, _ = automatic_brightness_and_contrast(tar_im)
# show_wait(color_transfer(src_im, tar_im_auto, True, True))
# show_wait(color_transfer(src_im, tar_im_auto, True, False))


# pccm = colortrans.transfer_pccm(tar_im[...,::-1],src_im[...,::-1])[...,::-1]
# lhm= colortrans.transfer_lhm(tar_im[...,::-1],src_im[...,::-1])[...,::-1]

# show_wait(np.hstack([src_im, tar_im, pccm,lhm]), scale = 0.5, interpolation = cv.INTER_AREA)
tarRandSampler = RandSampler(tar_im_list, 1)
srcRandSampler = RandSampler(src_im_list, 1)

for i, src_im_file, tar_im_file in zip(range(20),srcRandSampler, tarRandSampler):
    src_im = cv.imread(os.path.join(src_path,src_im_file[0]))
    tar_im = cv.imread(os.path.join(tar_path,tar_im_file[0]))
    
    lhm= colortrans.transfer_lhm(tar_im[...,::-1].astype(np.float32),src_im[...,::-1].astype(np.float32))[...,::-1].astype(np.uint8)
    pccm = colortrans.transfer_pccm(lhm[...,::-1].astype(np.float32),src_im[...,::-1].astype(np.float32))[...,::-1].astype(np.uint8)
    #pccm = colortrans.transfer_pccm(tar_im[...,::-1].astype(np.float32),src_im[...,::-1].astype(np.float32))[...,::-1].astype(np.uint8)

    #show_wait(np.hstack([src_im, tar_im, pccm,lhm]), scale = 0.7, interpolation = cv.INTER_AREA)
    show_wait(np.hstack([src_im, tar_im, lhm]), interpolation = cv.INTER_AREA)
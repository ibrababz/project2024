import cv2 as cv
import numpy as np


def claheHsv(weed, clip_limit, tileGridSize = (8,8)):
    weed_hsv = cv.cvtColor(weed, cv.COLOR_BGR2HSV)
    weed_v = weed_hsv[...,2]
    clahe = cv.createCLAHE(clipLimit = clip_limit, tileGridSize = tileGridSize)
    weed_v_adj = np.clip(clahe.apply(weed_v).astype(np.uint16) + 30, 0, 255).astype(np.uint8)
    weed_hsv_adj = np.concatenate([weed_hsv[...,0:2], weed_v_adj[...,None]], axis = 2)
    
    weed_new = cv.cvtColor(weed_hsv_adj, cv.COLOR_HSV2BGR)
    return weed_new
'''
import os
import file
from helper import show_wait
import keyboard
import time

ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))

weed_folder = 'data2019\\1\\train1'
weed_adj_folder = 'data2019\\1\\train1_contrast_adjusted'
weed_path = os.path.join(ROOT_DIR, weed_folder)
weed_adj_path = os.path.join(ROOT_DIR, weed_adj_folder)

for weed_file in os.listdir(weed_path)[1:2]:
    weed = cv.imread(os.path.join(weed_path, weed_file))
    
    show_wait(weed)
    # keyboard.add_hotkey('up, down')
    weed_prev = weed.copy()
    for clip_limit in range(1,3):
        # weed_hsv = cv.cvtColor(weed.copy(), cv.COLOR_BGR2HSV)
        # weed_v = weed_hsv[...,2]
        # clahe = cv.createCLAHE(clipLimit = clip_limit)
        # weed_v_adj = np.clip(clahe.apply(weed_v).astype(np.uint16) + 30, 0, 255).astype(np.uint8)
        # weed_hsv_adj = np.concatenate([weed_hsv[...,0:2], weed_v_adj[...,None]], axis = 2)
        
        # weed_new = cv.cvtColor(weed_hsv_adj, cv.COLOR_HSV2BGR)
        weed_new = claheHsv(weed, clip_limit)
        show_wait(weed_new)
        # diff = np.sum(weed_new - weed_prev)
        # if diff < 5000:
    #     break
    #     # print(new_zeros)
    #     # show_wait(weed_new)
    #     # if new_zeros > 4:
    #     #     if clip_limit ==1:
    #     #         print("this is the best we can do!")
    #     #         break
    #     #     else:
    #     #         clip_limit -=1
    #     #         break

    
    # weed_hsv = cv.cvtColor(weed.copy(), cv.COLOR_BGR2HSV)
    # weed_v = weed_hsv[...,2]
    # clahe = cv.createCLAHE(clipLimit = clip_limit)
    # weed_v_adj = np.clip(clahe.apply(weed_v).astype(np.uint16) + 30, 0, 255).astype(np.uint8)
    # weed_hsv_adj = np.concatenate([weed_hsv[...,0:2], weed_v_adj[...,None]], axis = 2)
    
    # weed_new = cv.cvtColor(weed_hsv_adj, cv.COLOR_HSV2BGR)  
    # print(clip_limit)
    # show_wait(weed_new)
'''    

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:33:11 2024

@author: i_bab
"""

# import cv2 as cv
import os
import file
import shutil


if __name__=='__main__':
    ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))
    wFolder = os.path.join('project2024', 'resnet_test_13_real_8bit_res2_ep_(0, 1000)_lr_1e-04')
    wNamesFile = 'test_files.txt'
    wFilePath = os.path.join(ROOT_DIR, wFolder, wNamesFile)
    with open(wFilePath) as wFile:
        wFilesList = [wLine.rstrip() for wLine in wFile]
        
    
    wTestImageFolder = 'test_images_448_bitmap'
    wSaveDir = os.path.join(ROOT_DIR, 'data4k', wTestImageFolder)
    os.makedirs(wSaveDir, exist_ok=True)
    for wFileName in wFilesList:
        wSrcPath = wFileName
        wDstPath = os.path.join(wSaveDir, wFileName.split('\\')[-1])
        shutil.copy2(wSrcPath, wDstPath)
        print(wFileName)
    
    
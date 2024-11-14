# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:20:22 2022

@author: i_bab
"""
import cv2 as cv

def adjust_number(number_string, length = 4):
    if type(number_string) is not str:
        number_string = str(number_string)
    if len(number_string)<length:
           return adjust_number('0'+number_string, length)
    else:
        return number_string

def show_wait(img, scale = 1, name = 'image', destroy = 1, wait = 1, x = 0, y =0, interpolation = cv.INTER_LINEAR):
    
    height = img.shape[0]
    width = img.shape[1]
    height *=scale
    width *=scale
    
    
    img = cv.resize(img,(int(width),int(height)), interpolation = interpolation)
    cv.namedWindow(name)
    cv.moveWindow(name, x, y)
    cv.imshow(name, img)
    
    if wait:
        cv.waitKey(0)
    if destroy:
        cv.destroyAllWindows()
    if wait:
        cv.waitKey(1)

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:04:38 2022

@author: i_bab
"""

from __future__ import print_function
import argparse
import cv2 as cv
import numpy as np
from numpy import argsort
from matplotlib import pyplot as plt
import pickle
import csv
import os
import file
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from operator import itemgetter

import time
import keyboard

from helper import show_wait, process

ROOT_DIR = file.ROOT_DIR

#go to path


class leafObject():
    
    def __init__(self, contour, hierarchy0_i, idx, im_shape):
        self.contour = contour
        #print("contour = ", self.contour)
        self.idx = idx
        self.hierarchy0_i = hierarchy0_i
        self.im_h, self.im_w = im_shape[0], im_shape[1]
        #self.saved = False
       
        if hierarchy0_i[2] == -1:
            self.isParent = False
            self.childIdx = None
        else:
            self.isParent = True
            self.childIdx = []
            self.name = 'P' #i.e. Parent
            
        if hierarchy0_i[3] == -1:
            self.isChild = False
            self.parentIdx = None
        else:
            self.isChild =True
            self.parentIdx = hierarchy0_i[3]
            self.name = 'C' #i.e. Child
     
        if self.isChild == False and self.isParent == False:
            self.name = 'B' #i.e. Bachelor ;)
                
        self.set_flag = False
        self.setLeaf()
        
    def setArea(self):
        self.area = cv.contourArea(self.contour)
        #print("the area is ", self.area)
        
        
    def setCentroid(self):
        M = cv.moments(self.contour)
        if M["m00"] != 0:
            self.cx = int(M['m10']//M['m00'])
            self.cy = int(M['m01']//M['m00'])
        else:
            self.cx = self.cy = int(0)
        #print('cx = ', self.cx, ' cy = ', self.cy)
        
    def setPerimeter(self):
        self.perimeter = cv.arcLength(self.contour, True)
        
    def setBbox(self):
        self.x,self.y,self.w,self.h = cv.boundingRect(self.contour)
        minh = minw = 100; 
        maxy, maxx = self.im_h-1, self.im_w-1;
        
        if self.h < minh:
            deltah = minh - self.h
            self.h = minh
            
            if(self.y - deltah//2 < 0):
                self.y = 0
            else:
                self.y -= deltah//2
       
        if self.w < minw:
            deltaw = minw - self.w
            self.w = minw
            if(self.x - deltaw//2 < 0):
                self.x = 0
            else:
                self.x -= deltaw//2
                
        y_plus_h = self.y + self.h
        if y_plus_h >=  self.im_h:
            
            deltayh = y_plus_h -self.im_h
            
            self.h -= deltayh
                        
        x_plus_w = self.x + self.w
        if x_plus_w >=  self.im_w:
            
            deltaxw = x_plus_w -self.im_w
            
            self.w -= deltaxw

    def setLeaf(self):
        
        self.setArea()
        self.setCentroid()
        self.setPerimeter()
        self.setBbox()
        self.set_flag = True
        return self.set_flag
        
    def getArea(self):
        return self.area
        
    def getCentroid(self):


        return self.cx, self.cy
    def getPerimeter(self):

        return self.perimeter
    def getBbox(self):
        return self.x,self.y,self.w,self.h
    
    def getLeaf(self, set_flag = None):
        if set_flag is None:
            set_flag = self.set_flag
        if set_flag:
            area = self.getArea()
            centroid = self.getCentroid()
            perimeter = self.getPerimeter()
            bbox = self.getBbox()
            return area, centroid, perimeter, bbox
        else:
            print("Error, values not set")
    '''def save_leaf(self, im, directory):
        if self.saved == False:
            self.saved_directory = directory
            im[self.y:self.y+self.h, self.x:self.x+self.w]
            self.saved = True
        else:
            print("already save")'''
        
    def isWeed(self):
        print("this should return true or false depending on checks")
        pass

    


#contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#show_wait(img2)

#cv.drawContours(img, contours, -1, 175, 3)
#show_wait(img)


class weedMask():
    def __init__(self, mask, ret_Mode = cv.RETR_EXTERNAL):
        '''RETR_EXTERNAL for the cv.findContours function'''
        self.mask = mask      
        self.contours, self.hierarchy = cv.findContours(mask, ret_Mode, cv.CHAIN_APPROX_SIMPLE)
        self.leafObjectList = []
        #self.is_lesser = is_lesser
        self.h, self.w = np.shape(mask)[0], np.shape(mask)[1]

    def drawContours(self, mask = None, contours = None, colour = 255, thickness = -1):
        if contours is None:
            contours = self.contours
        if mask is None:
            mask = self.mask
        cv.drawContours(mask, contours, -1, colour, thickness)
        return mask


class familyObject():
    def __init__(self, parentLeaf):
        self.parentLeaf = parentLeaf
        self.members = []
        self.children = []
        self.area = self.parentLeaf.getArea()
        self.id = None #index of parent in original hierarchy list
        self.cx, self.cy = self.parentLeaf.getCentroid()
        self.addMember(parentLeaf)
    def addMember(self, leaf):
        self.members.append(leaf)
    def getFamily(self):
         return self.members
        
    def getBbox(self, member_index = 0):
        return self.members[member_index].getBbox()
    def drawBbox(self, im):
        x, y, w, h = self.getBbox()
        if len(np.shape(im)) > 2:
            color = (0, 0 , 255)
        else:
            color = 175
        thickness = 2
        im = cv.rectangle(im, (x, y), (x + w, y + h), color, thickness)
        return im
    
class weedMaskSet(weedMask): 
    def __init__(self, mask, ret_Mode = cv.RETR_TREE ): #RETR_CCOMP for only 2 masks
        super().__init__(mask, ret_Mode)
        self.setLeafObjects()
        self.sortFamiliesbySize()
    def setLeafObjects(self):
        self.family_list = []
        self.family_areas = []
        self.family_counter = 0
        for c, h, i in zip(self.contours, self.hierarchy[0], range(len(self.contours))):
            leaf = leafObject(c, h, i, (self.h, self.w))
            #leaf.setLeaf()
            self.leafObjectList.append(leaf)
           
            
            if leaf.isParent:
                family = familyObject(leaf)
                self.family_list.append(family)
                self.family_areas.append(family.area)
                self.family_counter += 1
                family.id = i
                
            if leaf.isChild:
                self.leafObjectList[leaf.parentIdx].childIdx.append(leaf.idx)
                self.family_list[self.family_counter - 1].children.append(leaf)
                
                self.family_list[self.family_counter - 1].addMember(leaf)
                                
    def findRestofFamily(self, i):
        name = self.leafObjectList[i].name
        family = [self.leafObjectList[i]]
        if name == 'P':
            for childId in self.leafObjectList[i].childIdx:
                family.append(self.leafObjectList[childId])
            #print('found parent at Idx =', self.leafObjectList[i].idx, '!')
            #print(family)
            return family
        elif name == 'C':
            #print('lost child!')
            return self.findRestofFamily(self.leafObjectList[i].parentIdx)    
        else:
            #print('poor loner')
            return family
   
    def computeFamilyAttributes(self, family):
        familyAttributes = []
        if len(family) == 1:
            return False
        else:
            for member in family:
                familyAttributes.append(member.getLeaf())
            return familyAttributes
            
    def sortFamiliesbySize(self):
        self.sorted_families = np.flip(argsort(self.family_areas))
        
    def getFamilybySize(self, i_descending_order):
        return self.family_list[self.sorted_families[i_descending_order]]
                
    def drawLeaves(self, leafObjectList = None, im = None, blue_color_flag = 0):
        if im is None:
            im = self.mask.copy()
        #im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
        if leafObjectList == None:
            leafObjectList = self.leafObjectList
       
        if len(np.shape(im))>2:
            for leaf in leafObjectList:
                if leaf.name != 'C':
                    color = (255, 255, 255)
                else:
                    color = (175, 175, 175)
                if blue_color_flag:
                    color = (color[0], 0, 0)
                im = self.drawContours(im, [leaf.contour], color)
        else:    
            for leaf in leafObjectList:
                if leaf.name != 'C':
                    color = 255
                else:
                    color = 175
                im = self.drawContours(im, [leaf.contour], color)

        return im
    
    def drawFamilybyLeaf(self, i, im = None, blue_color_flag = 0):
        #this draws whole family from any index in the leaf_Object_list
        #not to be confused with family list
        if im is None:
            im = self.mask.copy()
        return self.drawLeaves(self.findRestofFamily(i), im, blue_color_flag)
    
    def drawFamilybySize(self, i_descending_order, im = None, blue_color_flag = 0):
        if im is None:
            im = self.mask.copy()
        family = self.getFamilybySize(i_descending_order)
        return self.drawLeaves(family.members, im, blue_color_flag)
                
        
    def drawCenters(self, leafObjectList = None, im = None):
        if im is None:
            im = self.mask.copy()
        if len(np.shape(im))==3:
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        if leafObjectList == None:
            leafObjectList = self.leafObjectList
        
        for leaf in leafObjectList:
            cx, cy = leaf.getCentroid()
            color = 100
            
            if leaf.isChild:
                size = 3 #smaller if it is from lesser mask
                thickness = 1
            else:
                size = 6 #larger if it is from greater mask
                thickness = -1            
            '''kp = cv.KeyPoint(cx,cy, size)
            im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
            im = cv.drawKeypoints(im, [kp], im, color = color)
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)'''
            
            im = cv.circle(im, (cx, cy), size, color, thickness)
            
        return im
    
    def drawBbox(self, leafObjectList = None, im = None):
        if im is None:
            im = self.mask.copy()
        if len(np.shape(im))==3:
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        if leafObjectList == None:
            leafObjectList = self.leafObjectList
        thickness = 2
        color = 100
        for leaf in leafObjectList:
            if leaf.isParent:
                x, y, w, h = leaf.getBbox()

                im = cv.rectangle(im, (x, y), (x + w, y + h), color, thickness)  
        return im

def load_image_from_file(file, scale = 1.0, flags = cv.IMREAD_COLOR):
    img =  cv.imread(file, flags = flags)
    h, w = np.shape(img)[0], np.shape(img)[1]
    if scale != 1.0:
        img= cv.resize(mask, (int(w/(scale)), int(h/(scale))))
    return img


        
class get_key_and_save_image():
    def __init__(self, im, name, list_of_output_dirs, key_list, extra_key_list):
        
        self.key = None
        self.im = im
        self.name = name
        self.list_of_output_dirs = list_of_output_dirs
        self.key_list = key_list
        self.extra_key_list = extra_key_list
        
        self.wait_for_key()
        self.flag = self.save_image()
    
    def wait_for_key(self):
        time.sleep(0.05)
        self.key = None
        while self.key not in self.key_list and self.key not in self.extra_key_list:
            self.key = keyboard.get_hotkey_name()    #make sure keyboard is in english US :)
            #print('key = ', self.key)
            #print(key not in key_list, key not in extra_key_list, key not in key_list or key not in extra_key_list )
        time.sleep(0.05)
        print('key = ', self.key) 


    def save_image(self):
        if self.key is not None: 
            if len(self.key_list) != len(self.list_of_output_dirs):
                print("Error: Amount of Keys do Not Match Amount of directories")
            else:
                if self.key in self.key_list:
                    directory_index = self.key_list.index(self.key)
                    write_path = self.list_of_output_dirs[directory_index]
                    full_save_path = os.path.join(write_path, self.name)
                    cv.imwrite(full_save_path, self.im)
                    print(full_save_path)
                    return 1
                elif self.key in self.extra_key_list:
                    print('did not save, placeholder functionality')
                    if self.key == self.extra_key_list[0]:
                        return 0
                    else:
                        return -1
# key = None
# while key not in key_list and key not in extra_key_list:
#     key = keyboard.get_hotkey_name()    #make sure keyboard is in english US :)
#     print('key = ', key)
#     #print(key not in key_list, key not in extra_key_list, key not in key_list or key not in extra_key_list )
# time.sleep(0.05)
# print('key = ', key)             

#reading data lists
mask_folder = '1_m_greater' #'0_m_greater'
mask_list = os.listdir(os.path.join(ROOT_DIR, mask_folder))

less_mask_folder = '1_m_lesser' #'0_m_lesser'
less_mask_list = os.listdir(os.path.join(ROOT_DIR, less_mask_folder))

original_image_folder =  '1_sifted'#'0'
original_image_list = os.listdir(os.path.join(ROOT_DIR, original_image_folder))

#writing folders
output_folder_0 = '0_segments'
output_folder_1 = '1_segments'

#writing directories
output_dir_0 = os.path.join(ROOT_DIR, output_folder_0)
output_dir_1 = os.path.join(ROOT_DIR, output_folder_1)

#list of output directories
list_of_output_dirs = [output_dir_0, output_dir_1]
key_list = ['left', 'right']
extra_key_list = ['home', 'esc']


break_flag = 0

with open(os.path.join(ROOT_DIR,'last_image_no.txt')) as f:
    csvreader = csv.reader(f, delimiter=' ', quotechar='|')
    row = csvreader.__next__()

#start_image_no = int(row[0])
start_image_no = 0
print('starting at image no.', start_image_no)
end_image_no = 1 #Next aim for 300

for image_no in range(start_image_no,end_image_no):

    if original_image_list[image_no][-3:] not in ['jpg', 'bmp', 'png']:
        print("Error: skipped this .'"+ original_image_list[image_no][-3:] + "' file")
    else:
        #Load original image and masks from respective directories (masks are preloaded to be 1/4 original image size)
        #Note: Eventually, binarization can be done here if binarizationHSV.py is converted to a function or class
        original_img = load_image_from_file(os.path.join(original_image_folder,original_image_list[image_no]))
        mask = load_image_from_file(os.path.join(mask_folder,mask_list[image_no]), 1.0, cv.IMREAD_GRAYSCALE)
        lesser_mask = load_image_from_file(os.path.join(less_mask_folder,less_mask_list[image_no]), 1.0, cv.IMREAD_GRAYSCALE)
        
        print('image name :' , original_image_list[image_no] )
              
        #Create WeedMask Object for greater and lesser masks
        
        wMgreater = weedMask(mask)
        wMlesser = weedMask(lesser_mask)

        #Recreate filled in greater and lesser masks (without the extra contour pixels)
        greaterContours = cv.erode(wMgreater.drawContours(np.zeros((wMgreater.h, wMgreater.w), dtype = np.uint8)), np.ones((3,3)))
        lesserContours = cv.erode(wMlesser.drawContours(np.zeros((wMlesser.h, wMlesser.w), dtype = np.uint8)), np.ones((3,3)))
        
        #Create composite with greater and lesser mask
        composite = np.bitwise_and(greaterContours,np.bitwise_not(lesserContours))
        
        #Create WeedMaskSet Object for computing composite image
        wMset = weedMaskSet(composite)
        
        h, w = np.shape(original_img)[0], np.shape(original_img)[1]
        mask_scale = wMset.w/w
            
        
        n = 10
        if len(wMset.family_list) < n:
            n = len(wMset.family_list)
        
        bw = np.zeros((np.shape(mask)[0], np.shape(mask)[1], np.shape(original_img)[2]), dtype = np.uint8)
        
        for i_descending_order in range(n):   
            bw = wMset.drawFamilybySize(i_descending_order, bw)
            
        for i_descending_order in range(n):
            
            
            family = wMset.getFamilybySize(i_descending_order)
            xm, ym, wm, hm = family.getBbox() #default gets parentLeaf Boundingbox
            x0, y0, w0, h0 = int(xm/mask_scale), int(ym/mask_scale), int(wm/mask_scale), int(hm/mask_scale)
            
            temp = original_img.copy()
            temp = cv.resize(temp, (wMset.w, wMset.h))
            bw_2 = bw.copy()
            bw_2 = wMset.drawFamilybySize(i_descending_order, bw_2, 1)
            side_by_side = np.hstack((family.drawBbox(temp),family.drawBbox(bw_2)))
            
            show_wait(side_by_side, 0.9, destroy = 0, wait = 0)
            show_wait(original_img[y0:y0+h0, x0:x0+w0], scale = 1.0, name = 'close up', x = 500, y = 500 )
             
            
            if i_descending_order < 10: name_index = '0' + str(i_descending_order) 
            else: name_index = str(i_descending_order)
            name = original_image_list[image_no][:-4] + '_'+name_index + original_image_list[image_no][-4:]
            break_flag = get_key_and_save_image(original_img[y0:y0+h0, x0:x0+w0], name, list_of_output_dirs, key_list, extra_key_list).flag
            if break_flag == -1:
                break

            
    #save last image_no to restart the labelling process later
    with open(os.path.join(ROOT_DIR,'last_image_no.txt'), 'w') as f:
        csvwriter = csv.writer(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([image_no])
    
    if break_flag == -1:
        break
        


















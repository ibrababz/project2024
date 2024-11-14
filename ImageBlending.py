# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 21:27:12 2024

@author: i_bab
laplacian pyramids

"""
import cv2 as cv
import numpy as np
# import os
# import file
# from helper import show_wait

# ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))

# weed_folder = 'data2019\\1\\train1_contrast'
# mask_folder = 'data2019\\1\\train1_contrast_masks_clean'
# grass_folder = 'data2019\\0\\train1_0'

# weed_path = os.path.join(ROOT_DIR, weed_folder)
# mask_path = os.path.join(ROOT_DIR, mask_folder)
# grass_path = os.path.join(ROOT_DIR, grass_folder)

# mask_file_list = os.listdir(mask_path)
# img_list = os.listdir(weed_path)

# for weed_file in img_list:
#     for mask_file in mask_file_list:
#         if weed_file in mask_file:
#             grass_file = '000000 (2).bmp'

#             weed = cv.imread(os.path.join(weed_path, weed_file))
#             mask = cv.imread(os.path.join(mask_path, mask_file))
#             grass= cv.imread(os.path.join(grass_path, grass_file))
#             print(weed_file)
#             show_wait(weed)
#             show_wait(mask)
#             show_wait(grass)
#             # gaussian_3 = cv.GaussianBlur(weed.copy(), (0, 0), 2.0)
#             # weed = cv.addWeighted(weed.copy(), 1.5, gaussian_3, -0.5, 0)
#             # #show_wait(weed)
            
            
#             mask_list = [mask.copy()/255.0]
#             for i in range(5):
#                 rows,cols,dpt = mask_list[i].shape
#                 den = 2
#                 mask_list.append(cv.resize(mask.copy()/255.0, (cols//den, rows//den), interpolation = cv.INTER_AREA))
#             mask_list.reverse()
#             for im in mask_list:
#             #     show_wait(im)
#                 print(im.shape)
            
            
#             A = weed.copy()/255.0
#             B = grass.copy()/255.0
#             C = mask.copy()/255.0
            
#             # generate Gaussian pyramid for A
#             G = A.copy()
#             gpA = [G]
#             for i in range(6):
#                 G = cv.pyrDown(G)
#                 gpA.append(G)
#             # generate Gaussian pyramid for B
#             G = B.copy()
#             gpB = [G]
#             for i in range(6):
#                 G = cv.pyrDown(G)
#                 gpB.append(G)
#             # generate Gaussian pyramid for C
#             G = C.copy()
#             gpC = [G]
#             for i in range(5):
#                 G = cv.pyrDown(G)
#                 gpC.append(G)
#             gpC.reverse()    
#             # generate Laplacian Pyramid for A
#             lpA = [gpA[5]]
#             for i in range(5,0,-1):
#                 GE = cv.pyrUp(gpA[i])
#                 L = cv.subtract(gpA[i-1],GE)
#                 lpA.append(L)
#             # generate Laplacian Pyramid for B
#             lpB = [gpB[5]]
#             for i in range(5,0,-1):
#                 GE = cv.pyrUp(gpB[i])
#                 L = cv.subtract(gpB[i-1],GE)
#                 lpB.append(L)
#             # Now add left and right halves of images in each level
#             LS = []
#             for la,lb,gc in zip(lpA,lpB,mask_list):
#                 rows,cols,dpt = la.shape
#                 #print(la.shape)
#                 #show_wait(np.hstack([la,lb]))
#                 #ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
#                 ls = la*(gc) + (lb*(1 - gc))
#                 LS.append(ls)
#             # now reconstruct
#             ls_ = LS[0]
#             for i in range(1,6):
#                 ls_ = cv.pyrUp(ls_)
#                 ls_ = cv.add(ls_, LS[i])
#             # image with direct connecting each half
            
#             show_wait(ls_)
    
#             gaussian_3 = cv.GaussianBlur(ls_, (0, 0), 2.0)
#             unsharp_image = cv.addWeighted(ls_, 1.5, gaussian_3, -0.5, 0)
#             show_wait(unsharp_image)
#             blur = cv.GaussianBlur(unsharp_image, (3,3), 0)
#             show_wait(blur)
#         break
def blendWeedGrass(weed, mask, grass):
    mask_list = [mask.copy()/255.0]
    for i in range(5):
        rows,cols = mask_list[i].shape
        den = 2
        mask_list.append(cv.resize(mask.copy()/255.0, (cols//den, rows//den), interpolation = cv.INTER_AREA))
    mask_list.reverse()
    # for im in mask_list:
    # #     show_wait(im)
    #     print(im.shape)

    A = weed.copy()/255.0
    B = grass.copy()/255.0
    C = mask.copy()/255.0
    
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpB.append(G)
    # generate Gaussian pyramid for C
    G = C.copy()
    gpC = [G]
    for i in range(5):
        G = cv.pyrDown(G)
        gpC.append(G)
    gpC.reverse()    
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5,0,-1):
        GE = cv.pyrUp(gpA[i])
        L = cv.subtract(gpA[i-1],GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5,0,-1):
        GE = cv.pyrUp(gpB[i])
        L = cv.subtract(gpB[i-1],GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    LS = []
    for la,lb,gc in zip(lpA,lpB,mask_list):
        rows,cols,dpt = la.shape
        #print(la.shape)
        #show_wait(np.hstack([la,lb]))
        #ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
        gc = gc[...,None]
        ls = la*(gc) + (lb*(1 - gc))
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in range(1,6):
        ls_ = cv.pyrUp(ls_)
        ls_ = cv.add(ls_, LS[i])
    # image with direct connecting each half
    
    #show_wait(ls_)
    
    gaussian_3 = cv.GaussianBlur(ls_, (0, 0), 2.0)
    unsharp_image = cv.addWeighted(ls_, 1.5, gaussian_3, -0.5, 0)
    #show_wait(unsharp_image)
    #blur = cv.GaussianBlur(unsharp_image, (3,3), 0)
    #show_wait(blur)
    return unsharp_image

def blendWeedStackGrass(weedsHstack, masksHstack, grass):
    weed = weedsHstack
    mask = masksHstack
    shape = grass.shape
    dim = shape[1], shape[0]
    W, H = dim
    stackShape = weed.shape
    stackDim = stackShape[1], stackShape[0]
    stackW, stackH = stackDim
    no_weeds = int(np.round(stackW/W))
    #print("no_weeds = ", no_weeds)
    mask_list = [mask.copy()/255.0]
    for i in range(5):
        rows,cols = mask_list[i].shape
        den = 2
        mask_list.append(cv.resize(mask.copy()/255.0, (cols//den, rows//den), interpolation = cv.INTER_AREA))
    mask_list.reverse()

    A = weed.copy()/255.0
    B = grass.copy()/255.0
    C = mask.copy()/255.0
    
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpB.append(G)
    # generate Gaussian pyramid for C
    G = C.copy()
    gpC = [G]
    for i in range(5):
        G = cv.pyrDown(G)
        gpC.append(G)
    gpC.reverse()    
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5,0,-1):
        GE = cv.pyrUp(gpA[i])
        L = cv.subtract(gpA[i-1],GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5,0,-1):
        GE = cv.pyrUp(gpB[i])
        L = cv.subtract(gpB[i-1],GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    LS = []
    for la,lb,gc in zip(lpA,lpB,mask_list):
        shapeB = lb.shape
        dimB = shapeB[1], shapeB[0]
        WB, HB = dimB
        gc = gc[...,None]
        #ls = la*(gc) + (lb*(1 - gc))
        ls = la*(gc)
        lsPrime= 0.
        gcPrime = 0.
        for i in range(no_weeds):
            lsPrime = lsPrime + ls[:,i*WB:(i+1)*WB]
            gcPrime  = gcPrime  + gc[:,i*WB:(i+1)*WB]
        lsPrime = lsPrime + (lb*(1 - gcPrime))
        LS.append(lsPrime)
    # now reconstruct
    ls_ = LS[0]
    for i in range(1,6):
        ls_ = cv.pyrUp(ls_)
        ls_ = cv.add(ls_, LS[i])
    # image with direct connecting each half
    
    gaussian_3 = cv.GaussianBlur(ls_, (0, 0), 2.0)
    unsharp_image = cv.addWeighted(ls_, 1.5, gaussian_3, -0.5, 0)
    return unsharp_image #ls_
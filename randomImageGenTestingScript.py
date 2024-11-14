# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:25:43 2024

@author: i_bab
"""
#%%
import os
import file
import cv2 as cv
import numpy as np
from helper import show_wait
from augment_utils import affine_augments
from augment_utils import transform_pts, inverse_transform_matrix, to_center, to_origin, rotz, scale, trans, to_point
from augment_utils import warpImage
from imageUtils import CombinedWeeds
from dataLoad import generate_weed_list
import random
from imageBlending import blendWeedGrass
from adjustContrast import claheHsv
from copy import deepcopy
#from loss_functions import act_list_3D

from RandomSampler import RandSampler
from colortrans import transfer_lhm
    
#%%    
    
ROOT_DIR = os.path.abspath(os.path.join(file.ROOT_DIR, os.pardir))

src_folder = 'data2019\\1\\train1_contrast_masks_clean'
src_dir = os.path.join(ROOT_DIR, src_folder)

src_folder2 = 'data2019\\1\\train1_contrast'
src_dir2 = os.path.join(ROOT_DIR, src_folder2)

pts_file = 'mask_centers.json'
pts_path = os.path.join(src_folder, pts_file)

src_folder3 = "data2019\\0\\train1_0_grass"
src_path3 = os.path.join(ROOT_DIR, src_folder3)

Wo, Ho = 640, 480
W, H = 640, 480 #224, 224
W_grid, H_grid = 14, 14
dim = (W,H)
dim_grid = (W_grid, H_grid)

scale_x, scale_y = W/Wo, H/Ho

weed_list = generate_weed_list(src_dir, src_dir2, pts_file, scale_x, scale_y)
grass_list = [cv.resize(cv.imread(os.path.join(src_path3,x)), dim, interpolation = cv.INTER_AREA) for x in  os.listdir(src_path3)[:100]]

#%%
seq = affine_augments()
weed_list_copy = deepcopy(weed_list)
for i, weed in zip(range(3),weed_list_copy):
    print("weed.name = ", weed.name)
    print(weed.path)
    print("weed.im_name = ", weed.im_name)
    print(weed.im_path)
    mask = weed.load_mask(dim)
    im = weed.load_image(dim)
    weed.set_comp_map(dim, dim_grid)
    hmap = weed.get_comp_map_resize(dim)
    show_wait(np.hstack([cv.cvtColor(im,cv.COLOR_BGR2GRAY)/255., mask/255., hmap]))
    seq.seed_(i)
    temp = seq.deepcopy()
    im_aug = temp(image = im)
    temp = seq.deepcopy()
    mask_aug = temp(image = mask) 
    hmap_aug = seq(image = weed.get_comp_map())
    hmap_aug_resize = cv.resize(hmap_aug, dim, interpolation = cv.INTER_NEAREST)
    show_wait(np.hstack([cv.cvtColor(im_aug,cv.COLOR_BGR2GRAY)/255., mask_aug/255., hmap_aug_resize]))
#%%    
weed_list_copy = deepcopy(weed_list)

for i, weed in zip(range(10),weed_list_copy):
    
    angle = np.random.randint(0,90)
    dx, dy = np.random.randint(-W//3, W//3), np.random.randint(-H//3, H//3)
    dx_grid, dy_grid = dx/W*W_grid, dy/H*H_grid
    alpha = (-1)**np.random.randint(1,3)
    mask = weed.load_mask(dim)
    im = weed.load_image(dim)
    weed.set_comp_map(dim, dim_grid)
    hmap = weed.get_comp_map_resize(dim)
    show_wait(np.hstack([cv.cvtColor(im,cv.COLOR_BGR2GRAY)/255., mask/255., hmap]))
    
    
    x1, y1 = weed.region_list[0].get_center()
    t12 = to_center(dim, x1, y1)
    
    
    im = cv.warpAffine(im, t12[:2,:], (dim[0], dim[1]))
    mask = cv.warpAffine(mask, t12[:2,:], dim)
    weed.region_list[0].reset_center(W//2, H//2)
    weed.set_comp_map(dim, dim_grid)
    hmap = weed.get_comp_map_resize(dim)
    show_wait(np.hstack([cv.cvtColor(im,cv.COLOR_BGR2GRAY)/255., mask/255., hmap]))
    
    mask = weed.load_mask(dim)
    im = weed.load_image(dim)
    t23 = trans(dx,dy)
    t13 = t23@t12
    x3, y3 = transform_pts(x1, y1, t13)
    weed.region_list[0].reset_center(x3,y3) 
    weed.set_comp_map(dim, dim_grid)
    hmap = weed.get_comp_map_resize(dim)
    im = cv.warpAffine(im, t13[:2,:], dim)
    mask = cv.warpAffine(mask, t13[:2,:], dim)
    show_wait(np.hstack([cv.cvtColor(im,cv.COLOR_BGR2GRAY)/255., mask/255., hmap]))
    

    mask = weed.load_mask(dim)
    im = weed.load_image(dim)
    t23 = trans(dx,dy)
    flip = np.array([[alpha,0.,0], [0., 1., 0], [0, 0 , 1.]])
    rotflip = to_center(dim)@rotz(angle)@scale(-1)@inverse_transform_matrix(to_center(dim))
    axis = np.array([[1,0.,0], [0., -1., 0], [0, 0 , 1.]])
    
    
    t13 = t23@rotflip@t12
    x3, y3 = transform_pts(x1, y1, t13)
    weed.region_list[0].reset_center(W//2, H//2)
    weed.set_comp_map(dim, dim_grid)
    hmap = weed.get_comp_map()
    t23 = trans(dx_grid, dy_grid)
    rotflip =to_center(dim_grid)@rotz(angle)@scale(-1)@inverse_transform_matrix(to_center(dim_grid))
    t23prime = t23@rotflip
    hmap = cv.warpAffine(hmap, t23prime[:2,:], dim_grid)
    hmap = cv.resize(hmap, dim, interpolation = cv.INTER_NEAREST)
    
    im = cv.warpAffine(im, t13[:2,:], dim)
    mask = cv.warpAffine(mask, t13[:2,:], dim)
    show_wait(np.hstack([cv.cvtColor(im,cv.COLOR_BGR2GRAY)/255., mask/255., hmap]))
    
    

#%%
weed_list_copy = deepcopy(weed_list)
for i, weed in zip(range(5),weed_list_copy):
    
    angle = 45#np.random.randint(0,90) #random rotation angle
    dx, dy = 0, 0# np.random.randint(-W//2, W//2), np.random.randint(-H//2, H//2) #random translation
    # dx, dy = W//2, H//2 #TEST
    dx_grid, dy_grid = dx/W*W_grid, dy/H*H_grid #random translation for label
    scaleX = (-1)**np.random.randint(1,3) #random flip on x-axis
    scaleY = 1
    dxy = (dx, dy)
    scaleXY = (scaleX, scaleY)
    mask = weed.load_mask(dim)
    count1 = np.count_nonzero(mask)
    
    im = weed.load_image(dim)
    weed.set_comp_map(dim, dim_grid)
    hmap = weed.get_comp_map_resize(dim)
    show_wait(np.hstack([cv.cvtColor(im,cv.COLOR_BGR2GRAY)/255., mask/255., hmap]))
    x1, y1 = weed.region_list[0].get_center()
    t12 = to_center(dim, x1, y1)
    
    mask = weed.load_mask(dim)
    im = weed.load_image(dim)
    t34 = trans(dx,dy)
    r23 = inverse_transform_matrix(to_origin(dim, (W//2, H//2)))@rotz(angle)@scale(scaleX, scaleY)@to_origin(dim, (W//2, H//2))
    #above line first brings image to origin (0,0) using "inverse...(to_center...), 
    #performs rotation and flipping then returns image to center
    
    t13 = r23@t12
    t14 = t34@r23@t12
    x3, y3 = transform_pts(x1, y1, t14) 
    weed.region_list[0].reset_center(W//2, H//2)
    weed.set_comp_map(dim, dim_grid)
    hmap = weed.get_comp_map()
    #Reset heatmap center to original position for reuse
    weed.region_list[0].reset_center(x1, y1)
    t34 = trans(dx_grid, dy_grid)
    
    r23 =inverse_transform_matrix(to_origin(dim_grid, (W_grid//2, H_grid//2)))@rotz(angle)@scale(scaleX, scaleY)@to_origin(dim_grid, (W_grid//2, H_grid//2))
    #above line first brings label to origin (0,0) using "inverse...(to_center...), 
    #performs rotation and flipping then returns label to center
    
    t24 = t34@r23
    hmap3 = cv.warpAffine(hmap, r23[:2,:], dim_grid)
    hmap3 = cv.resize(hmap3, dim, interpolation = cv.INTER_NEAREST)
    
    hmap4 = cv.warpAffine(hmap, t24[:2,:], dim_grid)
    hmap4 = cv.resize(hmap4, dim, interpolation = cv.INTER_NEAREST)
    
    im3 = cv.warpAffine(im, t13[:2,:], dim)
    mask3 = cv.warpAffine(mask, t13[:2,:], dim)
    count3 = np.count_nonzero(mask3) 
    #count mask pixels first at rotation because it increases pixel number
    
    im4 = cv.warpAffine(im, t14[:2,:], dim)
    mask4 = cv.warpAffine(mask, t14[:2,:], dim)
    count4 = np.count_nonzero(mask4)
    #count mask pixels again first after translation
            
    show_wait(np.hstack([cv.cvtColor(im3,cv.COLOR_BGR2GRAY)/255., mask3/255., hmap3])) #rotated
    show_wait(np.hstack([cv.cvtColor(im4,cv.COLOR_BGR2GRAY)/255., mask4/255., hmap4])) #rotated and translated
    #Reset heatmap center to original position
    # weed.setPlaced(dim, dim_grid, dxy, angle, scaleXY)
    # imPlaced, maskPlaced, hmapPlaced = weed.getPlaced()
    # countRot, countTrans = weed.getPlacedCounts()
    # hmapPlacedResized = cv.resize(hmapPlaced, dim, interpolation = cv.INTER_NEAREST)
    
    print("1- orig.(%):", np.round(count3/(H*W)*100), " red (%):", np.round((count3-count4)/(H*W)*100), " ratio:", np.round((count4/count3), 2) )
    # print("2- orig.(%):", np.round(countRot/(H*W)*100), " red (%):", np.round((countRot-countTrans)/(H*W)*100), " ratio:", np.round((countTrans/countRot), 2) )

    # show_wait(np.hstack([cv.cvtColor(imPlaced,cv.COLOR_BGR2GRAY)/255., maskPlaced/255., hmapPlacedResized]))

#%%
# def getWarpMatrix(iCenter, dim, angle = 0, scaleXY = (1,1), transXY = (0,0)):
#     #this returns the transformation matrix of the image rotated/scaled at iCenter
#     #then translated therefrom
#     scaleX, scaleY = scaleXY
#     tx, ty = transXY
#     return trans(tx,ty)@to_center((0,0),dim)@scale(scaleX, scaleY)@rotz(angle)@to_origin(iCenter)

# def warpImage(im, iCenter, dim, angle = 0, scaleXY = (1,1), transXY = (0,0)):
#     t12 = getWarpMatrix(iCenter, dim, angle, scaleXY, transXY)
#     return cv.warpAffine(im,  t12[:2,:], dim)

# def getTransFromPolar(r, angleDeg):
#     angleRad = angleDeg*np.pi/180
#     return trans(r*np.cos(angleRad), r*np.sin(angleRad))

# def getAngleBounds(n):
#     angleList = [0.]
#     for i in range(n):
#         angleList.append((i+1)*360/n)
#     return angleList               
             
#%%
weed_list_copy = deepcopy(weed_list)
for i, weed in zip(range(10),weed_list_copy):
    weedCenter = weed.region_list[0].get_center()
    weedCenter_grid = weedCenter[0]*W_grid//W, weedCenter[1]*H_grid//H
    mask = weed.load_mask(dim)
    im =  weed.load_image(dim)
    weed.set_comp_map(dim, dim_grid)
    hmap = weed.get_comp_map()
    hmapRes = cv.resize(hmap, dim, interpolation = cv.INTER_NEAREST)
    show_wait(np.hstack([cv.cvtColor(im,cv.COLOR_BGR2GRAY)/255., mask/255., hmapRes]))
    
    mask1 = cv.warpAffine(mask, to_origin(weedCenter)[:2, :], dim)
    im1 = cv.warpAffine(im, to_origin(weedCenter)[:2, :], dim)
    hmap1 = cv.warpAffine(hmap, to_origin(weedCenter_grid)[:2, :], dim_grid)
    hmapRes1 = cv.resize(hmap1, dim, interpolation = cv.INTER_NEAREST)
    show_wait(np.hstack([cv.cvtColor(im1,cv.COLOR_BGR2GRAY)/255., mask1/255., hmapRes1]))
    
    mask2 = cv.warpAffine(mask, (rotz(45)@to_origin(weedCenter))[:2,:], dim)
    im2 = cv.warpAffine(im,  (rotz(45)@to_origin(weedCenter))[:2,:], dim)
    hmap2 = cv.warpAffine(hmap, (rotz(45)@to_origin(weedCenter_grid))[:2, :], dim_grid)
    hmapRes2 = cv.resize(hmap2, dim, interpolation = cv.INTER_NEAREST)
    show_wait(np.hstack([cv.cvtColor(im2,cv.COLOR_BGR2GRAY)/255., mask2/255., hmapRes2]))
    
    mask3 = cv.warpAffine(mask, (scale(-1)@rotz(45)@to_origin(weedCenter))[:2,:], dim)
    im3 = cv.warpAffine(im,  (scale(-1)@rotz(45)@to_origin(weedCenter))[:2,:], dim)
    hmap3 = cv.warpAffine(hmap, (scale(-1)@rotz(45)@to_origin(weedCenter_grid))[:2, :], dim_grid)
    hmapRes3 = cv.resize(hmap3, dim, interpolation = cv.INTER_NEAREST)
    show_wait(np.hstack([cv.cvtColor(im3,cv.COLOR_BGR2GRAY)/255., mask3/255., hmapRes3]))
    
    mask4 = cv.warpAffine(mask, (to_center((0,0),dim)@scale(-1)@rotz(45)@to_origin(weedCenter))[:2,:], dim)
    im4 = cv.warpAffine(im,  (to_center((0,0),dim)@scale(-1)@rotz(45)@to_origin(weedCenter))[:2,:], dim)
    hmap4 = cv.warpAffine(hmap, (to_center((0,0),dim_grid)@scale(-1)@rotz(45)@to_origin(weedCenter_grid))[:2, :], dim_grid)
    hmapRes4 = cv.resize(hmap4, dim, interpolation = cv.INTER_NEAREST)
    show_wait(np.hstack([cv.cvtColor(im4,cv.COLOR_BGR2GRAY)/255., mask4/255., hmapRes4]))
    countRot = np.count_nonzero(mask4)
   
    mask6 = warpImage(mask, weedCenter, dim, 45, (-1,1), (W//4, H//4))
    im6 = warpImage(im, weedCenter, dim, 45, (-1,1), (W//4, H//4))
    hmap6 = warpImage(hmap, weedCenter_grid, dim_grid, 45, (-1,1), (W_grid//4, H_grid//4))
    hmapRes6 = cv.resize(hmap6, dim, interpolation = cv.INTER_NEAREST)
    countTrans = np.count_nonzero(mask6)
    print("2- orig.(%):", np.round(countRot/(H*W)*100), " red (%):", np.round((countRot-countTrans)/(H*W)*100), " ratio:", np.round((countTrans/countRot), 2) )
    show_wait(np.hstack([cv.cvtColor(im6,cv.COLOR_BGR2GRAY)/255., mask6/255., hmapRes6]))

    
#%%
weed_list_copy = deepcopy(weed_list)
for i, weed in zip(range(10),weed_list_copy):
    weedCenter = weed.region_list[0].get_center()
    weedCenter_grid = weedCenter[0]*W_grid//W, weedCenter[1]*H_grid//H
    mask = weed.load_mask(dim)
    im =  weed.load_image(dim)
    weed.set_comp_map(dim, dim_grid)
    hmap = weed.get_comp_map()
    mask6 = warpImage(mask, weedCenter, dim, 45, (-1,1))
    im6 = warpImage(im, weedCenter, dim, 45, (-1,1))
    hmap6 = warpImage(hmap, weedCenter_grid, dim_grid, 45, (-1,1))
    hmapRes6 = cv.resize(hmap6, dim, interpolation = cv.INTER_NEAREST)
    show_wait(np.hstack([cv.cvtColor(im6,cv.COLOR_BGR2GRAY)/255., mask6/255., hmapRes6]))
#%%   
sample_list = deepcopy(weed_list)

for i in range(1):
    nList = len(sample_list)
    nSample = random.randint(1,3)
    while nSample > nList:
        nSample = random.randint(1,3)
    cmb_weed1 = CombinedWeeds(random.sample(sample_list, nSample), seq)
    cmb_weed1.augment_weeds(dim, dim_grid, i)
    cmb_weed1.combine3d()
    cmb_weed1.flatten_weeds()
    show_wait(cmb_weed1.im)
    show_wait(cmb_weed1.mask)
#%%
b = ['a', 'b','c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
c = list(range(len(b)))
d = random
nList = len(c)
i=0
print(b)
while nList:
    #d.seed(i)
    if nList == 1:
        nSample = 1
    else:
        nSample = d.randint(1,3)
        while nSample > nList:
            nSample = d.randint(1,3)
    Sampled = d.sample(c, nSample)
    Sampled_values = [b[i] for i in Sampled]
    #print('sampled = ', Sampled)
    print('sampled values = ', Sampled_values)
    
    for i in sorted(Sampled, reverse = True):
        #print('popping =', i, ' value = ', b[i])
        b.pop(i)
        #print(b)
    print(b)
    c = list(range(len(b)))
    nList = len(b)
    i+=1

#%%

b = ['a', 'b','c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
myRandomSampler = RandSampler(b, 4)

for i, sample in zip(range(10),myRandomSampler):
    print(sample)
    
#%%
seq = affine_augments()
sample_list = deepcopy(weed_list)
weedRandomSampler = RandSampler(sample_list,2)
for i,sample in zip(range(4), weedRandomSampler):
    cmb_weed1 = CombinedWeeds(sample, seq)
    cmb_weed1.augment_weeds(dim, dim_grid, i)
    cmb_weed1.combine3d()
    cmb_weed1.flatten_weeds()
    cmb_weed1.flatten_maps()
    im = cmb_weed1.im
    mask = cmb_weed1.mask
    wmap = cmb_weed1.map
    mask_clr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    wmap_clr = cv.cvtColor(cv.resize(wmap[:,:], dim, interpolation = cv.INTER_NEAREST), cv.COLOR_GRAY2BGR)
    show_wait(np.hstack([im,mask_clr,(255*wmap_clr).astype(np.uint8)]))
#%%    
    
src_folder3 = "data2019\\0\\train1_0_grass"
src_path3 = os.path.join(ROOT_DIR, src_folder3)
grass_list = [cv.resize(cv.imread(os.path.join(src_path3,x)), dim, interpolation = cv.INTER_AREA) for x in  os.listdir(src_path3)[:100]]

seq = affine_augments()
sample_list = deepcopy(weed_list)
grass_list_copy = deepcopy(grass_list)
weedRandomSampler = RandSampler(sample_list,2)
grassRandomSampler = RandSampler(grass_list_copy, 1)
for i, sample, grass in zip(range(10), weedRandomSampler, grassRandomSampler):
    cmb_weed1 = CombinedWeeds(sample, seq)
    cmb_weed1.augment_weeds(dim, dim_grid, i)
    cmb_weed1.combine3d()
    cmb_weed1.flatten_weeds()
    cmb_weed1.flatten_maps()
    im = cmb_weed1.im
    mask = cmb_weed1.mask
    blended = blendWeedGrass(im, mask, grass[0])
    clip_limit = 1
    blended_nrm = np.zeros(blended.shape, dtype = np.uint8)
    claheHsv(cv.normalize(blended,blended_nrm, alpha = 0, beta= 255, norm_type = cv.NORM_MINMAX, dtype = cv. CV_8UC1), clip_limit, (8,8))
    print("weed len =" , weedRandomSampler.lenModList, " grass len =" , grassRandomSampler.lenModList)
    show_wait(blended)
    #show_wait(blended_nrm)

#%%
#with colortrans
seq = affine_augments()
sample_list = deepcopy(weed_list)
grass_list_copy = deepcopy(grass_list)
weedRandomSampler = RandSampler(sample_list,2)
grassRandomSampler = RandSampler(grass_list_copy, 1)
for i, sample, grass in zip(range(10), weedRandomSampler, grassRandomSampler):
    cmb_weed1 = CombinedWeeds(sample, seq)
    cmb_weed1.augment_weeds(dim, dim_grid, i)
    cmb_weed1.combine3d()
    cmb_weed1.flatten_weeds()
    cmb_weed1.flatten_maps()
    im = cmb_weed1.im
    mask = cmb_weed1.mask
    blended = blendWeedGrass(im, mask, grass[0])
    clip_limit = 1
    blended_nrm = np.zeros(blended.shape, dtype = np.uint8)
    claheHsv(cv.normalize(blended,blended_nrm, alpha = 0, beta= 255, norm_type = cv.NORM_MINMAX, dtype = cv. CV_8UC1), clip_limit, (8,8))
    print("weed len =" , weedRandomSampler.lenModList, " grass len =" , grassRandomSampler.lenModList)
    show_wait(blended)

#%%
weed_list_copy = deepcopy(weed_list)
grass_list_copy = deepcopy(grass_list)

weedRandomSampler = RandSampler(weed_list_copy,3)
grassRandomSampler = RandSampler(grass_list_copy, 1)

for i, sample, grass in zip(range(10), weedRandomSampler, grassRandomSampler):
    myCombinedWeeds = CombinedWeeds(sample, grass[0])
    myCombinedWeeds.loadImages(dim)
    myCombinedWeeds.loadMasks()
    myCombinedWeeds.loadMaps(dim_grid)
    myCombinedWeeds.transferColors()
    myCombinedWeeds.setRotFlipRanges()
    myCombinedWeeds.rotateFlipWeeds()
    myCombinedWeeds.setMinMaxTransPercent(1/6, 1/2)
    myCombinedWeeds.setPlacementMinPercent(0.5)
    myCombinedWeeds.placeWeeds()
    



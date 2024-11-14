
import numpy as np
import os
import json
import random
from imageUtils import Original_image
import matplotlib.pyplot as plt
from imgaug.augmentables.heatmaps import HeatmapsOnImage

def generate_data_list(data_path, file_points_path, scale_x = 1., scale_y = 1.):
    #the file_points uses the orignal image dimensions whereas the image folder
    #can use downsampled versions for which the scale_x,y must be adjusted
    with open(os.path.join(data_path, file_points_path), 'r') as f:
        via = json.load(f)    
    
    oimg_list = []    
    for fid in via['_via_img_metadata']:
        
        file_name = via['_via_img_metadata'][fid]['filename']
        file_path = os.path.join(data_path, file_name)

        if not os.path.isfile(file_path):
            print('File not found! %s' %(file_path))
            continue

        oim = Original_image(file_path)
        for region in via['_via_img_metadata'][fid]['regions']:
            oim.set_region_from_attr(region['shape_attributes'], scale_x, scale_y)

        oimg_list.append(oim)

    return oimg_list

def image_list_from_batch(batch, dim = None):
    im_list =[]
    for oim in batch:
        im_list.append(oim.load_image(dim)) #ADD [...,::-1] TO MAKE IT RGB 
    return im_list

def name_list_from_batch(batch):
    name_list = []
    for oim in batch:
        name_list.append(oim.name)
    return name_list    

def flatten_map_v2(im_map, inv = 0):
    max_map = np.max(im_map, axis = (0,1), keepdims=True) #get maximum of each channel
    min_map = np.min(im_map, axis = (0,1), keepdims=True)
    im_map = (im_map-min_map)/(max_map - min_map + 1e-15)
    if not inv:
        im_map = np.maximum.reduce(im_map, axis=2, keepdims = True)
    else:
        im_map = np.minimum.reduce(im_map, axis =2, keepdims = True)
    return im_map

def flat_map_list_v2(map_list, inv = 0):
    flat_list = []
    for map1 in map_list:
        flat = flatten_map_v2(map1, inv)
        flat_list.append(flat)
    return flat_list

def weight_list_3D(map_list, lo_val = 0.0):
    weight_list = []
    for mapi in map_list:
        weight_map=  (1.-lo_val)*(1-mapi) + lo_val #scale so that black cell weights of inverse map are not zero
        weight_list.append(weight_map)
    return weight_list

def scale_map(im_map):
    max_map = np.max(im_map, axis = (0,1), keepdims=True) #get maximum of each channel
    min_map = np.min(im_map, axis = (0,1), keepdims=True)
    im_map = (im_map-min_map)/(max_map - min_map + 1e-15)

    return im_map

def scale_map_list_3D(map_list):
    scaled_map_list = []
    for mapi in map_list:
        scaled = scale_map(mapi)
        scaled_map_list.append(scaled)
    return scaled_map_list

def map_list_from_batch(batch, dim, dim_grid):
    map_list = []
    for oim in batch:
        oim.set_comp_map(dim, dim_grid)
        map_list.append(oim.get_comp_map_3d())
    return map_list

def map2heatmap_list(map_list, dim):
    heatmap_list = []
    for mapi in map_list:
        heatmap_list.append(HeatmapsOnImage(mapi, dim))
    return heatmap_list

def heatmap2map_list(heatmap_list):
    map_list = []
    for heatmap in heatmap_list:
        map_list.append(heatmap.get_arr())
    return map_list

def process_batch_3D(batch, dim, dim_grid, seq = None, lo_val = 0.0):
    map_list = map_list_from_batch(batch, dim, dim_grid)
    if seq is not None:
        heatmap_aug_list = seq(heatmaps = map2heatmap_list(map_list, dim))
        map_aug_list = heatmap2map_list(heatmap_aug_list)
    else:
        map_aug_list = map_list
    
    map_aug_list = scale_map_list_3D(map_aug_list)
    weight_list = weight_list_3D(map_aug_list, lo_val)
        
    return map_aug_list, weight_list, seq    

def draw_batch(X_batch, y_batch_koi, y_batch_bboi, batch_names, color1 = (255,0,0), color2 =(0,0,255)):
    #print('drawingBatch')    
    X_draw = []
    for img, pts, bboi, name in zip(X_batch, y_batch_koi, y_batch_bboi, batch_names):
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #bbox = bboi.clip_out_of_image()
        #print(name)
        #print(bboi.to_xyxy_array())
        img = bboi.draw_on_image(img, color1, size = 5)
        img = pts.draw_on_image(img, color2, size=5)
        
        X_draw.append(img)
    return X_draw

def show_batch(X_batch, batch_names = None, size = 40.):
    #print('ShowingBatch')    
    _, axs = plt.subplots(1, len(X_batch), figsize=(40, 40))
    if len(X_batch)>1:
        axs = axs.flatten()
    else:
        axs = [axs]
    if batch_names is not None:
        for img, name, ax in zip(X_batch, batch_names, axs):
            h,w = np.shape(img)[0], np.shape(img)[1]
            img = img[...,::-1]
            ax.text(w/3, h/6, name[-13:-4], size = size, weight = 'bold', color = 'r')
            ax.imshow(img)
    else:
        for img, ax in zip(X_batch, axs):
            h,w = np.shape(img)[0], np.shape(img)[1]
            img = img[...,::-1]
            ax.text(w/3, h/6, '', size = size, weight = 'bold', color = 'r')
            ax.imshow(img)

    plt.show()
    
class generate_batch():
    
    def __init__(self, XY_train, batch_size, seed = 4):
        #print('constructing class')
        if seed:
            random.Random(seed).shuffle(XY_train)
        self.XY_train = XY_train
        #print('type of XY_train = ', type(self.XY_train))
        self.batch_size = batch_size
        self.steps = int(np.ceil(len(XY_train)/batch_size))
        self.extra_step_size = len(XY_train)%batch_size
        self.index = 0
    def __next__(self):
        #calls next item (batch) in sequence
        if self.index < self.steps:
            if self.index == self.steps-1 and self.extra_step_size:
                batch = self.XY_train[self.index*self.batch_size:self.index*self.batch_size + self.extra_step_size]
                #print('last batch, batch index = ', self.index)
            else:
                #print('normal batch, batch index = ', self.index)
                batch = self.XY_train[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index +=1
            return batch
        else:
            raise StopIteration
    
    def __iter__(self):
        #returns iterator object
        return self
    
    


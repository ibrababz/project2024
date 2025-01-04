import numpy as np
import cv2 as cv
from helper import adjust_number
import os
import json
import random
from imageUtils import Original_image, Original_weed, CombinedWeed
import matplotlib.pyplot as plt
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from RandomSampler import RandSampler
from copy import deepcopy


def makeNewDir(iSaveDir, iIdx=0, iSuffix='_test_'):
    wSaveDir = iSaveDir + iSuffix + adjust_number(iIdx, 2)
    if not os.path.exists(wSaveDir):
        print("\nMaking new directory:\n%s\n"%wSaveDir)
        return wSaveDir
    else:
        return makeNewDir(iSaveDir, iIdx+1, iSuffix)
    
def makeNewDirV2(iParDir, iFolderName, iKey, iIdx=0, iPrefix='test'):
    wPrefix = iPrefix +'_'+ adjust_number(iIdx, 2)
    wNew=True
    for wFile in os.listdir(iParDir)[::-1]: #reverse it because it's usually order 
                                            #so might as well search from bottom up
        try:
            if iKey == wFile.split('_')[2] and wPrefix == '_'.join(wFile.split('_')[0:2]):
                wNew=False
                break
        except IndexError:
            pass
        
    if wNew:
        return os.path.join(iParDir, "_".join([wPrefix, iKey, iFolderName]))
    else:
        return makeNewDirV2(iParDir, iFolderName, iKey, iIdx+1, iPrefix)
    
 
            

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

class file_writing():
    def __init__(self, iFile):
        self.file = iFile
    def write_file(self, string):
        self.file.write(string)
        self.file.write('\n')
        
        
def save_original_image_names(save_dir, XY_data, subset_name, label ='1'): #just for orignal image objects
    with open(os.path.join(save_dir, 'original_image_name_' + subset_name+ '_'+label+ '.txt'), 'w' ) as file:
        fwriting = file_writing(file)
        for XY in XY_data:
            fwriting.write_file(XY.name)
            
def getMapListsFromBatch(iBatch):
    mapBatch = []
    for wDataObj in iBatch:
        mapBatch.append(wDataObj.getMapList())  
    return np.array(mapBatch, dtype = object).T.tolist()
    # return mapBatch

def getNameListFromBatch(iBatch):
    oNameBatch = []
    for wDataObj in iBatch:
        oNameBatch.append(wDataObj.getNamesList())  
    return oNameBatch

def getImageListFromBatch(iBatch, iNorm = 0):
    imageBatch =[]
    for wDataObj in iBatch:
        wImage = wDataObj.getImage()
        if iNorm:
            wImage = wImage/iNorm
        imageBatch.append(wImage)
    return imageBatch

def genDataFilesFromOrgImBatchGen(iOrgImBatchGen, iImDim, iMapDims, iDestPath = None, iNameType = 'train', iNormal = 0):
    os.makedirs(iDestPath, exist_ok=True)
    wSize = iOrgImBatchGen.getDataLen()
    k = 0
    for batch in iOrgImBatchGen:
        batchImages = image_list_from_batch(batch, None)
        batchNames = name_list_from_batch(batch)
        wBatchLen = len(batch)
        for i in range(wBatchLen):
            wDataName = iNameType + '_'+adjust_number(k*wBatchLen + i)
            wDataDir = os.path.join(iDestPath, wDataName)
            # print(wDataDir)
            os.makedirs(wDataDir, exist_ok=True)
            wImageName = wDataName + '.npy'
            wWriteImagePath = os.path.join(wDataDir,wImageName)
            wImage = batchImages[i]
            if iNormal:
                wImage = wImage/iNormal
            np.save(wWriteImagePath, wImage)
            with open(os.path.join(wDataDir, "source_directory.txt"), 'w') as file:
                fwriting = file_writing(file)
                wSrcPath = os.path.dirname(batch[i].get_path())
                fwriting.write_file(wSrcPath)            
            wMapDir = os.path.join(wDataDir, "maps") 
            os.makedirs(wMapDir, exist_ok=True)
            for j, wDim in zip(range(len(iMapDims)), iMapDims):
                mapList, _, _ = process_batch_3D([batch[i]], iImDim, wDim)
                wMapName = wDataName +'_' +'map_'+ adjust_number(j, 1)+  '.npy'
                wWriteMapPath = os.path.join(wMapDir,wMapName)
                np.save(wWriteMapPath, mapList[0])
            wNamesDir = os.path.join(wDataDir, "names")
            os.makedirs(wNamesDir, exist_ok=True)
            with open(os.path.join(wNamesDir, "names.txt"), 'w') as file:
                fwriting = file_writing(file)
                #print(len(batchNames[0]))
                wName = batchNames[i]
                fwriting.write_file(wName)
            #     print(wWriteMapPath)     
            # show_wait(np.hstack(batchImages), 2)
            # show_wait(np.hstack(batchMasks), 2)
        k+=1

            
def genDataFiles(iWeedBatchGen, iDestPath = None, iNameType = 'train', iUintFlag = 0):
    wSize = iWeedBatchGen.getSize()
    os.makedirs(iDestPath, exist_ok=True)
    for i, batch in zip(range(wSize), iWeedBatchGen):
        batchNames, batchImages, batchMasks, batchMaps = batch
        #print("myWeedBatchGen.getCounter() = %s" %iWeedBatchGen.getCounter())
        #print("batchImages[0].dtype %s" %batchImages[0].dtype)
        wDataName = iNameType + '_'+adjust_number(i)
        wDataDir = os.path.join(iDestPath, wDataName)
        # print(wDataDir)
        os.makedirs(wDataDir, exist_ok=True)
        wImageName = wDataName + '.npy'
        wWriteImagePath = os.path.join(wDataDir,wImageName)
        wImage = batchImages[0]
        if iUintFlag:
            wImage = np.uint8(np.round(wImage*255., 0))
        np.save(wWriteImagePath, wImage)
        # print(wWriteImagePath)
        wMapDir = os.path.join(wDataDir, "maps") 
        os.makedirs(wMapDir, exist_ok=True)
        for j, mapList in zip(range(len(batchMaps)),batchMaps):
            wMapName = wDataName +'_' +'map_'+ adjust_number(j, 1)+  '.npy'
            wWriteMapPath = os.path.join(wMapDir,wMapName)
            np.save(wWriteMapPath, mapList[0])
        wNamesDir = os.path.join(wDataDir, "names")
        os.makedirs(wNamesDir, exist_ok=True)
        with open(os.path.join(wNamesDir, "names.txt"), 'w') as file:
            fwriting = file_writing(file)
            #print(len(batchNames[0]))
            for wName in batchNames[0]:
                fwriting.write_file(wName)
        #     print(wWriteMapPath)     
        # show_wait(np.hstack(batchImages), 2)
        # show_wait(np.hstack(batchMasks), 2)

def loadDataFilesAsObjects(iSrcPath, iStopFlag = 0):
    oDataObjectList = []
    for i, wFolder in zip(range(len(os.listdir(iSrcPath))),os.listdir(iSrcPath)): 
        if iStopFlag and i == iStopFlag:
            break
        wDataPath = os.path.join(iSrcPath,wFolder)
        wNameList = []
        wImSrcPath = None
        for wDir in os.listdir(wDataPath):
            if wDir.split('.')[-1] == 'npy':
                wImage = np.load(os.path.join(wDataPath,wDir))
            elif wDir.split('.')[-1] == 'txt':
                wFilePath = os.path.join(wDataPath, wDir)
                wFile = open(wFilePath, "r")
                for wLine in wFile:
                    wImSrcPath = wLine.strip('\n')
                wFile.close()
            elif wDir == 'maps':
                wMapDir = os.path.join(wDataPath, wDir)
                wMapList= []
                for wMapFile in os.listdir(wMapDir):
                    wMap = np.load(os.path.join(wMapDir,wMapFile))
                    wMapList.append(wMap)
            elif wDir == 'names':
                wNamesDir = os.path.join(wDataPath, wDir)
                wNamesFile = os.listdir(wNamesDir)[0]
                wNamesFilePath = os.path.join(wNamesDir, wNamesFile)
                wFile = open(wNamesFilePath, "r")
                for wLine in wFile:
                    wNameList.append(wLine.strip('\n'))
                wFile.close()
                
        oDataObjectList.append(LoadedDataObject(wImage, wMapList, wImSrcPath, wNameList))

    return oDataObjectList
            
class LoadedDataObject:
    def __init__(self, iImage, iMapList, iLoadDir, iNamesList = []):
        self.mImage = iImage
        self.mMapList = iMapList
        self.mNoDims = len(self.mMapList)
        self.mShape = self.mImage.shape
        self.mDim = self.mShape[1], self.mShape[0]
        self.mNamesList = iNamesList
        self.mLoadDir = iLoadDir
        
    def getImage(self):
        return self.mImage
    def getMapList(self):
        return self.mMapList
    def getNoDims(self):
        return self.mNoDims
    def getMapAtIndex(self, iIndex):           
        return self.mMapList[iIndex]
    def getDim(self):
        return self.mDim
    def getShape(self):
        return self.mShape
    def getNamesList(self):
        return self.mNamesList
    def getLoadDir(self):
        return self.mLoadDir

class RandomWeedBatchGenerator:
    def __init__(self, iBatchSize, iWeedData, iDimGridList, iGrassList, iWeedSampleSize, iSamplerSeed, iColorTrans=True, iBlend=True):
        self.mBatchSize = iBatchSize
        self.mWeedData = iWeedData
        self.mWeedData.setDimGridList(iDimGridList)        
        self.mWeedData.addAllGridDims()
        self.mWeedData.setGrassList(iGrassList)
        self.mSampleSize = iWeedSampleSize
        self.mWeedData.setAllRandomSamplers(self.mSampleSize, iSamplerSeed)
        self.mColorTrans = iColorTrans
        self.mBlend=iBlend
        self.mCounter = 0
    
    def reseedAllRandomSamplers(self, iSamplerSeed):
        self.mWeedData.setAllRandomSamplers(self.mSampleSize, iSamplerSeed)
    
    def setBatchDim(self, iBatchDim):
        #sets the networks input dimension size different from the native image sizes
        self.mBatchDim = iBatchDim
    def getBatchDim(self):
        return self.mBatchDim
    def setTranLimits(self, iMinMaxTransPercent, iMinRatio, iBuffer):
        self.mMinMaxTransPrecent = iMinMaxTransPercent
        self.mMinRatio = iMinRatio
        self.mBuffer = iBuffer
        
    def __next__(self):
        wNameRand, wImRand, wMaskRand, wCenterRand, wMapRandList, wGridCenterRandList, wGrassRand = self.mWeedData.getAllRandomSamplers()
        self.mNameBatch = [] #for debugging
        self.mImBatch = []
        self.mMaskBatch = [] #for debugging
        self.mMapListBatch = []
        
        for i in range(self.getBatchSize()):
            wSuccessFlag = False
            while (not wSuccessFlag):
                names = next(wNameRand)
                images = next(wImRand)
                masks = next(wMaskRand)
                centers = next(wCenterRand)
                map_list = []
                for j in range(len(wMapRandList)):
                    wMapRandj = wMapRandList[j]
                    map_list.append(next(wMapRandj))    
                grid_center_list =[]
                for j in range(len(wGridCenterRandList)):
                    wGridCenterRandj  =wGridCenterRandList[j]
                    grid_center_list.append(next(wGridCenterRandj))
                grass = next(wGrassRand)
                wCombinedWeeds = CombinedWeed(names, images, masks, centers, map_list, grid_center_list, grass[0])
                wCombinedWeeds.transferColors(self.mColorTrans)
                wCombinedWeeds.setSeed(i + self.mCounter)
                wCombinedWeeds.setRotFlipRanges()
                wCombinedWeeds.rotateFlipWeeds()
                wCombinedWeeds.setMinMaxTransPercent(self.mMinMaxTransPrecent[0], self.mMinMaxTransPrecent[1])
                wCombinedWeeds.setPlacementMinRatio(self.mMinRatio)
                wCombinedWeeds.setBuffer(self.mBuffer)
                wCombinedWeeds.placeWeeds()
                wSuccessFlag = wCombinedWeeds.isSuccess()
            wCombinedWeeds.setPosedMasks()
            wCombinedWeeds.flattenWeeds()
            wCombinedWeeds.setBlend(self.mBlend)
            wCombinedWeeds.setStackedMaps()
            
            self.mNameBatch.append(names) #for debugging
            self.mImBatch.append(wCombinedWeeds.getBlend(self.mBatchDim))
            self.mMaskBatch.append(wCombinedWeeds.getCombinedMask(self.mBatchDim)) #for debugging

            self.mMapListBatch.append(wCombinedWeeds.getStackedMaps())

        
        self.mMapListBatch = np.array(self.mMapListBatch, dtype= object).T.tolist() 
        if self.mTrain:
            self.mWeedData.refreshAllRandomSamplers(wNameRand, wImRand, wMaskRand, wCenterRand, wMapRandList, wGridCenterRandList, wGrassRand)
        self.incrementCounter()
        return self.mNameBatch, self.mImBatch, self.mMaskBatch, self.mMapListBatch
    
    def __iter__(self):
        return self
    
    def getCounter(self):
        return self.mCounter
    
    def getBatchSize(self):
        return self.mBatchSize
    
    def setNoRepeat(self, iBool = True):
        self.mTrain = iBool     
    
    def setSize(self, iSize):
        self.mSize = iSize
    
    def getSize(self):
        return self.mSize
    
    def incrementCounter(self):
        if self.mTrain:
            self.mCounter +=1
        elif self.mCounter >= int(self.mSize/self.getBatchSize())-1:
            self.mCounter = 0
            self.reseedAllRandomSamplers(self.mCounter)
        else:
            self.mCounter +=1
    
class WeedDataLoader:
    def __init__(self, weed_list, dim):
        self.myLoader = WeedLoader(weed_list)
        self.myLoader.loadImages(dim)    
        self.im_list = self.myLoader.getImages()
        self.myLoader.loadNames()
        self.name_list = self.myLoader.getNames() 
        self.myLoader.loadMasks()
        self.mask_list = self.myLoader.getMasks()
        self.myLoader.loadCenters()
        self.center_list = self.myLoader.getCenters()
        
    def setDimGridList(self, iDimGridList):
        self.multidim_map_lists = []
        self.multidim_grid_center_lists = []
        self.mDimGridList = iDimGridList
        
    def addMultiDimMapList(self, dim_grid):
        #tuple of (W,H)
        self.myLoader.loadMaps(dim_grid)     
        self.multidim_map_lists.append(self.myLoader.getHmaps()) 
    
    def addMultiDimGridCenterList(self):
        self.multidim_grid_center_lists.append(self.myLoader.getGridCenters())
        
    def addMultiDimLists(self, dim_grid):
        #tuple of (W,H)
        self.addMultiDimMapList(dim_grid)
        self.addMultiDimGridCenterList()
    
    def addAllGridDims(self):
        #list of dim_grid tuples
        for dim_grid in self.mDimGridList:
            self.addMultiDimLists(dim_grid)
    
    def getImList(self):
        return self.im_list
    
    def getMaskList(self):
        return self.mask_list
    
    def getNameList(self):
        return self.name_list
    
    def getMapList(self):
        return self.map_list
    
    def getCenterList(self):
        return self.center_list
    
    def getGridCenterList(self):
        return self.grid_center_list
    
    def setGrassList(self, grass_list):
        self.grass_list = grass_list
    
    def getGrassList(self):
        return self.grass_list
    
    def getMultiDimMapLists(self):
        return self.multidim_map_lists
    
    def getMultiDimGridCenterLists(self):
        return self.multidim_grid_center_lists
    
    def setImageRandomSampler(self, iSampleSize, iSeed):
        self.ImRand = RandSampler(self.getImList(), iSampleSize, iSeed)

    def setMaskRandomSampler(self, iSampleSize, iSeed):
        self.MaskRand = RandSampler(self.getMaskList(), iSampleSize, iSeed)
        
    def setNameRandomSampler(self, iSampleSize, iSeed):
        self.NameRand = RandSampler(self.getNameList(), iSampleSize, iSeed)
    
    def setCenterRandomSampler(self, iSampleSize, iSeed):
        self.CenterRand = RandSampler(self.getCenterList(), iSampleSize, iSeed)
        
    def setMapRandomSamplers(self, iSampleSize, iSeed):
        self.multiDimMapRand = []
        for map_list in self.getMultiDimMapLists():
            self.multiDimMapRand.append(RandSampler(map_list, iSampleSize, iSeed))
    
    def setGridCenterRandomSamplers(self, iSampleSize, iSeed):
        self.multiDimGridCenterRand = []
        for grid_center_list in self.getMultiDimGridCenterLists():
            self.multiDimGridCenterRand.append(RandSampler(grid_center_list, iSampleSize, iSeed))
    
    def setGrassRandomSampler(self, iSampleSize, iSeed):
        self.GrassRand = RandSampler(self.getGrassList(), iSampleSize, iSeed)    
    
    def setAllRandomSamplers(self, iSampleSize, iSeed):
        self.setImageRandomSampler(iSampleSize, iSeed)      
        self.setNameRandomSampler(iSampleSize, iSeed)
        self.setMaskRandomSampler(iSampleSize, iSeed)
        self.setCenterRandomSampler(iSampleSize, iSeed)        
        self.setMapRandomSamplers(iSampleSize, iSeed)
        self.setGridCenterRandomSamplers(iSampleSize, iSeed)
        self.setGrassRandomSampler(1, iSeed)
    
    def getGrassRandomSampler(self):
        return self.GrassRand
    
    def getImageRandomSampler(self):
        return self.ImRand
        
    def getMaskRandomSampler(self):
        return self.MaskRand
        
    def getNameRandomSampler(self):
        return self.NameRand
    
    def getCenterRandomSampler(self):
        return self.CenterRand    
    
    def getMapRandomSamplers(self):
        return self.multiDimMapRand
    
    def getGridCenterRandomSamplers(self):
        return self.multiDimGridCenterRand
    
    def getAllRandomSamplers(self):
        return self.getNameRandomSampler(),\
            self.getImageRandomSampler(),\
                self.getMaskRandomSampler(),\
                    self.getCenterRandomSampler(),\
                        self.getMapRandomSamplers(),\
                            self.getGridCenterRandomSamplers(),\
                                self.getGrassRandomSampler()
        

    def refreshAllRandomSamplers(self, iNameRand, iImRand, iMaskRand,iCenterRand, iMultiDimMapRand, iMultiDimGridCenterRand, iGrassRand):
        self.NameRand = iNameRand
        self.ImRand = iImRand
        self.MaskRand = iMaskRand
        self.CenterRand = iCenterRand
        self.multiDimMapRand = iMultiDimMapRand
        self.multiDimGridCenterRand = iMultiDimGridCenterRand
        self.GrassRand = iGrassRand

class WeedLoader():
    def __init__(self, weed_sub_list):
        self.weed_list = weed_sub_list #list of Original_weed()
        self.no_weeds = len(self.weed_list)
        
    def loadNames(self):
        self.name_list = []
        for weed in self.weed_list:
            self.name_list.append(weed.getName())
    
    def getNames(self):
        return self.name_list
    
    def loadImages(self, dim, flag = cv.IMREAD_COLOR, inter = cv.INTER_AREA):
        self.im_list =[]
        self.dim = dim
        self.W, self.H = self.dim
        for weed in self.weed_list:
            self.im_list.append(weed.load_image(self.dim, flag, inter))
    
    def loadMasks(self, dim = None, flag = cv.IMREAD_GRAYSCALE, inter = cv.INTER_AREA):
        if dim is None:
            dim = self.dim
        self.mask_list =[]
        for weed in self.weed_list:
            self.mask_list.append(weed.load_mask(dim, flag, inter))
    
    def loadCenters(self):
        self.center_list = []
        for weed in self.weed_list:
            self.center_list.append(weed.region_list[0].get_center())
    
    def getCenters(self):
        return self.center_list
    
    def getCenterAtIndex(self, iIndex):
        return self.getCenters()[iIndex]
    
    def loadMaps(self,dim_grid):
        self.map_list =[]
        self.center_grid_list = []
        self.dim_grid = dim_grid
        self.W_grid, self.H_grid = self.dim_grid
        for weed, weedCenter in zip(self.weed_list, self.getCenters()):
            weed.set_comp_map(self.dim, self.dim_grid)
            self.map_list.append(weed.get_comp_map())
            # weedCenter = weed.region_list[0].get_center()
            weedCenter_grid = weedCenter[0]*self.W_grid//self.W, weedCenter[1]*self.H_grid//self.H
            self.center_grid_list.append(weedCenter_grid)
    
    def getImages(self):
        return self.im_list
    
    def getMasks(self):
        return self.mask_list
    
    def getHmaps(self):
        return self.map_list
    
    def getGridCenters(self):
        return self.center_grid_list
 
    
def GenerateDataList(data_dir, file_points_path, scale_x = 1., scale_y = 1.):
    #the file_points uses the orignal image dimensions whereas the image folder
    #can use downsampled versions for which the scale_x,y must be adjusted
    with open(os.path.join(data_dir, file_points_path), 'r') as f:
        via = json.load(f)    
    
    oimg_list = []    
    for fid in via['_via_img_metadata']:
        
        file_name = via['_via_img_metadata'][fid]['filename']
        file_path = os.path.join(data_dir, file_name)

        if not os.path.isfile(file_path):
            print('File not found! %s' %(file_path))
            continue

        oim = Original_image(file_path)
        for region in via['_via_img_metadata'][fid]['regions']:
            oim.set_region_from_attr(region['shape_attributes'], scale_x, scale_y)

        oimg_list.append(oim)

    return oimg_list


def GenerateWeedDataList(data_dir, im_dir, file_points_path, scale_x = 1., scale_y = 1.):
    #the file_points uses the orignal image dimensions whereas the image folder
    #can use downsampled versions for which the scale_x,y must be adjusted
    with open(os.path.join(data_dir, file_points_path), 'r') as f:
        via = json.load(f)    
    
    oimg_list = []    
    for fid in via['_via_img_metadata']:
        
        file_name = via['_via_img_metadata'][fid]['filename']
        file_path = os.path.join(data_dir, file_name)

        if not os.path.isfile(file_path):
            print('File not found! %s' %(file_path))
            continue

        oim = Original_weed(file_path, im_dir)
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

def flatten_map(im_map, invert = 1):
    #print(np.shape(im_map))
    max_map = np.max(im_map, axis = (0,1), keepdims=True) #get maximum of each channel
    min_map = np.min(im_map, axis = (0,1), keepdims=True)
    im_act = im_map #for activations later
    im_map = (im_map-min_map)/(max_map - min_map + 1e-15)
    im_map = np.maximum.reduce(im_map, axis=2)
    if invert:
        im_map = 1 - im_map

    im_act = np.where(im_act < max_map, 0.,1.) #set max element of each channel to 1 and all else to zero.
    im_act = np.maximum.reduce(im_act, axis=2)
    im_act= np.float32(im_act) 
    
    return im_map, im_act

def flat_map_list(map_list):
    flat_list = []
    for map1 in map_list:
        flat, _ = flatten_map(map1)
        flat_list.append(flat)
  
    return flat_list 

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

def process_batch(batch, dim, dim_grid, seq = None, lo_val = 0.1):
    map_list = map_list_from_batch(batch, dim, dim_grid)
    if seq is not None:
        _, map_aug_list = seq(images = map_list, heatmaps = map_list)
    else:
        map_aug_list = map_list

    flat_list1 = flat_map_list(map_aug_list)
    flat_arr1 = np.array(flat_list1)

    weight_arr1=  (1.-lo_val)*(1-flat_arr1) + lo_val #scale so that black cell weights of inverse map are not zero

    return flat_arr1, weight_arr1, seq

def ProcessMapList3D(iMapList, dim, dim_grid, seq = None, lo_val = 0.0):
    wMapList = iMapList
    if seq is not None:
        heatmap_aug_list = seq(heatmaps = map2heatmap_list(wMapList, dim))
        map_aug_list = heatmap2map_list(heatmap_aug_list)
    else:
        map_aug_list = wMapList
    
    map_aug_list = scale_map_list_3D(map_aug_list)
    weight_list = weight_list_3D(map_aug_list, lo_val)
        
    return map_aug_list, weight_list, seq 

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

def show_batch(X_batch, batch_names = None, size = 40., iEndIdx=4):
    #print('ShowingBatch')    
    X_batch= X_batch[:iEndIdx]
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
    
    
def get_batch_plots(X_batch, batch_names = None, size = 40., iEndIdx=4):
    #print('ShowingBatch')    
    X_batch = X_batch[:iEndIdx]
    _, axs = plt.subplots(1, len(X_batch))#, figsize=(40, 40))
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

    plt.tight_layout()
    return plt
    
class generate_batch():
    
    def __init__(self, XY_train, batch_size, seed = 4):
        #print('constructing class')
        self.XY_train = deepcopy(XY_train)
        if seed:
            random.Random(seed).shuffle(self.XY_train)
        self.batch_size = batch_size
        self.data_len = len(self.XY_train)
        self.steps = int(np.ceil(self.data_len/batch_size))
        self.extra_step_size = self.data_len%batch_size
        self.index = 0
        self.data_len = len(self.XY_train)
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
    
    def getBatchSize(self):
        return self.batch_size
    
    def getDataLen(self):
        return self.data_len
    


import cv2 as cv
import numpy as np
import os
from loss_functions import cells_object, flatten_map_v2
from augment_utils import inverse_transform_matrix, getAngleBounds, warpImage, getTransFromPolar
from colortrans import transfer_lhm
#from helper import show_wait

def putTitleOnImage(ioIm, iTitle, iHeaderSize= 3.5, iFont=cv.FONT_HERSHEY_SIMPLEX, iFontScale=3.5, iTextColor=(0, 0, 0), iThick=10):
    H,W = ioIm.shape[:2]
    
    wTextSize, _ = cv.getTextSize(iTitle, iFont, iHeaderSize, iThick)
    _, wTextH = wTextSize
    
    wTextSize, _ = cv.getTextSize(iTitle, iFont, iFontScale, iThick)
    wTextW, _ = wTextSize
    
    wPos = (int(W/2 -wTextW/2), int((wTextH + iFontScale)*1.1))
    # (wX, wY+ wTextH + int(iFontScale) - 1)
    wHeadShape = list(ioIm.shape)
    wHeadShape[0] = int((wTextH + iHeaderSize)*1.5)
    # print(wHeadShape)
    wHeader = np.ones(tuple(wHeadShape), dtype=np.uint8)*255
    ioIm = np.vstack([wHeader, ioIm])
    ioIm = cv.putText(ioIm, iTitle , wPos, iFont , iFontScale, iTextColor, iThick, lineType=cv.LINE_AA)
    return ioIm
    
class Original_image():
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(self.path) #works for windows, on linux returns full path
        self.region_list = []
        self.type_list = []
        self.weight = 0
        self.comp_map = None
        self.comp_pn = None
        
    def __repr__(self):
        return "Original Image"
        
    def set_region_from_attr(self, region_attr, scale_x = 1., scale_y = 1.):
        rtype = region_attr['name']

        if rtype == 'point':
            region = Point.from_attr(region_attr, scale_x, scale_y) 
        elif rtype == 'rect':
            pass
        elif rtype == 'circle':
            region = Circle.from_attr(region_attr, scale_x, scale_y) 
        elif rtype == 'ellipse':
            region = Ellipse.from_attr(region_attr, scale_x, scale_y)
        
        self.region_list.append(region)
        self.type_list.append(str(region))
        self.weight+=1
        
    def set_comp_map(self, dim, dim_grid):
        map_list = []
        map_3d_list = []
        pn_list = []
        pn_3d_list =[]
        for region in self.region_list:
            region.set_map(dim, dim_grid)
            grad_map = region.get_map()
            map_list.append(grad_map)
            map_3d_list.append(grad_map[:,:,None])
            pn = region.get_pn()
            pn_list.append(pn)
            pn_3d_list.append(pn[:,:,None])
            
            
        self.comp_map = np.maximum.reduce(map_list)
        self.comp_map_3d = np.concatenate(map_3d_list, 2)
        self.comp_pn = np.maximum.reduce(pn_list)
        self.comp_pn_3d = np.concatenate(pn_3d_list, 2)
        
    
    def get_comp_map(self):
        return np.float32(self.comp_map)
    def get_comp_map_3d(self):
        return np.float32(self.comp_map_3d)
    
    def get_comp_pn(self):
        return np.float32(self.comp_pn)
    
    def get_comp_pn_3d(self):
        return np.float32(self.comp_pn_3d)
    
    def get_comp_map_resize(self, dim, inter = cv.INTER_NEAREST):
        
        return cv.resize(self.get_comp_map(), dim, interpolation = inter)
        
    def load_image(self, dim = None, flag = cv.IMREAD_COLOR, inter = cv.INTER_AREA):

        im = cv.imread(self.path, flag)
        
        if dim is not None:
            im = cv.resize(im, dim, interpolation = inter)  

        return im 
    def get_path(self):
        return self.path
    
    def get_regions(self):
        return self.region_list
    def get_center_list(self):
        return [wRegion.get_center() for wRegion in self.get_regions()]
    def get_center_dict(self):
        i=0
        oDict= {}
        for wCenter in self.get_center_list():
            oDict.update({'t'+str(i):wCenter})
            i+=1
            
        return oDict
    def get_name(self):
        return self.name

class Original_weed(Original_image):
    def __init__(self, mask_path, im_dir):
        super().__init__(mask_path)
        self.im_name = self.name[2:]
        self.im_path = os.path.join(im_dir, self.im_name)

    def load_mask(self, dim = None, flag = cv.IMREAD_GRAYSCALE, inter = cv.INTER_AREA):
        return super().load_image(dim, flag, inter)
    
    def load_image(self, dim = None, flag = cv.IMREAD_COLOR, inter = cv.INTER_AREA):
        im = cv.imread(self.im_path, flag) 
        if dim is not None:
            im = cv.resize(im, dim, interpolation = inter)  
        return im
    
    def getName(self):
        return self.im_name
    
class CombinedWeed():
    
    def __init__(self, name_list, im_list, mask_list, center_list, map_list_list, grid_center_list_list, grass):
        self.name_list = name_list
        self.im_list = im_list
        self.mask_list = mask_list
        self.map_list_list = map_list_list #list of lists of maps
        self.center_list = center_list
        self.grid_center_list_list = grid_center_list_list #list of list of maps
        self.shape = im_list[0].shape
        self.dim = self.shape[1], self.shape[0]
        self.W, self.H = self.dim
        self.dim_grid_list = []
        for map_list in map_list_list:
            wShape = map_list[0].shape
            wDim = wShape[1], wShape[0]
            self.dim_grid_list.append(wDim)
        self.grass = grass
        self.no_weeds = len(im_list)
        self.random = np.random

    def setSeed(self, iSeed = None):
        self.seed = iSeed
        self.random.seed(self.seed)        
    
    def getNames(self):
        return self.name_list
        
    def getImages(self):
        return self.im_list
    
    def getMasks(self):
        return self.mask_list
    
    def getHmaps(self):
        return self.map_list_list
    
    def getGrass(self):
        return self.grass
    
    def transferColors(self, flag = True):
        if flag:
            srcIm = self.grass
            tarImList = self.im_list
            self.colouredWeeds = []
            for tarIm in tarImList:
                lhm = transfer_lhm(tarIm[...,::-1].astype(np.float32),srcIm[...,::-1].astype(np.float32))[...,::-1].astype(np.uint8)
                self.colouredWeeds.append(lhm)
        else:
            self.colouredWeeds = self.getImages()
    
    def getAngleBounds(self):
        return getAngleBounds(self.no_weeds)
    
    def setRotFlipRanges(self, seed = 0):
        self.AngleList = []
        self.RadiusList = []
        self.FlipList = []
        for i in range(self.no_weeds):
            self.AngleList.append(self.random.randint(0,90))
            self.FlipList.append(((-1)**self.random.randint(1,3),(-1)**self.random.randint(1,3)))

    def rotateFlipWeeds(self, seed = 0):
        self.rotFlipWeeds = dict()
        wImList =[]
        wMaskList = []
        wHmapListList = []
        self.countRotList = []
        for i in range(self.no_weeds):
            angle = self.AngleList[i]
            flip = self.FlipList[i]
            im = self.colouredWeeds[i]
            mask = self.mask_list[i]
            weedCenter = self.center_list[i]
            mask6 = warpImage(mask, weedCenter, self.dim, angle, flip)
            im6 = warpImage(im, weedCenter, self.dim, angle, flip)
            wImList.append(im6)
            wMaskList.append(mask6)    
            wHmapList = []
            for j in range(len(self.map_list_list)):
                hmap = self.map_list_list[j][i]
                weedCenter_grid = self.grid_center_list_list[j][i]
                dim_grid = self.dim_grid_list[j]
                hmap6 = warpImage(hmap, weedCenter_grid, dim_grid, angle, flip)
                wHmapList.append(hmap6) 
            wHmapListList.append(wHmapList)
        wHmapListListTransposed= np.array(wHmapListList, dtype = object).T.tolist()
    
        self.rotFlipWeeds.update({'images': wImList, 'masks':wMaskList, 'heatmaps': wHmapListListTransposed})
    
    def getRotFlipWeeds(self):
        return self.rotFlipWeeds['images']
    
    def getRotFlipMasks(self):
        return self.rotFlipWeeds['masks']

    def getRotFlipHmaps(self):
        return self.rotFlipWeeds['heatmaps']  
    
    def getRotFlipWeedAtIndex(self, iIndex):
        return self.getRotFlipWeeds()[iIndex]
    
    def getRotFlipMaskAtIndex(self, iIndex):
        return self.getRotFlipMasks()[iIndex]

    def getRotFlipHmapAtIndex(self, iDimIndex, iIndex):
        return self.getRotFlipHmaps()[iDimIndex][iIndex] #change
    
    def setPlacementMinRatio(self, iPlacementMinRatio):
        self.mPlacementMinRatio = iPlacementMinRatio
    
    def setMinMaxTransPercent(self, iMinPercent, iMaxPercent):
        self.mMinPercent = iMinPercent
        self.mMaxPercent = iMaxPercent
    
    def setBuffer(self, iBuffer = 0):
        self.mBuffer = iBuffer
    
    def randomTrans(self):
        dxMin, dxMax = self.mMinPercent*self.W, self.mMaxPercent*self.W
        dyMin, dyMax = self.mMinPercent*self.H, self.mMaxPercent*self.H
        dx = self.random.randint(dxMin, dxMax)
        dy = self.random.randint(dyMin, dyMax)
        #print("r:", "N/A","angle:", "N/A", "dx:", np,round(dx), "dy:", np.round(dy))
        return dx, dy
        
    def setAngleBounds(self):
        self.angleBoundList = self.getAngleBounds()
    
    def randomTransFromPolar(self, i):
        rMin, rMax = self.mMinPercent*self.H, self.mMaxPercent*self.H
        r = self.random.randint(rMin, rMax)
        angle = self.random.randint(self.angleBoundList[i]+self.mBuffer, self.angleBoundList[i+1]-self.mBuffer)
        dx, dy = getTransFromPolar(r, angle)
        #print("r:", np.round(r),"angle:", np.round(angle), "dx:", np.round(dx), "dy:", np.round(dy))
        return np.int32(dx), np.int32(dy)
    
    def randomTransMin(self, iAddLength = .1):
        drMax = (1+iAddLength)*self.mMinPercent*self.H
        angle = self.random.randint(0, 90)
        angleRad = angle*np.pi/180
        dxMax = int(drMax*np.cos(angleRad))
        dyMax = int(drMax*np.sin(angleRad))
        # print("drMax: %s dxMax: %s dyMax: %s"%(drMax, dxMax, dyMax))
        
        if dxMax ==0:
            dx = 0
        else:
            dx = self.random.randint(-dxMax, dxMax)
        if dyMax == 0:
            dy = 0
        else:
            dy = self.random.randint(-dyMax, dyMax)
            
        return dx, dy
        
    
    def computeShiftedCount(self, mask, countRot):
        maskPrime = warpImage(mask, (self.W//2,self.H//2), self.dim, transXY = (0.5, 0.5))
        countRotPrime = np.count_nonzero(maskPrime)
        if countRotPrime>countRot:
            print("switching counts")
            countRot = countRotPrime
        return countRot
    
    def placeWeeds(self, flag = True, shift = True):
        self.translatedWeeds = dict()
        wImList = []
        wMaskList = []
        wHmapListList = []
        wSizePercentList = []
        # if shift:
        #     ddx, ddy = self.randomTransMin()
        for i in range(self.no_weeds):
            if shift:
                ddx, ddy = self.randomTransMin()            
            mask = self.getRotFlipMaskAtIndex(i)
            countRot = np.count_nonzero(mask)
            im1 = None
            for j in range(10): #15 tries to get a good random translation
                if flag:
                    self.setAngleBounds()
                    dx, dy = self.randomTransFromPolar(i)
                else:
                    dx,dy = self.randomTrans()     
                if shift:
                    # ddx, ddy = self.randomTransMin()
                    dx, dy = dx+ddx, dy+ddy
                mask1 = warpImage(mask, (self.W//2,self.H//2), self.dim, transXY = (dx, dy))
                countTrans = np.count_nonzero(mask1)
                #print("weed#:", i+1 , "of", self.no_weeds, "ratio:", np.round(countTrans/countRot, 2))
                if countTrans/countRot < self.mPlacementMinRatio:
                    continue
                else:
                    im = self.getRotFlipWeedAtIndex(i)
                    sizePercent = np.round(countTrans/(self.W*self.H)*100)                    
                    im1 = warpImage(im, (self.W//2,self.H//2), self.dim, transXY = (dx, dy)) 
                    wHmapList=[]
                    for k in range(len(self.dim_grid_list)):
                        dim_grid = self.dim_grid_list[k]
                        W_grid, H_grid = dim_grid
                        hmap = self.getRotFlipHmapAtIndex(k,i)   
                        hmap1 = warpImage(hmap, (W_grid//2,H_grid//2), dim_grid, transXY = (dx*W_grid/self.W, dy*H_grid/self.H))
                        wHmapList.append(hmap1)
                    break
            if im1 is not None:    
                wImList.append(im1)
                wMaskList.append(mask1)
                wHmapListList.append(wHmapList)
                wSizePercentList.append(sizePercent)
        wHmapListListTranspose = np.array(wHmapListList, dtype = object).T.tolist()
        self.mNoCombinedWeeds =len(wImList)
        if self.mNoCombinedWeeds > 0:
            self.translatedWeeds.update({'images': wImList, 'masks': wMaskList, 'heatmaps': wHmapListListTranspose, 'sizes': wSizePercentList})
            self.mSuccess = True
        else:
            self.mSuccess = False
            print("Did not place any weeds for this combination!")
    
    def isSuccess(self):
        return self.mSuccess
    
    def shiftWeeds(self):
        wImList   = []# self.getTranslatedWeeds()
        wMaskList = []#self.getTranslatedMasks()
        wHmapListList =[]# self.getTranslatedMasks()
        
        for j in range(10):
            dx, dy = self.randomTransMin()
            wSuccessFlag = False
            for i in range(self.mNoCombinedWeeds):
                mask = self.getTranslatedMaskAtIndex(i)
                count = np.count_nonzero(mask)
                mask1 = warpImage(mask, (self.W//2,self.H//2), self.dim, transXY = (dx, dy))
                countTrans = np.count_nonzero(mask1)
                if countTrans/count < self.mPlacementMinRatio:
                    wSuccessFlag = False
                    break
                else:
                    wSuccessFlag = True        
            if wSuccessFlag:
                break
        
        if wSuccessFlag:
            for i in range(self.mNoCombinedWeeds):
                mask = warpImage(self.getTranslatedMaskAtIndex(i), (self.W//2,self.H//2), self.dim, transXY = (dx, dy))
                im = warpImage(self.getTranslatedWeedAtIndex(i), (self.W//2,self.H//2), self.dim, transXY = (dx, dy)) 
                wHmapList=[]
                for k in range(len(self.dim_grid_list)):
                    dim_grid = self.dim_grid_list[k]
                    W_grid, H_grid = dim_grid
                    hmap = warpImage(self.getTranslatedHmapAtIndex(k,i), (W_grid//2,H_grid//2), dim_grid, transXY = (dx*W_grid/self.W, dy*H_grid/self.H))
                    wHmapList.append(hmap)
                    
                wImList.append(im)
                wMaskList.append(mask)
                wHmapListList.append(wHmapList)
                
            self.translatedWeeds['images'] = wImList
            self.translatedWeeds['masks'] = wMaskList
            wHmapListListTranspose = np.array(wHmapListList, dtype = object).T.tolist()
            self.translatedWeeds['heatmaps'] = wHmapListListTranspose
            print('weeds shifted dx: %s dy: %s'%(dx,dy))
        
        self.mSuccess = wSuccessFlag

    def getTranslatedWeeds(self):
        return self.translatedWeeds['images']
    
    def getTranslatedMasks(self):
        return self.translatedWeeds['masks']

    def getTranslatedHmaps(self):
        return self.translatedWeeds['heatmaps']  
    
    def getTranslatedSizes(self):
        return self.translatedWeeds['sizes']
    
    def getTranslatedWeedAtIndex(self, iIndex):
        return self.getTranslatedWeeds()[iIndex]
    
    def getTranslatedMaskAtIndex(self, iIndex):
        return self.getTranslatedMasks()[iIndex]

    def getTranslatedHmapAtIndex(self, iDimIndex, iIndex):
        return self.getTranslatedHmaps()[iDimIndex][iIndex]
    
    def getTranslatedSizeAtIndex(self, iIndex):
        return self.getTranslatedSizes()[iIndex]

    def combine3d(self):
        wHmapList = self.getTranslatedHmaps()
        self.comp_map_3d = np.stack(wHmapList, -1)

    def setPosedMasks(self):
        sortedIndexes = list(np.argsort(self.getTranslatedSizes()))
        if self.mNoCombinedWeeds >1:
            wMaskList = [None]*self.mNoCombinedWeeds
            #adjust masks for superposing images
            wIndexes = list(range(self.mNoCombinedWeeds))
            for i in wIndexes:
                maskI = self.getTranslatedMaskAtIndex(sortedIndexes[i]).astype(np.int32)
                for j in wIndexes[i+1:]:
                    maskJ = self.getTranslatedMaskAtIndex(sortedIndexes[j]).astype(np.int32)
                    maskI = np.clip((maskI - maskJ), 0 , 255)
                wMaskList[sortedIndexes[i]] = maskI.astype(np.uint8)
        else:
            wMaskList = [self.getTranslatedMaskAtIndex(0)]
                
        self.posedMasksList = wMaskList     
    
    def getPosedMasks(self):
        return self.posedMasksList
    
    def getPosedMaskAtIndex(self, iIndex):
        return self.getPosedMasks()[iIndex]
    
    def getStackedPosedMasks(self):
        return np.hstack(self.getPosedMasks())
    
    def getStackedWeeds(self):
        return np.hstack(self.getTranslatedWeeds())
       
    def flattenWeeds(self):
        wImList = []
        wMaskList = []
        if self.mNoCombinedWeeds>1:
            for wIndex in range(self.mNoCombinedWeeds):
                im = self.getTranslatedWeedAtIndex(wIndex).astype(np.uint32)
                mask = self.getPosedMaskAtIndex(wIndex)
                im = (im*(mask[...,None]/255.)).astype(np.uint8)
                wImList.append(im)
                wMaskList.append(mask)
            flatIm = sum(wImList).astype(np.uint8)  
            flatMask = sum(wMaskList).astype(np.uint8)

        else:
            im = self.getTranslatedWeedAtIndex(0)
            mask = self.getTranslatedMaskAtIndex(0)
            flatIm = (im*(mask[...,None]/255.)).astype(np.uint8)
            flatMask = mask
   
        self.combinedWeed = flatIm
        self.combinedMask = flatMask

    def setBlend(self, iBlendFlag=True):
        weed = self.getStackedWeeds()
        mask = self.getStackedPosedMasks()
        grass = self.getGrass()        
        
        A = weed.copy()/255.0
        B = grass.copy()/255.0
        C = mask.copy()/255.0

        if iBlendFlag:
            mask_list = [mask.copy()/255.0]
            for i in range(5):
                rows,cols = mask_list[i].shape
                den = 2
                mask_list.append(cv.resize(mask.copy()/255.0, (cols//den, rows//den), interpolation = cv.INTER_AREA))
            mask_list.reverse()

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
                WB = lb.shape[1]
                gc = gc[...,None]
                #ls = la*(gc) + (lb*(1 - gc))
                ls = la*(gc)
                lsPrime= 0.
                gcPrime = 0.
                for i in range(self.mNoCombinedWeeds):
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
            self.finalImage = np.clip(cv.addWeighted(ls_, 1.5, gaussian_3, -0.5, 0), 0., 1., dtype = np.float32)
        else:
            H,W,_ = B.shape
            wWeed = 0.
            wMask = 0.
            for i in range(self.mNoCombinedWeeds):
                wWeed += A[:, i*W:(i+1)*W]*C[:, i*W:(i+1)*W, None]
                wMask += C[:, i*W:(i+1)*W, None]

            self.finalImage = np.clip(wWeed+ B*(1-wMask), 0., 1., dtype = np.float32)
            
    def getBlend(self, iDim = None, inter = cv.INTER_AREA):
        if iDim is None:
            im = self.finalImage
        else:
            im = cv.resize(self.finalImage, iDim, interpolation = inter)
        return im

    def getCombinedWeed(self):
        return self.combinedWeed

    def getCombinedMask(self, iDim = None, inter = cv.INTER_AREA):
        if iDim is None:
            mask = self.combinedMask
        else:
            mask = cv.resize(self.combinedMask, iDim, interpolation = inter)
        return mask

    def setStackedMaps(self):
        self.stackedMaps = []
        #print("len(self.dim_grid_list)", len(self.dim_grid_list))
        for i in range(len(self.dim_grid_list)):
            wMapList = self.getTranslatedHmaps()[i]
            self.stackedMaps.append(np.stack(wMapList,-1))
        
    def setFlatMaps(self):
        self.flatMaps = [] 
        for i in range(len(self.dim_grid_list)):
            self.flatMaps.append(flatten_map_v2(self.stackedMaps[i]))
    
    def getStackedMaps(self):
        return self.stackedMaps
        
    def getFlatMaps(self):
        return self.flatMaps
    
    def getFlatMapAtDimIndex(self, iDimIndex):
        return self.getFlatMaps()[iDimIndex]
    
    def getFlatMapResized(self, iDimIndex, dim = None, inter = cv.INTER_NEAREST):
        if dim is None:
            dim = self.dim
        return cv.resize(self.getFlatMapAtDimIndex(iDimIndex), dim, interpolation = inter)
    
    def getFlatMapColor(self, iDimIndex, dim = None, code = cv.COLOR_GRAY2BGR):
        if dim is not None:
            im = self.getFlatMapResized(iDimIndex, dim)
        else:
            im = self.getFlatMapAtDimIndex(iDimIndex)
        return cv.cvtColor(im, code)
        
    def getFlatMapColorPreset(self, iDimIndex):
        return self.getFlatMapColor(iDimIndex, self.dim) 

    def getAllSideBySide(self):
        wIm = self.getBlend()
        wMask = cv.cvtColor(self.getCombinedMask(), cv.COLOR_GRAY2BGR)
        wLabels = []
        for i in range(len(self.dim_grid_list)):
            wLabels.append(self.getFlatMapColorPreset(i))
        wImMask = [wIm, wMask]
        wImMask.extend(wLabels)
        return np.hstack(wImMask)
         
class Region_obj():
    def __init__(self, region_name):
        self.rtype = region_name
    
    @classmethod    
    def from_attr(cls, region_attr):
        rtype = region_attr['name']
        return cls(rtype)

    def __repr__(self):
        return self.rtype
    
    
class Point(Region_obj):
    def __init__(self, rtype, x, y):
        super().__init__(rtype)
        self.x, self.y = x, y
    
    @classmethod
    def from_attr(cls, region_attr):
        rtype = region_attr['name']
        x, y = region_attr['cx'], region_attr['cy']
        return cls(rtype, x, y)
    
    def reset_center(self, x,y):
        self.x, self.y = x, y
    
    def get_center(self):
        return self.x, self.y

class Ellipse(Point):
    def __init__(self, rtype, x, y, rx, ry, theta):
        super().__init__(rtype, x, y)
        self.rx, self.ry = rx, ry
        self.theta = theta
        
        
    @classmethod
    def from_attr(cls, region_attr, scale_x = 1., scale_y = 1.):
        rtype = region_attr['name']
        x, y = region_attr['cx'], region_attr['cy']
        rx, ry = region_attr['rx'], region_attr['ry']
        theta =  region_attr['theta']
        
        x = scale_x*x
        y = scale_y*y
        
        #here we scale rx and ry proportionately to how far they x components and y components
        #we have to divide by the unscaled sum components to normalize
        rx = (np.abs(np.cos(theta)*scale_x) + np.abs(np.sin(theta)*scale_y))/(np.abs(np.cos(theta)) + np.abs(np.sin(theta)))*rx
        ry = (np.abs(np.sin(theta)*scale_x) + np.abs(np.cos(theta)*scale_y))/(np.abs(np.cos(theta)) + np.abs(np.sin(theta)))*ry
        
        #here we find the new angle by transforming a unit circle coordinate and computing the new angle
        tscale = np.array([[scale_x,0.], [0., scale_y]])
        angle_components = np.array([[np.cos(theta)], [np.sin(theta)]])
        scaled_components = tscale@angle_components
        theta = np.arctan2(scaled_components[1,0], scaled_components[0,0] )
        return cls(rtype, x, y, rx, ry, theta)
    
    def set_map(self, dim, dim_grid):
        x,y = self.x, self.y
        rx, ry = self.rx, self.ry
        theta =  self.theta    

        rx_nrm, ry_nrm = rx/dim[0], ry/dim[1]
        r_min = np.min([rx_nrm, ry_nrm])
        r_base= 0.04 

        if r_min < r_base:
            factor = r_base/r_min
            rx_nrm *=factor
            ry_nrm *=factor
        #Define origin center (center grid)    
        cobj1 = cells_object.map_2_grid_from_xy(dim, dim_grid, (dim[0]/2,dim[1]/2))
        _, r_map1 = cobj1.load_radial_map()
        center = cobj1.grid_centers_ji[0]
        o_x, o_y = center[1], center[0]
        #Define region center
        cobj2 = cells_object.map_2_grid_from_xy(dim, dim_grid, (x,y))
        _, r_map_alt = cobj2.load_radial_map()
        center2 = cobj2.grid_centers_ji[0]
        x_grid, y_grid = center2[1], center2[0]

        #Perform image transformations to generate final transformation matrix
        t12 = np.array([[1.,0.,-o_x], [0., 1., -o_y], [0., 0., 1.]]) #if we only translate origin without reflecting y-axis
        scale_matrix = np.array([[2*rx_nrm , 0.0, 0.], [0.0, 2*ry_nrm, 0.],[0., 0., 1.]])
        
        #flip = np.array([[1.,0.,0], [0., 1., 0], [0., 0., 1.]])
        rot_mat = np.array([[np.cos(theta),-np.sin(theta),0], [np.sin(theta), np.cos(theta), 0], [0., 0., 1.]])
        tx, ty = x_grid-o_x, y_grid-o_y
        trans_mat = np.array([[1.,0.,tx], [0., 1., ty], [0., 0., 1.]])
        t21 = inverse_transform_matrix(t12)
        #Neglect bottom row for openCV function
        M = (t21@trans_mat@rot_mat@scale_matrix@t12)[:2,:]
        #Apply transformations
        r_map1_warped = cv.warpAffine(r_map1, M, dim_grid)
        
        #scale = 30
        #show_wait(r_map1_warped,scale, interpolation = cv.INTER_NEAREST)
        self.x_grid, self.y_grid = x_grid, y_grid
        posneg = np.zeros(dim_grid)
        posneg[y_grid,  x_grid] = 1.
        self.pn = posneg
        self.map = r_map1_warped
        #self.map = r_map_alt
        
    def get_map(self):
        return self.map
    
    def get_pn(self):
        return self.pn


class Circle(Ellipse):
    def __init__(self, rtype, x, y, rx, ry):
        super().__init__(rtype, x, y, rx, ry, 0)
    @classmethod
    def from_attr(cls, region_attr, scale_x = 1., scale_y = 1.): 
        rtype = region_attr['name']
        x, y = region_attr['cx'], region_attr['cy']
        rx, ry = region_attr['r'], region_attr['r']
        return cls(rtype, scale_x*x, scale_y*y, scale_x*rx, scale_y*ry)
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:05:22 2024

@author: i_bab
"""

    wLoadFolder4 = 'project2024'
    wLoadSubFolder4 = os.path.join('resnet_test_13_real_8bit_res2_ep_(0, 1000)_lr_1e-04', 'test_output_truth_acts')
    wLoadDir4= os.path.join(ROOT_DIR, wLoadFolder4, wLoadSubFolder4)
    wBnMetrics4 = BinaryMetricsManager(wLoadDir4)
    wBnMetrics4.loadData()
   
    H4, W4 = wBnMetrics4.getShape()
    wDataDir4 = 'data4k'
    wFolder4 = '1'
    wDataPath4 = os.path.join(ROOT_DIR, wDataDir4, wFolder4)

    
    wPointsFolder = '1'
    wPointsFile = "plant_centers_sifted_FINAL.json"
    wPointsPath = os.path.join(ROOT_DIR, wDataDir4, wPointsFolder, wPointsFile) 
    wPointsDir = os.path.join(ROOT_DIR, wDataDir4, wPointsFolder)
    Wo, Ho = 4056, 3040
    wShape4 = (H4,W4,3)
    wDim4 = (W4,H4)
    wScaleX4, wScaleY4 = W4/Wo, H4/Ho

    wBnMetrics4.setTruthDir(wDataPath4)    
    wBnMetrics4.setTruthLabelsDir(wPointsDir)
    wBnMetrics4.setTruthLabelsFile(wPointsFile)
    wBnMetrics4.setTruthScale(wScaleX4, wScaleY4)
    wBnMetrics4.loadTruthSet()
  
    wHypotenuse4 = np.sqrt(W4*W4 + H4*H4)
    wMaxDist4 = wHypotenuse4
    wMinDist4 = 0 #wHypotenuse4/8
    wStep4 = wHypotenuse4/32
    

    for wThresh in np.arange(wMinDist4, wMaxDist4, wStep4)[0:1]:
        wThresh = np.round(wThresh,3)
        wBnMetrics4.computeMetrics(wThresh, iMinPercentArea = 0.0)
        for i in range(wBnMetrics4.getDataSize()):
            show_wait(wBnMetrics4.getDataAtIdx(i).drawPredictions(wThresh),0.5)
            
    for i in range(wBnMetrics4.getDataSize()):
        print(wBnMetrics4.getDataAtIdx(i).getNumberOfClusters())

    
    wBnMetrics4.evaluateMetrics(iNorm=True)
    wBnMetrics4.printMetrics()
    wMetrics4, _ = wBnMetrics4.getMetrics()
    wMetricsPredicted4, wNoUnpredict4= wBnMetrics4.getMetricsPredicted()
    # print(wMetrics4)
    print(len(wMetrics4))
    wValues4 = [wMetrics4[wKey] for wKey in wMetrics4]


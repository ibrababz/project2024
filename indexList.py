# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:00:23 2024

@author: i_bab
"""

import numpy as np

a = list(range(100))
n = len(a)

dw0 = 3
w=0
j = 0
dw_list = []
for l in range(300):
    dw = int(np.ceil(dw0*np.exp(w)))
    print(dw)
    w+=0.05
    dw_list.append(int(np.ceil(dw)))
    if l%5 ==0:
        print(a[n-1-(j+1)*dw:n-1 -j*dw])
        j+=1

runSum = 0

for i, dw in zip(range(len(dw_list)), dw_list):
    runSum += dw
    if runSum >= 176:
        break





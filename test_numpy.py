# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 13:44:33 2017

@author: Morteza
"""

import numpy as np

a = np.random.random((3, 3))
b = np.ones((3, 3))
c = a*b
c[1,1]=4
print("a=",a ,"\nb=",b,c)



W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)
print("D===",D)

p=0.5
g = D < p
D*=g

print(D)
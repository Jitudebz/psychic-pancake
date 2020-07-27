# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:37:14 2020

@author: DEBASIS
"""

#from scipy.stats import beta


import matplotlib.pyplot as plt

import scipy.stats as st

import numpy as np

xx = np.linspace(0.01, 0.99, 100)

betapdf1 = st.beta.pdf(xx, 0.1, 0.1)
betapdf3 = st.beta.pdf(xx, 2, 2)

plt.plot(xx, betapdf1)

import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
xx = np.linspace(0.01, 0.99, 100)

#for (i in 1:)
a = int(input("Enter the value of a: "))
b = int(input("Enter the value of b: "))
a = a - 1
b = b - 1
betapdf2 = st.beta.pdf(xx, a, b)
plt.plot(xx, betapdf2)


plt.plot(xx, betapdf1)
plt.plot(xx, betapdf3)

# importing scipy 
from scipy.stats import beta
  
numargs = beta.numargs 
[a, b] = [0.6, ] * numargs 
rv = beta(a, b) 
  
print ("RV : \n", rv) 



import cv2 as ocv


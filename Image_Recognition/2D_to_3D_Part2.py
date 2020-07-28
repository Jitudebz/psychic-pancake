# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:39:26 2020

@author: DEBASIS
"""

%cd "C:\Users\DEBASIS\Documents\Machine Learning\Image Processing"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as ndimage
 
%cd "C:\Users\DEBASIS\Documents\Machine Learning\Image Processing"
imageFile = 'Melvin.jpg'
mat = imread(imageFile)

mat = mat[:,:,0] # get the first channel
rows, cols = mat.shape
xv, yv = np.meshgrid(range(cols), range(rows)[::-1])
 
blurred = ndimage.gaussian_filter(mat, sigma=(5, 5), order=0)
fig = plt.figure(figsize=(6,6))
 
ax = fig.add_subplot(221)
ax.imshow(mat, cmap='gray')
 
ax = fig.add_subplot(222, projection='3d')
ax.elev= 75
ax.plot_surface(xv, yv, mat)
 
ax = fig.add_subplot(223)
ax.imshow(blurred, cmap='gray')
 
ax = fig.add_subplot(224, projection='3d')
ax.elev= 75
ax.plot_surface(xv, yv, blurred)

plt.show()
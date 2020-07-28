# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:14:37 2020

@author: Debasis
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
xx = np.linspace(0, 1, 100)
yy = np.sin(2 * np.pi * xx)
p = np.random.normal(0, 0.1, 10)
x1 = np.linspace(0,1,10)
y1 = np.sin(2 * np.pi * x1) + p
plt.plot(xx, yy, color = 'orange')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Plot of a training data set of N =10 points')
plt.scatter(x1, y1)
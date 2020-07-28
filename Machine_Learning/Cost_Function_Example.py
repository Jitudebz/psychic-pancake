# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:32:14 2020

@author: Debasis
"""

import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

def cost_function(x,y):
    theta = [-0.5,0,0.5,1,1.5,2, 2.5]
    n = len(x)
    cost_fun = []
    
    
    for i in theta:
        y_predicted = i * x
        cost = (1/(2 * n)) * sum ([val**2 for val in (y-y_predicted)])
        print("Cost= {}, theta= {}".format(cost, i))
        
        cost_fun.append(cost)
    
    return cost_fun


x = np.array([1,2,3])
y = np.array([1,2,3])

cost_fun = cost_function(x,y)
theta = [-0.5,0,0.5,1,1.5,2, 2.5]

fig,a =  plt.subplots(1,2, figsize=(8,4))

a[0].plot(x,y)
a[0].scatter(x,y)
a[0].set_xlabel('X')
a[0].set_ylabel('Y')
a[0].set_title('$H(\Theta)$(x) for fixed value $\Theta1$')
a[1].plot(theta, cost_fun)
a[1].set_xlabel('$\Theta1$')
a[1].set_ylabel('$J(\Theta)$')
a[1].set_title('Cost function of parameter $\Theta1$')
plt.tight_layout()
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 01:35:51 2020

@author: Debasis
"""

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.001
    cost_fun = []
    
    
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/(n)) * sum ([val**2 for val in (y-y_predicted)])
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        
        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr,cost, i))
        
        cost_fun.append(cost)
        
    return cost_fun
        

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

cost_fun = gradient_descent(x,y)

fig,a =  plt.subplots(1,2, figsize=(8,4))

a[0].plot(x,y)
a[0].scatter(x,y)
a[0].set_xlabel('X')
a[0].set_ylabel('Y')
a[0].set_title('X vs Y')
a[1].plot(cost_fun)
a[1].set_xlabel('Iteration')
a[1].set_ylabel('$J(\Theta)$')
a[1].set_title('Cost function using Gradient Descent')
plt.tight_layout()

plt.show()


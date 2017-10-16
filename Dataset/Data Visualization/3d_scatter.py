# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 21:11:41 2017

@author: Everton
"""
import sys
sys.path.insert(0, '../Algoritmos')

import coral as coral
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()#figsize=(16,16))
ax = fig.add_subplot(111, projection='3d')

x_s =randrange(200,-0.7,0.7)
y_s =randrange(200,-0.7,0.7)
z_s =randrange(200,-5,5)

ax.scatter(x_s, y_s, z_s, c='r', marker='o')

x_t =randrange(100,-5,5)
y_t =randrange(100,-0.7,0.7)
z_t =randrange(100,-0.7,0.7)

df_s = pd.DataFrame({'f0':x_s, 'f1': y_s, 'f2': z_s})
df_t = pd.DataFrame({'f0':x_t, 'f1': y_t, 'f2': z_t})
#df_s.to_csv('sint_source.csv', index=False)

ax.scatter(x_t, y_t, z_t, c='b', marker='o')

#df_c = coral.correlation_alignment(df_s, df_t, class_column='')
#df_c.columns = ['f0','f1','f2']

#df_c.to_csv('sint_coral.csv', index=False)

#ax.scatter(df_c.f0, df_c.f1, df_c.f2, c='g', marker='^')

ax.set_xlabel('f0')
ax.set_xlim(xmin=-5,xmax=5)
ax.set_ylabel('f1')
ax.set_ylim(ymin=-5,ymax=5)
ax.set_zlabel('f2')

#plt.xlim(-5, 5)
#plt.ylim(-5, 5)


plt.show()




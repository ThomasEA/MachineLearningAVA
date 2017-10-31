# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 06:15:28 2017

@author: Everton

    Simulação para teste do algoritmo CORAL em variável
    sintética - 2D

"""

import sys
sys.path.insert(0, '../Algoritmos')

import pandas as pd
import numpy as np
import seaborn as sns
import coral as coral

from sklearn import preprocessing

import matplotlib.pyplot as plt

import seaborn
seaborn.set(style='ticks')

#Carrega dataset
df_s = pd.read_csv('../Data Visualization/Coral Simulado/source_coral_2.csv', sep=',')
df_t = pd.read_csv('../Data Visualization/Coral Simulado/test_coral_2.csv', sep=',')

#Normaliza
df_s_tmp = df_s.copy()
df_t_tmp = df_t.copy()

scaler = preprocessing.StandardScaler().fit(df_s_tmp)
df_s_std = pd.DataFrame(scaler.transform(df_s_tmp), columns = list(df_s_tmp))
    
df_s_std['evadido'] = df_s.evadido
    
scaler = preprocessing.StandardScaler().fit(df_t_tmp)
df_t_std = pd.DataFrame(scaler.transform(df_t_tmp), columns = list(df_t_tmp))
    
df_t_std['evadido'] = df_t.evadido

#df_s_std.to_csv('../Data Visualization/Coral Simulado/source_normalized.csv', index=False)
        
df_coral = coral.correlation_alignment(df_s_std, df_t_std, lambda_par=1, class_column='evadido')
df_coral.columns = ['f0','f1','evadido']
df_coral.to_csv('../Data Visualization/Coral Simulado/source_adapted_2.csv', index=False)
"""
fig = plt.figure()#figsize=(16,16))
ax = fig.add_subplot(111, projection='3d')

x_s =randrange(200,-0.7,0.7)
y_s =randrange(200,-0.7,0.7)
z_s =randrange(200,-3,3)

ax.scatter(x_s, y_s, z_s, c='r', marker='o')

x_t =randrange(100,-5,5)
y_t =randrange(100,-0.7,0.7)
z_t =randrange(100,-0.7,0.7)

df_s = pd.DataFrame({'f0':x_s, 'f1': y_s, 'f2': z_s})
df_t = pd.DataFrame({'f0':x_t, 'f1': y_t, 'f2': z_t})
df_s.to_csv('sint_source.csv', index=False)
df_t.to_csv('sint_test.csv', index=False)

ax.scatter(x_t, y_t, z_t, c='b', marker='o')

df_c = coral.correlation_alignment(df_s, df_t, class_column='')
df_c.columns = ['f0','f1','f2']

df_c.to_csv('sint_coral.csv', index=False)

ax.scatter(df_c.f0, df_c.f1, df_c.f2, c='g', marker='^')

ax.set_xlabel('f0')
ax.set_xlim(xmin=-5,xmax=5)
ax.set_ylabel('f1')
ax.set_ylim(ymin=-5,ymax=5)
ax.set_zlabel('f2')

plt.show()
"""


#sns.distplot(df_s.domain)
#sns.distplot(df_t.domain)

#g = sns.JointGrid(x="Forum_Quantidade_Post_Somado", y="Login_Quantidade", data=df_t) 
#g.plot_joint(sns.regplot, order=2) 
#g.plot_marginals(sns.distplot)
#sns.set(style="ticks")
#sns.pairplot(df_s)
#sns.pairplot(df_t)
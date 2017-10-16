# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 11:20:48 2017

@author: Everton

    Simulação para teste do algoritmo CORAL em variável
    sintética

"""

import sys
sys.path.insert(0, '../Algoritmos')

import filter as filter
import graficos as graficos
import pandas as pd
import numpy as np
import seaborn as sns
import coral as coral

from sklearn.ensemble import ExtraTreesClassifier
#Importando e configurando classificador (DecisionTree)
from sklearn.tree import DecisionTreeClassifier
#Importando gerador de parametros otimizados
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import matplotlib as mpl
## agg backend is used to create plot as a .png file
mpl.use('agg')

import seaborn
seaborn.set(style='ticks')

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

#Carrega dataset
df_s = pd.read_csv('../coral_source.csv', sep=',')
df_t = pd.read_csv('../coral_test.csv', sep=',')

df_s_cp = df_s.copy()
df_s_cp['x2']= randrange(112, 0, 1)
df_s_cp.to_csv('coral_source_refactor.csv', index=False)

df_coral = coral.correlation_alignment(df_s, df_t, lambda_par=0, class_column='classe')
df_coral.columns = ['x0','x1','classe']
df_s.to_csv('coral_result.csv', index=False)



#cols = list(df_s)

df_s['Origem']='Treino'
df_t['Origem']='Teste'
df_coral['Origem']='CORAL'

df_x = df_s.append(df_t, ignore_index=True)

df_x_tmp = df_x.copy()

df_z = df_x.append(df_coral, ignore_index=True)

colors = np.where(df_x_tmp.Origem=='Treino','r','g')

df_z.plot.scatter('x0','x1',c=colors)

fg = seaborn.FacetGrid(data=df_x_tmp, col='Origem', hue_order=[0, 1], hue="classe", hue_kws=dict(marker=["^", "v"]), aspect=1.61)
fg.map(plt.scatter, 'x0', 'x1').add_legend()

df_z.boxplot(['x0','x1'], by='Origem', grid=True)



#sns.distplot(df_s.domain)
#sns.distplot(df_t.domain)

#g = sns.JointGrid(x="Forum_Quantidade_Post_Somado", y="Login_Quantidade", data=df_t) 
#g.plot_joint(sns.regplot, order=2) 
#g.plot_marginals(sns.distplot)
#sns.set(style="ticks")
#sns.pairplot(df_s)
#sns.pairplot(df_t)
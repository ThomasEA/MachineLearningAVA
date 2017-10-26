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

import seaborn
seaborn.set(style='ticks')

#Carrega dataset
df_s = pd.read_csv('../Data Visualization/Coral Simulado/source_coral.csv', sep=',')
df_t = pd.read_csv('../Data Visualization/Coral Simulado/test_coral.csv', sep=',')

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
df_coral.to_csv('../Data Visualization/Coral Simulado/source_adapted.csv', index=False)



#sns.distplot(df_s.domain)
#sns.distplot(df_t.domain)

#g = sns.JointGrid(x="Forum_Quantidade_Post_Somado", y="Login_Quantidade", data=df_t) 
#g.plot_joint(sns.regplot, order=2) 
#g.plot_marginals(sns.distplot)
#sns.set(style="ticks")
#sns.pairplot(df_s)
#sns.pairplot(df_t)
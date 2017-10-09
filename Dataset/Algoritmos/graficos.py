# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 22:52:26 2017

@author: Everton

Métodos para plotar alguns gráficos auxiliares
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier

def feature_importance(X, y, model):
    #model = ExtraTreesClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature selection")
    plt.bar(range(X.shape[1]), importances[indices], color="gray", align="center")
    plt.xticks(range(X.shape[1]), X.columns, rotation = 90)
    
    plt.errorbar(range(X.shape[1]), importances[indices], capsize=5, capthick=0.5, fmt=' ', linewidth=3, elinewidth=1, ecolor='blue')
    #plt.errorbar(range(X.shape[1]), importances[indices], yerr=std[indices], capsize=5, capthick=0.5, fmt=' ', linewidth=3, elinewidth=1, ecolor='blue')
    plt.xlim([-1, X.shape[1]])
    plt.show()

def plot_cov_matrix(df_s, title='', calc_cov=True):
    df_s_tmp = df_s[df_s.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo', 'NumeroModulo','Evadido'])]
    
    if (calc_cov==True):
        corr = df_s_tmp.cov()
    else:
        corr = df_s_tmp
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    ax.set_title(title)
    
def plot_matrix(df_s, title=''):
    corr = df_s
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    ax.set_title(title)

    
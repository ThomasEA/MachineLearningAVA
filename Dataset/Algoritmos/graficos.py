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

def plot_corr_matrix(corr, title=''):
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

    
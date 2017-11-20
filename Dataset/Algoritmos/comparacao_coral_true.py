# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:51:13 2017

@author: Everton

Este é um teste de comparação do algoritmo CORAL, para verificar se está
corretamente implementado

"""

import pandas as pd
import numpy as np
from scipy import linalg as linear_alg
import graficos as graficos

def covariancia_mais_diag(df, lambda_par=1):
    df_ = df.cov()
    df_.replace(np.inf, 0,inplace=True)
    df_.replace(np.nan, 0,inplace=True)
    ID_S = np.eye(len(list(df_)))
    np.fill_diagonal(ID_S, lambda_par)
    return df_ + ID_S;

def whitening_values(df, kernel):
    c = linear_alg.fractional_matrix_power(kernel, -0.5)
    df = df.dot(c)
    return df;

def recolor_values(df, kernel):
    c = linear_alg.fractional_matrix_power(kernel, 0.5)
    df = df.dot(c)
    return df;

def correlation_alignment(df_s, df_t, lambda_par=1, class_column='Evadido', plot=False):
    
    if class_column in df_s:
        df_s_tmp = df_s[df_s.columns.difference([class_column])]
    else:
        df_s_tmp = df_s
    
    if class_column in df_t:
        df_t_tmp = df_t[df_t.columns.difference([class_column])]
    else:
        df_t_tmp = df_t
    
    if (plot==True):
        graficos.plot_cov_matrix(df_s_tmp, 'CORAL - Ds - Covariância Original')
        graficos.plot_cov_matrix(df_t_tmp, 'CORAL - Ts - Covariância Original')
    
    df_s_cov = covariancia_mais_diag(df_s_tmp, lambda_par=lambda_par)
    
    if (plot==True):
        graficos.plot_cov_matrix(df_s_cov, 'CORAL - Ds - Covariância + Identidade')
    
    df_t_cov = covariancia_mais_diag(df_t_tmp, lambda_par=lambda_par)

    if (plot==True):
        graficos.plot_cov_matrix(df_t_cov, 'CORAL - Ts - Covariância + Identidade')
    
    df_s_ = whitening_values(df_s_tmp, df_s_cov)
    
    if (plot==True):
        graficos.plot_cov_matrix(df_s_, 'CORAL - Ds - Covariancia Whitening')
    
    df_s_ = recolor_values(df_s_, df_t_cov)
    
    if (plot==True):
        graficos.plot_cov_matrix(df_s_, 'CORAL - Ds - Covariancia Re-color')
    
    df_s_.columns = df_s.columns.difference([class_column])
    
    if class_column in df_s:
        df_s_[class_column] = df_s[class_column]
    
    return df_s_;

#------------------------------
#Carrega dataset
df = pd.read_csv('../coral_source.csv', sep=',')
dft = pd.read_csv('../coral_test.csv', sep=',')
dfs_coral = correlation_alignment(df, dft, lambda_par=1, class_column='Evadido', plot=False)

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:20:18 2017

@author: Everton

    Implementação algoritmo ProbAl

"""

import numpy as np
from scipy import linalg as linear_alg
import graficos as graficos
from collections import Counter

def calc_mkv(df):
    for index, row in df.iterrows():
        a = row.values.tolist()

        b = []

        for x in range(len(a)):
            b.append(sum([a*b for a,b in zip(a[1:],a)]))

        #for x in xrange(len(a)): #for each row of the matrix
        #multiply each element of the row by each element of the vector and sum
        #the "zip" function is great for this
            
             

    print(b)
    return b;

def prob_mais_diag(df, lambda_par=1):
    df_ = calc_mkv(df)
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

def probability_alignment(df_s, df_t, lambda_par=1, class_column='Evadido'):
    
    if class_column in df_s:
        df_s_tmp = df_s[df_s.columns.difference([class_column])]
    else:
        df_s_tmp = df_s
    
    if class_column in df_t:
        df_t_tmp = df_t[df_t.columns.difference([class_column])]
    else:
        df_t_tmp = df_t
    
    graficos.plot_cov_matrix(df_s_tmp, 'CORAL - Ds - Covariância Original')
    graficos.plot_cov_matrix(df_t_tmp, 'CORAL - Ts - Covariância Original')
    
    df_s_cov = prob_mais_diag(df_s_tmp, lambda_par=lambda_par)
    
    graficos.plot_cov_matrix(df_s_cov, 'CORAL - Ds - Covariância + Identidade')
    
    df_t_cov = prob_mais_diag(df_t_tmp, lambda_par=lambda_par)

    graficos.plot_cov_matrix(df_t_cov, 'CORAL - Ts - Covariância + Identidade')
    
    df_s_ = whitening_values(df_s_tmp, df_s_cov)
    
    graficos.plot_cov_matrix(df_s_, 'CORAL - Ds - Covariancia Whitening')
    
    df_s_ = recolor_values(df_s_, df_t_cov)
    
    graficos.plot_cov_matrix(df_s_, 'CORAL - Ds - Covariancia Re-color')
    
    df_s_.columns = df_s.columns.difference([class_column])
    
    if class_column in df_s:
        df_s_[class_column] = df_s[class_column]
    
    return df_s_;

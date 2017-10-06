# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 05:58:49 2017

@author: Everton
@purpose: Teste de aplicação do método CORAL entre duas disciplinas
    Cs = cov(Ds) + eye(size(Ds,2))          
    Ct = cov(Dt) + eye(size(Dt,2))
    Ds = Ds * Cs^(-1/2)
    Ds = Ds * Ct^(1/2)
"""

# -*- coding: utf-8 -*-
from scipy import *

import numpy as np
import pandas as pd
import util as util

import graficos as graficos
import algoritmos as algoritmos

from sklearn.utils import shuffle

normalizar = True
plot_var_cov = True

modulo = '3'
s_disciplina = 'logica'
#s_disciplina = 'mat_adm'

df_s = pd.read_csv('../Week 3/m' + modulo + '_' + s_disciplina + '_ext_2012_01.csv', sep=',')
df_t = pd.read_csv('../Week 3/m' + modulo + '_' + s_disciplina + '_ext_2012_02_2014_01.csv', sep=',')

#Limpa e organiza algumas features e normaliza com z-score
df_s_std = util.clean_data(df_s, normalizar, plot_cov=False, title="Clean Data - Covariancia (Ds)")
df_t_std = util.clean_data(df_t, normalizar, plot_cov=False, title="Clean Data - Covariancia (Dt)")

#df_s_std = util.correlation_alignment(df_s_std, df_t_std,1)

#Embaralha dataframe normalizado
df_normalized = shuffle(df_s_std)
df_t_normalized = shuffle(df_t_std)

cm_final = algoritmos.predict_decision_tree(df_normalized, df_t_normalized)#, group_fold_column='CodigoTurma')

#Plota a matriz de confusão para o modelo
util.show_confusion_matrix(cm_final, class_labels=['Insucesso', 'Sucesso'])


"""
#Cria matriz identidade
    ID_S = np.eye(len(list(df_std)))
    C_S = df_std.cov() + ID_S
    #graficos.plot_corr_matrix(C_S, 'Cs [' + s_disciplina + ']')
    
    df_std = df_std * (C_S**(-1/2))
    df_std = df_std * (C_S**(1/2))
    #print(df_std.describe())
    print(C_S.describe())
    graficos.plot_corr_matrix(df_std.cov(), 'Covariância APÓS AJUSTE[' + s_disciplina + ']')
"""
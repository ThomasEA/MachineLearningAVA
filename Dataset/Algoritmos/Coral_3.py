# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 19:05:15 2017

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

#--------------------------------------------------------------

disciplinas = {50404: 'Fund. Proc. Administrativo', 
               60463: 'Ofic. Raciocínio Lógico',
               60465: 'Matemática Administração',
               60500: 'Lógica'}


#-------------------------------------------------------
# Configuração de filtros para o dataset
disciplina = 60500
modulo = 3 #0 = ignora o módulo. Lembrando que só existem os módulos 3 e 6
periodo_letivo_source = [20120101,20120102,20130101]
periodo_letivo_test   = [20130102,20140101]
#-------------------------------------------------------

s_periodo = str(periodo_letivo_source)
t_periodo = str(periodo_letivo_test)

#Carrega dataset
df = pd.read_csv('../dataset_m3_m6.csv', sep=';')

#Apaga colunas com zeros
df = df.loc[:, (df != 0).any(axis=0)]

print('Total registros geral: ' + str(len(df)))

print('Análise disciplina: ' + disciplinas[disciplina] + ' / Módulo: ' + ('Todos' if modulo == '0' else str(modulo)))
print('Período origem..: ' + s_periodo)
print('Período teste...: ' + t_periodo)

df_s = df.copy()
df_t = df.copy()

#Filta registros conforme os parâmetros
if (modulo == 0):
    df_s = df_s.loc[df_s['CodigoDisciplina'] == disciplina]
    df_t = df_t.loc[df_t['CodigoDisciplina'] == disciplina]

if (disciplina > 0):
    df_s = df_s.loc[(df_s['CodigoDisciplina'] == disciplina)]
    df_t = df_t.loc[(df_t['CodigoDisciplina'] == disciplina)]

if (len(periodo_letivo_source) > 0):
    df_s = df_s.loc[(df_s['PeriodoLetivo'].isin(periodo_letivo_source))]

if (len(periodo_letivo_test) > 0):
    df_t = df_t.loc[(df_t['PeriodoLetivo'].isin(periodo_letivo_test))]

df_s = df_s.reset_index(drop=True)
df_t = df_t.reset_index(drop=True)

df_s_filter = util.clean_data(df_s, standardize=True, plot_cov=False, title='Matriz de Covariância - ' + disciplinas[disciplina] + ' / ' + s_periodo)
df_t_filter = util.clean_data(df_t, standardize=True, plot_cov=False, title='Matriz de Covariância - ' + disciplinas[disciplina] + ' / ' + t_periodo)

print('Registros source: ' + str(len(df_s_filter)))
print('Registros target: ' + str(len(df_t_filter)))


#df_s_std = util.correlation_alignment(df_s_filter, df_t_filter,1)
df_s_std = df_s_filter

#Embaralha dataframe normalizado
df_s_normalized = shuffle(df_s_std)
df_t_normalized = shuffle(df_t_filter)

cm_final = algoritmos.predict_decision_tree(df_s_normalized, df_t_normalized, group_fold_column='CodigoTurma')

#Plota a matriz de confusão para o modelo
util.show_confusion_matrix(cm_final, class_labels=['Insucesso', 'Sucesso'])


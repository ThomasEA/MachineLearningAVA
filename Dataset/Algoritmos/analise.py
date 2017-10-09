# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 05:52:26 2017

@author: Everton

    Monta análise estatística sobre um dataset
"""

import pandas as pd
import numpy as np
import util as util
import graficos as graficos
import algoritmos as algoritmos

from sklearn import preprocessing

import matplotlib.pyplot as plt
from scipy import stats, integrate
import seaborn as sns

#Importando e configurando classificador (DecisionTree)
from sklearn.tree import DecisionTreeClassifier
#Importando gerador de parametros otimizados
from sklearn.model_selection import GridSearchCV

pd.options.mode.chained_assignment = None


    

#--------------------------------------------------------------

disciplinas = {50404: 'Fund. Proc. Administrativo', 
               60463: 'Ofic. Raciocínio Lógico',
               60465: 'Matemática Administração',
               60500: 'Lógica'}

"""
Configuração de filtros para o dataset
"""
disciplina = 60500
modulo = 6 #0 = ignora o módulo. Lembrando que só existem os módulos 3 e 6
#periodo_letivo = [20120101,20120201,20130101]
periodo_letivo = [20130201,20140101]

s_periodo = '[2012/01 a 2014/01]' if len(periodo_letivo) == 0 else str(periodo_letivo)
"""
"""

df = pd.read_csv('../dataset_m3_m6.csv', sep=';')

df = df.loc[:, (df != 0).any(axis=0)]

print('Total registros geral: ' + str(len(df)))

print('Análise disciplina: ' + disciplinas[disciplina] + ' / Módulo: ' + ('Todos' if modulo == '0' else str(modulo)))
print('Período: ' + s_periodo)

df_f = df

#Filta registros conforme os parâmetros
if (modulo == 0):
    df_f = df_f.loc[df_f['CodigoDisciplina'] == disciplina]

if (disciplina > 0):
    df_f = df_f.loc[(df_f['CodigoDisciplina'] == disciplina)]

if (len(periodo_letivo) > 0):
    df_f = df_f.loc[(df_f['PeriodoLetivo'].isin(periodo_letivo))]

df_f = df_f.reset_index(drop=True)

df_filter = util.clean_data(df_f, standardize=True, plot_cov=True, title='Matriz de Covariância - ' + disciplinas[disciplina] + ' / ' + s_periodo)

total_regs = len(df_filter)
total_sucesso = len(df_filter.loc[df_filter['Evadido'] == 1])
total_insucesso = len(df_filter.loc[df_filter['Evadido'] == 0])
    
print('\tTotal registros: ' + str(total_regs))
print('\tSucesso: %d / %.2f%%' % (total_sucesso, total_sucesso / total_regs * 100))
print('\tInsucesso: %d / %.2f%%' % (total_insucesso, total_insucesso / total_regs * 100))

target = df_filter.Evadido
features = df_filter[df_filter.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','NumeroModulo','Evadido'])]

clf = DecisionTreeClassifier()
#ft = algoritmos.fit_features(clf, features, target)#, group_fold_column='CodigoTurma')

graficos.feature_importance(features, target, clf)


sns.distplot(df_filter.Evadido)



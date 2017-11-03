# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 23:07:40 2017

@author: Everton

Métodos para filtragem de datasets

"""

import pandas as pd
from sklearn import preprocessing

def filter_dataset(df_original, modulo, disciplinas, disciplina, periodo_letivo_s, periodo_letivo_t, feature_normalization=True):
    
    print('Total registros geral: ' + str(len(df_original)))
    print('Disciplina......: ' + disciplinas[disciplina] + ' / Módulo: ' + ('Todos' if modulo == '0' else str(modulo)))
    
    #Aproveita e apaga colunas com zeros
    df = df_original.copy()
    df = df.loc[:, (df != 0).any(axis=0)]

    #Altera os valores da coluna Evadido para binário
    df.Evadido = df.Evadido.map({'ReprEvadiu': 1, 'Sucesso': 0})

    df_s = df.copy()
    df_t = df.copy()

    #Filta registros conforme os parâmetros
    if (modulo > 0):
        df_s = df_s.loc[df_s['NumeroModulo'] == modulo]
        df_t = df_t.loc[df_t['NumeroModulo'] == modulo]
    
    if (disciplina > 0):
        df_s = df_s.loc[(df_s['CodigoDisciplina'] == disciplina)]
        df_t = df_t.loc[(df_t['CodigoDisciplina'] == disciplina)]
    
    if (len(periodo_letivo_s) > 0):
        df_s = df_s.loc[(df_s['PeriodoLetivo'].isin(periodo_letivo_s))]
    
    if (len(periodo_letivo_t) > 0):
        df_t = df_t.loc[(df_t['PeriodoLetivo'].isin(periodo_letivo_t))]
    
    s_periodo = str(periodo_letivo_s)
    t_periodo = str(periodo_letivo_t)
    
    print('Período source..: ' + s_periodo)
    print('\tQtd. Registros: ' + str(len(df_s)))
    print('\tSucesso.......: %d / %.2f%%' % (len(df_s[df_s.Evadido == 0]), len(df_s[df_s.Evadido == 0]) / len(df_s) * 100))
    print('\tInsucesso.....: %d / %.2f%%' % (len(df_s[df_s.Evadido == 1]), len(df_s[df_s.Evadido == 1]) / len(df_s) * 100))
    
    print('Período test....: ' + t_periodo)
    print('\tQtd. Registros: ' + str(len(df_t)))
    print('\tSucesso.......: %d / %.2f%%' % (len(df_t[df_t.Evadido == 0]), len(df_t[df_t.Evadido == 0]) / len(df_t) * 100))
    print('\tInsucesso.....: %d / %.2f%%' % (len(df_t[df_t.Evadido == 1]), len(df_t[df_t.Evadido == 1]) / len(df_t) * 100))
    
    #Como já fez o filtro, remove algumas colunas que não são mais necessárias
    df_s = df_s.drop(['NumeroModulo','CodigoDisciplina','PeriodoLetivo','CodigoTurma'], axis=1)
    df_t = df_t.drop(['NumeroModulo','CodigoDisciplina','PeriodoLetivo','CodigoTurma'], axis=1)

    df_s = df_s.reset_index(drop=True)
    df_t = df_t.reset_index(drop=True)
    
    if (feature_normalization==True):
        #Computa o Z-Score para o dataset
        scaler = preprocessing.StandardScaler().fit(df_s)
        df_s_std = pd.DataFrame(scaler.transform(df_s), columns = list(df_s))
    
        df_s_std['Evadido'] = df_s.Evadido
    
        scaler = preprocessing.StandardScaler().fit(df_t)
        df_t_std = pd.DataFrame(scaler.transform(df_t), columns = list(df_t))
    
        df_t_std['Evadido'] = df_t.Evadido
                
        return df_s_std, df_t_std
    else:
        return df_s, df_t
    
    
def filter_dataset_mult(df_original, 
                        disciplinas, 
                        modulo_s, 
                        modulo_t, 
                        disciplina_s, 
                        disciplina_t, 
                        periodo_letivo_s, 
                        periodo_letivo_t, 
                        features = None,
                        feature_normalization=True):
    
    print('Total registros geral: ' + str(len(df_original)))
    
    #Aproveita e apaga colunas com zeros
    df = df_original.copy()
    df = df.loc[:, (df != 0).any(axis=0)]

    #Altera os valores da coluna Evadido para binário
    df.Evadido = df.Evadido.map({'ReprEvadiu': 1, 'Sucesso': 0})

    df_s = df.copy()
    df_t = df.copy()

    #Filta registros conforme os parâmetros para o filtro de origem
    if (modulo_s > 0):
        df_s = df_s.loc[df_s['NumeroModulo'] == modulo_s]
    
    if (modulo_t > 0):
        df_t = df_t.loc[df_t['NumeroModulo'] == modulo_t]
    
    if (disciplina_s > 0):
        df_s = df_s.loc[(df_s['CodigoDisciplina'] == disciplina_s)]
    
    if (disciplina_t > 0):
        df_t = df_t.loc[(df_t['CodigoDisciplina'] == disciplina_t)]
    
    if (len(periodo_letivo_s) > 0):
        df_s = df_s.loc[(df_s['PeriodoLetivo'].isin(periodo_letivo_s))]
    
    if (len(periodo_letivo_t) > 0):
        df_t = df_t.loc[(df_t['PeriodoLetivo'].isin(periodo_letivo_t))]
    
    s_periodo = str(periodo_letivo_s)
    t_periodo = str(periodo_letivo_t)

    print('Features: ' + ('Todas' if features == None else str(features)))
    print('---------------------------------------------------------')
    print('TREINO: ' + disciplinas[disciplina_s] + ' / Módulo: ' + ('Todos' if modulo_s == '0' else str(modulo_s)))
    print('\tPeríodo........: ' + s_periodo)
    print('\tQtd. Registros.: ' + str(len(df_s)))
    print('\tSucesso........: %d / %.2f%%' % (len(df_s[df_s.Evadido == 0]), len(df_s[df_s.Evadido == 0]) / len(df_s) * 100))
    print('\tInsucesso......: %d / %.2f%%' % (len(df_s[df_s.Evadido == 1]), len(df_s[df_s.Evadido == 1]) / len(df_s) * 100))
    
    print('TESTE: ' + disciplinas[disciplina_t] + ' / Módulo: ' + ('Todos' if modulo_t == '0' else str(modulo_t)))    
    print('\tPeríodo........: ' + t_periodo)
    print('\tQtd. Registros.: ' + str(len(df_t)))
    print('\tSucesso........: %d / %.2f%%' % (len(df_t[df_t.Evadido == 0]), len(df_t[df_t.Evadido == 0]) / len(df_t) * 100))
    print('\tInsucesso......: %d / %.2f%%' % (len(df_t[df_t.Evadido == 1]), len(df_t[df_t.Evadido == 1]) / len(df_t) * 100))
    
    #Como já fez o filtro, remove algumas colunas que não são mais necessárias
    df_s = df_s.drop(['NumeroModulo','CodigoDisciplina','PeriodoLetivo','CodigoTurma'], axis=1)
    df_t = df_t.drop(['NumeroModulo','CodigoDisciplina','PeriodoLetivo','CodigoTurma'], axis=1)

    if features!=None:
        df_s = df_s[features]
        df_t = df_t[features]

    df_s = df_s.reset_index(drop=True)
    df_t = df_t.reset_index(drop=True)
    
    if (feature_normalization==True):
        #Computa o Z-Score para o dataset
        scaler = preprocessing.StandardScaler().fit(df_s)
        df_s_std = pd.DataFrame(scaler.transform(df_s), columns = list(df_s))
    
        df_s_std['Evadido'] = df_s.Evadido
    
        scaler = preprocessing.StandardScaler().fit(df_t)
        df_t_std = pd.DataFrame(scaler.transform(df_t), columns = list(df_t))
    
        df_t_std['Evadido'] = df_t.Evadido
                
        return df_s_std, df_t_std
    else:
        return df_s, df_t
    
def filter_dataset_turmas(df_original, 
                        disciplinas, 
                        modulo_s, 
                        modulo_t, 
                        disciplina_s, 
                        features = None,
                        feature_normalization=True):
    
    print('Total registros geral: ' + str(len(df_original)))
    
    #Aproveita e apaga colunas com zeros
    df = df_original.copy()
    df = df.loc[:, (df != 0).any(axis=0)]

    #Altera os valores da coluna Evadido para binário
    df.Evadido = df.Evadido.map({'ReprEvadiu': 1, 'Sucesso': 0})

    df_s = df.copy()
    df_t = df.copy()

    #Filta registros conforme os parâmetros para o filtro de origem
    if (modulo_s > 0):
        df_s = df_s.loc[df_s['NumeroModulo'] == modulo_s]
    
    if (modulo_t > 0):
        df_t = df_t.loc[df_t['NumeroModulo'] == modulo_t]
    
    if (disciplina_s > 0):
        df_s = df_s.loc[(df_s['CodigoDisciplina'] == disciplina_s)]
    
    if (disciplina_s > 0):
        df_t = df_t.loc[(df_t['CodigoDisciplina'] != disciplina_s)]
    
    print('Features: ' + ('Todas' if features == None else str(features)))
    print('---------------------------------------------------------')
    print('TREINO: ' + disciplinas[disciplina_s] + ' / Módulo: ' + ('Todos' if modulo_s == '0' else str(modulo_s)))
    print('\tQtd. Registros.: ' + str(len(df_s)))
    print('\tSucesso........: %d / %.2f%%' % (len(df_s[df_s.Evadido == 0]), len(df_s[df_s.Evadido == 0]) / len(df_s) * 100))
    print('\tInsucesso......: %d / %.2f%%' % (len(df_s[df_s.Evadido == 1]), len(df_s[df_s.Evadido == 1]) / len(df_s) * 100))
    
    print('TESTE:  / Módulo: ' + ('Todos' if modulo_t == '0' else str(modulo_t)))    
    print('\tQtd. Registros.: ' + str(len(df_t)))
    print('\tSucesso........: %d / %.2f%%' % (len(df_t[df_t.Evadido == 0]), len(df_t[df_t.Evadido == 0]) / len(df_t) * 100))
    print('\tInsucesso......: %d / %.2f%%' % (len(df_t[df_t.Evadido == 1]), len(df_t[df_t.Evadido == 1]) / len(df_t) * 100))
    
    #Como já fez o filtro, remove algumas colunas que não são mais necessárias
    df_s = df_s.drop(['NumeroModulo','CodigoDisciplina','PeriodoLetivo'], axis=1)
    df_t = df_t.drop(['NumeroModulo','CodigoDisciplina','PeriodoLetivo'], axis=1)

    if features!=None:
        df_s = df_s[features]
        df_t = df_t[features]

    df_s = df_s.reset_index(drop=True)
    df_t = df_t.reset_index(drop=True)
    
    if (feature_normalization==True):
        #Computa o Z-Score para o dataset
        scaler = preprocessing.StandardScaler().fit(df_s)
        df_s_std = pd.DataFrame(scaler.transform(df_s), columns = list(df_s))
    
        df_s_std['Evadido'] = df_s.Evadido
        df_s_std['CodigoTurma'] = df_s.CodigoTurma
    
        scaler = preprocessing.StandardScaler().fit(df_t)
        df_t_std = pd.DataFrame(scaler.transform(df_t), columns = list(df_t))
    
        df_t_std['Evadido'] = df_t.Evadido
        df_t_std['CodigoTurma'] = df_t.CodigoTurma
                
        return df_s_std, df_t_std
    else:
        return df_s, df_t

"""
    Filtra o dataset por disciplina, mantendo a coluna CodigoTurma e Evadido
    Esse método é utilizado pelo algoritmo de teste de acurácia baseado
    em turma-fold
"""
def filter_ds_turma(df_original, 
                        disciplinas, 
                        modulo_s, 
                        disciplina_s, 
                        features = None,
                        feature_normalization=True):
    
    print('Total registros geral: ' + str(len(df_original)))
    
    #Aproveita e apaga colunas com zeros
    df = df_original.copy()
    df = df.loc[:, (df != 0).any(axis=0)]

    #Altera os valores da coluna Evadido para binário
    df.Evadido = df.Evadido.map({'ReprEvadiu': 1, 'Sucesso': 0})

    df_s = df.copy()

    #Filta registros conforme os parâmetros para o filtro de origem
    if (modulo_s > 0):
        df_s = df_s.loc[df_s['NumeroModulo'] == modulo_s]
    
    if (disciplina_s > 0):
        df_s = df_s.loc[(df_s['CodigoDisciplina'] == disciplina_s)]
    
    print('Features: ' + ('Todas' if features == None else str(features)))
    print('---------------------------------------------------------')
    print(disciplinas[disciplina_s] + ' / Módulo: ' + ('Todos' if modulo_s == '0' else str(modulo_s)))
    print('\tQtd. Registros.: ' + str(len(df_s)))
    print('\tSucesso........: %d / %.2f%%' % (len(df_s[df_s.Evadido == 0]), len(df_s[df_s.Evadido == 0]) / len(df_s) * 100))
    print('\tInsucesso......: %d / %.2f%%' % (len(df_s[df_s.Evadido == 1]), len(df_s[df_s.Evadido == 1]) / len(df_s) * 100))
    
    #Como já fez o filtro, remove algumas colunas que não são mais necessárias
    df_s = df_s.drop(['NumeroModulo','CodigoDisciplina','PeriodoLetivo'], axis=1)

    if features!=None:
        df_s = df_s[features]

    df_s = df_s.reset_index(drop=True)
    
    if (feature_normalization==True):
        #Computa o Z-Score para o dataset
        scaler = preprocessing.StandardScaler().fit(df_s)
        df_s_std = pd.DataFrame(scaler.transform(df_s), columns = list(df_s))
    
        df_s_std['Evadido'] = df_s.Evadido
        df_s_std['CodigoTurma'] = df_s.CodigoTurma
    
        return df_s_std
    else:
        return df_s
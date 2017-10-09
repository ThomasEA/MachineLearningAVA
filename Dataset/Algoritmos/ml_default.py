# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 22:57:58 2017

@author: Everton

Modelo preditivo padrão
    - Cross validation
    - Balanceamento de classes
"""
import pandas as pd

import filter as filter

from sklearn.model_selection import StratifiedKFold

#-------------------------------------------------------
# Configuração de filtros para o dataset
disciplina = 60500
modulo = 3 #0 = ignora o módulo. Lembrando que só existem os módulos 3 e 6
periodo_letivo_source = [20120101,20120102,20120201,20120202]
periodo_letivo_test   = [20130101,20130102,20130201,20130202,20140101,20140102]
#-------------------------------------------------------

disciplinas = {
        50404: 'Fund. Proc. Administrativo', 
        60463: 'Ofic. Raciocínio Lógico',
        60465: 'Matemática Administração',
        60500: 'Lógica'
    }

s_periodo = str(periodo_letivo_source)
t_periodo = str(periodo_letivo_test)

#Carrega dataset
df = pd.read_csv('../dataset_m3_m6.csv', sep=';')

#Filtra o dataset conforme a configuração selecionada e faz alguns ajustes no
#dataset
df_s, df_t = filter.filter_dataset(
                            df, 
                            modulo, 
                            disciplinas, 
                            disciplina, 
                            periodo_letivo_source, 
                            periodo_letivo_test)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)
i = 1
for train, test in skf.split(df_s, df_s.Evadido):
    y_pred = test.copy()
    fold = df_s.ix[test]
    
    
    print('Fold: %d ' % i)
    print('\tQtd. Registros: %d' % len(fold))
    #df_s = df_s.loc[(df_s['Evadido'] == 1)]
    #print(train)
    print('\tSucesso.......: %d / %.2f%%' % (len(fold[fold.Evadido == 1]), len(fold[fold.Evadido == 1]) / len(fold) * 100))
    print('\tInsucesso.....: %d / %.2f%%' % (len(fold[fold.Evadido == 0]), len(fold[fold.Evadido == 0]) / len(fold) * 100))
    
    #print(fold.head())
    i+=1


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:34:51 2017

@author: kratos
"""
import numpy as np
import util as util

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#Importando e configurando classificador (DecisionTree)
from sklearn.tree import DecisionTreeClassifier
#Importando gerador de parametros otimizados
from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing

def __decision_tree(df_s, df_t, dt):
    qtd_ex_treino = len(df_s)
    qtd_ex_teste = len(df_t)
    
    print('\t[DT: ] Registros: ' + str(qtd_ex_teste + qtd_ex_treino) + ' / Treino: ' + str(qtd_ex_treino) + ' / Teste: ' + str(qtd_ex_teste))
 
    #Separa os dados de treino do atributo de predição
    features = df_s[df_s.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
    target = df_s.Evadido
    
    dt.fit(features, target)
    
    #Separa os dados de teste do atributo de predição
    features_test = df_t[df_t.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','NumeroModulo','Evadido'])]
    target_test = df_t.Evadido
    
    predicted = dt.predict(features_test)
    
    return confusion_matrix(target_test, predicted);

def predict_decision_tree(df_source, df_test=None, group_fold_column=None):
    clf_param = {'max_depth': range(3,10)}
    clf = DecisionTreeClassifier()
    
    #Cross-validation
    cm_final = np.matrix('0 0; 0 0')
    
    dt = GridSearchCV(clf, clf_param)
    
    if (group_fold_column==None):
        cm_final = cm_final + __decision_tree(df_source, df_test, dt)
    else:
        #Separa os folds por turma
        folds = util.sep_folds(df_source, group_fold_column)
        
        qtd_folds = len(folds.groups)
    
        print('====================')
        print('Coss-validation: (k = ' + str(qtd_folds) + ')')
        for key in folds.groups.keys():
            fold_teste = folds.get_group(name=key).copy()
            fold_treino = folds.filter(lambda x: x.name!=key).copy()
            
            #fold_treino = util.correlation_alignment(fold_treino, fold_teste,1)
            cm_final = cm_final + __decision_tree(fold_treino, fold_teste, dt)
        
        #Separa os dados de teste do atributo de predição
        features_test = df_test[df_test.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','NumeroModulo','Evadido'])]
        target_test = df_test.Evadido
    
        predicted = dt.predict(features_test)
    
        cm_final = confusion_matrix(target_test, predicted);
    return cm_final;        
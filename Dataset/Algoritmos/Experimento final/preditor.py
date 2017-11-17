# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 07:54:01 2017

@author: Everton

Modelo preditivo parametrizável

"""
import sys
sys.path.insert(0, '../../Algoritmos')

import pandas as pd
import numpy as np
import filter as filter
import util as util
import coral as coral

from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
#Importando e configurando classificador (DecisionTree)
from sklearn.tree import DecisionTreeClassifier
#Importando e configurando classificador (Naive Bayes)
from sklearn.naive_bayes import GaussianNB
#Importando e configurando classificador (SVM)
from sklearn import svm
#Importando gerador de parametros otimizados
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def process(df, disciplina_s, modulo_s, classificador, use_coral, use_normalization, use_normalization_turma):
    #-------------------------------------------------------
    # Configuração de filtros para o dataset
    features = {
            50404: ['Questionario_Quantidade_Somado', 'Forum_TempoUso_Somado', 'Log_Post_Quantidade_Somado', 'Questionario_TempoUso_Somado', 'Login_Quantidade','Turno_TempoUsoTotal_Somado', 'Evadido','CodigoTurma'],
            60463: ['Turno_TempoUsoTotal_Somado', 'Login_Quantidade', 'Log_Post_Quantidade_Somado', 'Questionario_TempoUso_Somado', 'Log_View_Quantidade_Somado', 'Numero_Dias_Acessados_Modulo_Somado', 'Evadido','CodigoTurma'],
            60465: ['Log_Post_Quantidade_Somado', 'Login_Quantidade', 'Assignment_View_Quantidade_Somado', 'Turno_TempoUsoTotal_Somado', 'Numero_Dias_Acessados_Modulo_Somado','Log_View_Quantidade_Somado', 'Evadido','CodigoTurma'],
            60500: ['Assignment_View_Quantidade_Somado', 'Turno_TempoUsoTotal_Somado', 'Log_View_Quantidade_Somado', 'Login_Quantidade', 'Resource_View_Tempo_Somado','Forum_TempoUso_Somado', 'Evadido','CodigoTurma']
        }
            
    shuf = True
    coral_lambda = 1
    #-------------------------------------------------------
    disciplinas = {
            50404: 'Fund. Proc. Administrativo', 
            60463: 'Ofic. Raciocínio Lógico',
            60465: 'Matemática Administração',
            60500: 'Lógica'
        }
    
    classificadores = {
                1: 'Naive Bayes',
                2: 'Decision Tree',
                3: 'SVM'
            }
    
    disciplina_string = str(disciplinas[disciplina_s])
    
    print('CLASSIFICADOR: %s' % classificadores[classificador])
    
    #Filtra o dataset conforme a configuração selecionada e faz alguns ajustes no dataset
    df_s = filter.filter_ds_turma(
                                df,
                                disciplinas, 
                                modulo_s, 
                                disciplina_s, 
                                features[disciplina_s],
                                feature_normalization=use_normalization)
    
    df_s_folds = util.sep_folds(df_s,'CodigoTurma')
    
    result = pd.DataFrame()

    model = None

    if classificador == 1:
        model = GaussianNB()
    if classificador == 2:
        clf = DecisionTreeClassifier(max_depth=3)
        clf_param = {'max_depth': range(3,10)}
        model = GridSearchCV(clf, clf_param)
    if classificador == 3:
        parameters = {'kernel':('linear', 'rbf'), 'C': range(1, 20), 'cache_size': [1, 500000]}
        clf = svm.SVC()
        model = GridSearchCV(clf, parameters)    

    i = 1

    print('---- Folds -----')
    
    for name, fold in df_s_folds:
        fold_t = fold[features[disciplina_s]]
        
        if (shuf==True): fold_t = shuffle(fold_t)
    
        target_t = fold_t.Evadido
        
        if (use_normalization_turma == True):
            fold_t_norm = util.normalize(fold_t, ['Evadido','CodigoTurma'])
        else:
            fold_t_norm = fold_t
        
        fold_t_norm = fold_t_norm.loc[:, fold_t_norm.columns != 'Evadido']
        fold_t_norm = fold_t_norm.loc[:, fold_t_norm.columns != 'CodigoTurma']
        
        fold_s = pd.DataFrame()
        
        turmas_treino = []
         
        #monta o dataset de treinamento
        for name_s, fold_stmp in df_s_folds:
            if (name_s != name):
                turmas_treino.append(name_s)
                
                if (use_normalization_turma == True):
                    f_norm = util.normalize(fold_stmp, ['Evadido','CodigoTurma'])
                else:
                    f_norm = fold_stmp
                
                fold_s = pd.concat([fold_s,f_norm])
        
        fold_s = fold_s[features[disciplina_s]]
        
        if (shuf==True): fold_s = shuffle(fold_s)
        
        target_s = fold_s.Evadido
        fold_s = fold_s.loc[:, fold_s.columns != 'Evadido']
        fold_s = fold_s.loc[:, fold_s.columns != 'CodigoTurma']
    
        print('\tFold: %d: [Treino %d registros] - Turmas: %s' % (i, len(fold_s), turmas_treino))
        print('\t\t\tSucesso: %d (%.2f%%) / Insucesso: %d (%.2f%%)' % 
              (len(target_s[target_s == 0]), 
               len(target_s[target_s == 0]) / len(target_s) * 100, 
               len(target_s[target_s == 1]), 
               len(target_s[target_s == 1]) / len(target_s) * 100))
        
        print('\t         [Teste %d registros] - Turma: [%s]' % (len(fold_t), name))
        print('\t\t\tSucesso: %d (%.2f%%) / Insucesso: %d (%.2f%%)' % 
              (len(target_t[target_t == 0]), 
               len(target_t[target_t == 0]) / len(target_t) * 100, 
               len(target_t[target_t == 1]), 
               len(target_t[target_t == 1]) / len(target_t) * 100))
    
        if (use_coral == True):
            fold_s_norm = coral.correlation_alignment(fold_s, fold_t_norm, lambda_par=coral_lambda)
            model.fit(fold_s_norm, target_s)    
        else:
            model.fit(fold_s, target_s)    

        predicted = model.predict(fold_t_norm)
    
        accuracy = accuracy_score(target_t, predicted)                                
        
        print('\t         Acurárica......: [%.2f]' % accuracy)
        
        result.set_value(i,'Classificador',classificadores[classificador])
        result.set_value(i,'Disciplina',disciplina_string)
        result.set_value(i,'Turma','T' + str(i))
        result.set_value(i,'Coral', use_coral)
        result.set_value(i,'Acur',accuracy * 100)
        result.set_value(i,'TreinoSucesso',len(target_s[target_s == 0]) / len(target_s) * 100)
        result.set_value(i,'TreinoInsucesso',len(target_s[target_s == 1]) / len(target_s) * 100)
        result.set_value(i,'TreinoDesbalanceamento',(len(target_s[target_s == 1]) / len(target_s) * 100) / (len(target_s[target_s == 0]) / len(target_s) * 100) * 5)
        result.set_value(i,'TesteSucesso',len(target_t[target_t == 0]) / len(target_t) * 100)
        result.set_value(i,'TesteInsucesso',len(target_t[target_t == 1]) / len(target_t) * 100)
        result.set_value(i,'TesteDesbalanceamento',(len(target_t[target_t == 1]) / len(target_t) * 100) / (len(target_t[target_t == 0]) / len(target_t) * 100) * 5)
        
        i += 1

    return result;
  

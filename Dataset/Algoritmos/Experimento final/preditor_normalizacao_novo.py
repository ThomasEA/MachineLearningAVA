# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 07:54:01 2017

@author: Everton

Modelo preditivo parametrizável
Utiliza o modelo de normalização descrito pelo Wilson.
- Normalizar o teste dentro do fold
- Normalizar todo o treino dentro do fold

"""
import sys
sys.path.insert(0, '../../Algoritmos')

import pandas as pd
import numpy as np
import filter as filter
import util as util
import coral as coral

import seaborn as sns

from sklearn.utils import shuffle
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
from math import sqrt

def process(df, disciplina_s, modulo_s, classificador, use_coral):
    #-------------------------------------------------------
    # Configuração de filtros para o dataset
    features = {
            50404: ['Questionario_Quantidade_Somado', 'Forum_TempoUso_Somado', 'Log_Post_Quantidade_Somado', 'Questionario_TempoUso_Somado', 'Login_Quantidade','Turno_TempoUsoTotal_Somado', 'Evadido','CodigoTurma'],
            60463: ['Turno_TempoUsoTotal_Somado', 'Login_Quantidade', 'Log_Post_Quantidade_Somado', 'Questionario_TempoUso_Somado', 'Log_View_Quantidade_Somado', 'Numero_Dias_Acessados_Modulo_Somado', 'Evadido','CodigoTurma'],
            60465: ['Log_Post_Quantidade_Somado', 'Login_Quantidade', 'Assignment_View_Quantidade_Somado', 'Turno_TempoUsoTotal_Somado', 'Numero_Dias_Acessados_Modulo_Somado','Log_View_Quantidade_Somado', 'Evadido','CodigoTurma'],
            60500: ['Assignment_View_Quantidade_Somado', 'Turno_TempoUsoTotal_Somado', 'Log_View_Quantidade_Somado', 'Login_Quantidade', 'Resource_View_Tempo_Somado','Forum_TempoUso_Somado', 'Evadido','CodigoTurma']
        }
            
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
                                feature_normalization=False)
    
    df_s.to_csv('../../temp/m{}_{}_turmas.csv'.format(modulo_s, disciplina_string))
    
    df_s_folds = util.sep_folds(df_s,'CodigoTurma')
    
    sns.distplot(df_s.Evadido)
    
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
    
    
    
    cm = np.matrix('0.0 0.0; 0.0 0.0')
    
    print('---- Folds -----')
    
    for name, fold in df_s_folds:
        fold_t = fold[features[disciplina_s]].copy()
        
        #faz o shuffle do teste
        fold_t = shuffle(fold_t)
    
        target_t = fold_t.Evadido
        
        #normaliza o teste
        fold_t_norm = util.normalize(fold_t, ['Evadido','CodigoTurma'])
        
        fold_t_norm = fold_t_norm.loc[:, fold_t_norm.columns != 'Evadido']
        fold_t_norm = fold_t_norm.loc[:, fold_t_norm.columns != 'CodigoTurma']
        
        fold_s = pd.DataFrame()
        
        turmas_treino = []
         
        #monta o dataset de treinamento
        for name_s, fold_stmp in df_s_folds:
            if (name_s != name):
                turmas_treino.append(name_s)
                
                f_norm = fold_stmp.copy()
                
                fold_s = pd.concat([fold_s,f_norm])
        
        #faz o shuffke do treino
        fold_s = shuffle(fold_s)
        
        #normaliza o treino
        fold_s = util.normalize(fold_s, ['Evadido','CodigoTurma'])
        
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
        
        cm_tmp = confusion_matrix(target_t, predicted)
        
        cm += confusion_matrix(target_t, predicted)
        
        TN = 0.0
        FN = 0.0
        TP = 0.0
        FP = 0.0
        
        TN = cm_tmp[0][0]
        FN = cm_tmp[1][0]
        FP = cm_tmp[0][1]
        TP = cm_tmp[1][1]
        
        TPR = (TP / (TP+FN))
        TNR = (TN / (TN+FP))
        gmean = sqrt(TPR*TNR)
        
        print('\t         Acurárica......: [%.2f]' % accuracy)
        
        result.set_value(i,'Classificador',classificadores[classificador])
        result.set_value(i,'Disciplina',disciplina_string)
        result.set_value(i,'Turma','T' + str(i))
        result.set_value(i,'Coral', use_coral)
        result.set_value(i,'Acur',accuracy * 100)
        result.set_value(i,'Precision Sucesso', (TN / (TN + FN)) * 100 )
        result.set_value(i,'Recall Sucesso', (TN / (TN+FP)) * 100 )
        result.set_value(i,'Precision Insucesso', (TP / (TP + FP)) * 100 )
        result.set_value(i,'Recall Insucesso', (TP / (TP+FN)) * 100 )
        result.set_value(i,'TreinoSucesso',len(target_s[target_s == 0]) / len(target_s) * 100)
        result.set_value(i,'TreinoInsucesso',len(target_s[target_s == 1]) / len(target_s) * 100)
        result.set_value(i,'TreinoDesbalanceamento',(len(target_s[target_s == 1]) / len(target_s) * 100) / (len(target_s[target_s == 0]) / len(target_s) * 100) * 5)
        result.set_value(i,'TesteSucesso',len(target_t[target_t == 0]) / len(target_t) * 100)
        result.set_value(i,'TesteInsucesso',len(target_t[target_t == 1]) / len(target_t) * 100)
        result.set_value(i,'TesteDesbalanceamento',(len(target_t[target_t == 1]) / len(target_t) * 100) / (len(target_t[target_t == 0]) / len(target_t) * 100) * 5)
        result.set_value(i,'GMean', gmean)
        
        i += 1
    
    #util.show_confusion_matrix(cm, class_labels=['Insucesso', 'Sucesso'])
    
    return result;
  

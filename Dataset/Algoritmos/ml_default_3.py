# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 05:24:03 2017

@author: Everton

Modelo preditivo padrão
    - Cross validation
    - Balanceamento de classes
"""
import pandas as pd
import numpy as np
import filter as filter
import util as util
import coral as coral
import probal as probal

import algoritmos as algoritmos
import graficos as graficos

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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#-------------------------------------------------------
# Configuração de filtros para o dataset
disciplina_s = 60465
disciplina_t = 60463
modulo_s = 6 #0 = ignora o módulo. Lembrando que só existem os módulos 3 e 6
modulo_t = 6 #0 = ignora o módulo. Lembrando que só existem os módulos 3 e 6
periodo_letivo_source = [20120101,20120102,20120201,20120202,20130101]
periodo_letivo_test   = [20130102,20130201,20130202,20140101,20140102]
"""
features = ['Assignment_View_Quantidade_Somado',
            'Turno_TempoUsoTotal_Somado',
            'Log_View_Quantidade_Somado',
            'Login_Quantidade',
            'Resource_View_Tempo_Somado',
            'Forum_TempoUso_Somado',
            'Evadido']
"""
features = ['Log_Post_Quantidade_Somado',
            'Login_Quantidade',
            'Assignment_View_Quantidade_Somado',
            'Turno_TempoUsoTotal_Somado',
            'Numero_Dias_Acessados_Modulo_Somado',
            'Log_View_Quantidade_Somado',
            'Evadido']
classificador = 2
use_coral = True
use_normalization = True
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

s_periodo = str(periodo_letivo_source)
t_periodo = str(periodo_letivo_test)

#Carrega dataset
df = pd.read_csv('../dataset_m3_m6.csv', sep=';')

print('CLASSIFICADOR: %s' % classificadores[classificador])

#Filtra o dataset conforme a configuração selecionada e faz alguns ajustes no dataset
df_s, df_t = filter.filter_dataset_mult(
                            df,
                            disciplinas, 
                            modulo_s, 
                            modulo_t, 
                            disciplina_s, 
                            disciplina_t, 
                            periodo_letivo_source, 
                            periodo_letivo_test,
                            features,
                            feature_normalization=use_normalization)

if use_coral == True:
    #df_s = coral.correlation_alignment(df_s, df_t, lambda_par=coral_lambda)
    df_s = probal.probability_alignment(df_s, df_t, lambda_par=coral_lambda)

df_s.to_csv('../Data Visualization/amostra_original.csv', index=False, sep=';')

model = None

if classificador == 1:
    model = GaussianNB()
if classificador == 2:
    #clf_param = {'max_depth': range(3,10)}
    #model = GridSearchCV(clf, clf_param)
    model = DecisionTreeClassifier(max_depth=3)
if classificador == 3:
    parameters = {'kernel':('linear', 'rbf'), 'C': range(1, 20), 'cache_size': [1, 500000]}
    clf = svm.SVC()
    model = GridSearchCV(clf, parameters)    

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)

i = 1

cm_final = np.matrix('0 0; 0 0')

#Cria um Dataframe sem a coluna Evadido para ser utilizada no CV
df_source = df_s[df_s.columns.difference(['Evadido'])]
target = df_s.Evadido

print('---- Folds -----')

for train, test in skf.split(df_source, target):

    fold_s = df_source.iloc[train]
    target_s = target.iloc[train]
    
    print('\tFold: %d: %d registros' % (i, len(fold_s)))
    print('\t\tSucesso: %d (%.2f%%) / Insucesso: %d (%.2f%%)' % 
          (len(target_s[target_s == 0]), 
           len(target_s[target_s == 0]) / len(target_s) * 100, 
           len(target_s[target_s == 1]), 
           len(target_s[target_s == 1]) / len(target_s) * 100))
    
    model.fit(fold_s, target_s)
    
    #Separa os dados de teste do atributo de predição
    fold_t   = df_source.iloc[test]
    target_t = target[test]
    
    predicted = model.predict(fold_t)
    
    cm_final += confusion_matrix(target_t, predicted);
    
    i += 1

#util.show_confusion_matrix(cm_final, class_labels=['Sucesso', 'Insucesso'])

#Separa os dados de teste do atributo de predição
features_test = df_t[df_t.columns.difference(['Evadido'])]
target_test = df_t.Evadido

predicted = model.predict(features_test)

cm_final = confusion_matrix(target_test, predicted);

util.show_confusion_matrix(cm_final, class_labels=['Sucesso', 'Insucesso'])

"""
---------------------------------------------------------------------------
fig = plt.figure()#(figsize=(16,16))
ax = fig.add_subplot(111, projection='3d')

x_s =df_s.Login_Quantidade
y_s =df_s.Log_Post_Quantidade_Somado
z_s =df_s.Turno_TempoUsoTotal_Somado

ax.scatter(x_s, y_s, z_s, c='r', marker='o')

x_t =df_t.Login_Quantidade
y_t =df_t.Log_Post_Quantidade_Somado
z_t =df_t.Turno_TempoUsoTotal_Somado

ax.scatter(x_t, y_t, z_t, c='b', marker='^')

ax.set_xlabel('f0')
ax.set_xlim(xmin=-10,xmax=10)
ax.set_ylabel('f1')
ax.set_ylim(ymin=-10,ymax=10)
ax.set_zlabel('f2')

plt.show()
"""
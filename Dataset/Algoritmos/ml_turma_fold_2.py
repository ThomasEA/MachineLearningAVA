# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 07:54:01 2017

@author: Everton

Modelo preditivo baseado em turma-fold
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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.style.use('seaborn-colorblind')
plt.rcParams['figure.figsize'] = (11,7)

#-------------------------------------------------------
# Configuração de filtros para o dataset
disciplina_s = 60500
disciplina_t = 60463
modulo_s = 6 #0 = ignora o módulo. Lembrando que só existem os módulos 3 e 6
modulo_t = 6 #0 = ignora o módulo. Lembrando que só existem os módulos 3 e 6
periodo_letivo_source = [20120101,20120102,20120201,20120202,20130101]
periodo_letivo_test   = [20130102,20130201,20130202,20140101,20140102]
features = {
        50404: ['Questionario_Quantidade_Somado', 'Forum_TempoUso_Somado', 'Log_Post_Quantidade_Somado', 'Questionario_TempoUso_Somado', 'Login_Quantidade','Turno_TempoUsoTotal_Somado', 'Evadido','CodigoTurma'],
        60463: ['Turno_TempoUsoTotal_Somado', 'Login_Quantidade', 'Log_Post_Quantidade_Somado', 'Questionario_TempoUso_Somado', 'Log_View_Quantidade_Somado', 'Numero_Dias_Acessados_Modulo_Somado', 'Evadido','CodigoTurma'],
        60465: ['Log_Post_Quantidade_Somado', 'Login_Quantidade', 'Assignment_View_Quantidade_Somado', 'Turno_TempoUsoTotal_Somado', 'Numero_Dias_Acessados_Modulo_Somado','Log_View_Quantidade_Somado', 'Evadido','CodigoTurma'],
        60500: ['Assignment_View_Quantidade_Somado', 'Turno_TempoUsoTotal_Somado', 'Log_View_Quantidade_Somado', 'Login_Quantidade', 'Resource_View_Tempo_Somado','Forum_TempoUso_Somado', 'Evadido','CodigoTurma']
    }
            
classificador = 2
use_coral = False
use_normalization = False
use_normalization_turma = True
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

s_periodo = str(periodo_letivo_source)
t_periodo = str(periodo_letivo_test)

#Carrega dataset
df = pd.read_csv('../dataset_m3_m6.csv', sep=';')

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

#df_s.to_csv('../Data Visualization/amostra_original.csv', index=False, sep=';')

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

i = 1

cm_final = np.matrix('0 0; 0 0')

print('---- Folds -----')

result = pd.DataFrame()

for name, fold in df_s_folds:
    fold_t = fold[features[disciplina_s]]
    
    if (shuf==True):
        fold_t = shuffle(fold_t)
        
    target_t = fold_t.Evadido
    fold_t = fold_t.loc[:, fold_t.columns != 'Evadido']
    fold_t = fold_t.loc[:, fold_t.columns != 'CodigoTurma']
    
    if (use_normalization_turma == True):
        scaler = preprocessing.StandardScaler().fit(fold_t)
        fold_t_norm = pd.DataFrame(scaler.transform(fold_t), columns = list(fold_t))
    else:
        fold_t_norm = fold_t

    fold_s = pd.DataFrame()
    
    #monta o dataset de treinamento
    for name_s, fold_stmp in df_s_folds:
        if (name_s != name):
            fold_s = pd.concat([fold_s,fold_stmp])
    
    fold_s = fold_s[features[disciplina_s]]
    
    if (shuf==True):
        fold_s = shuffle(fold_s)
        
    target_s = fold_s.Evadido
    fold_s = fold_s.loc[:, fold_s.columns != 'Evadido']
    fold_s = fold_s.loc[:, fold_s.columns != 'CodigoTurma']
    
    if (use_normalization_turma == True):
        scaler = preprocessing.StandardScaler().fit(fold_s)
        fold_s_norm = pd.DataFrame(scaler.transform(fold_s), columns = list(fold_s))
    else:
        fold_s_norm = fold_s
    
    print('\tFold: %d: [Treino %d registros]' % (i, len(fold_s)))
    print('\t\tSucesso: %d (%.2f%%) / Insucesso: %d (%.2f%%)' % 
          (len(target_s[target_s == 0]), 
           len(target_s[target_s == 0]) / len(target_s) * 100, 
           len(target_s[target_s == 1]), 
           len(target_s[target_s == 1]) / len(target_s) * 100))
    
    print('\tFold: %d: [Teste %d registros]' % (i, len(fold_t)))
    print('\t\tSucesso: %d (%.2f%%) / Insucesso: %d (%.2f%%)' % 
          (len(target_t[target_t == 0]), 
           len(target_t[target_t == 0]) / len(target_t) * 100, 
           len(target_t[target_t == 1]), 
           len(target_t[target_t == 1]) / len(target_t) * 100))
    
    model.fit(fold_s_norm, target_s)    

    predicted = model.predict(fold_t_norm)
    
    accuracy = accuracy_score(target_t, predicted)                                
    
    result.set_value(i,'Turma','T' + str(i))
    result.set_value(i,'Acur',accuracy * 100)
    result.set_value(i,'TreinoSucesso',len(target_s[target_s == 0]) / len(target_s) * 100)
    result.set_value(i,'TreinoInsucesso',len(target_s[target_s == 1]) / len(target_s) * 100)
    result.set_value(i,'TreinoDesbalanceamento',(len(target_s[target_s == 1]) / len(target_s) * 100) / (len(target_s[target_s == 0]) / len(target_s) * 100) * 5)
    result.set_value(i,'TesteSucesso',len(target_t[target_t == 0]) / len(target_t) * 100)
    result.set_value(i,'TesteInsucesso',len(target_t[target_t == 1]) / len(target_t) * 100)
    result.set_value(i,'TesteDesbalanceamento',(len(target_t[target_t == 1]) / len(target_t) * 100) / (len(target_t[target_t == 0]) / len(target_t) * 100) * 5)
    
    i += 1

#---- CORAL ----#

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

i = 1

for name, fold in df_s_folds:
    
    fold_t = fold[features[disciplina_s]]
    
    if (shuf==True):
        fold_t = shuffle(fold_t)
        
    target_t = fold_t.Evadido
    fold_t = fold_t.loc[:, fold_t.columns != 'Evadido']
    fold_t = fold_t.loc[:, fold_t.columns != 'CodigoTurma']
    
    #if (use_normalization_turma == True):
    #    scaler = preprocessing.StandardScaler().fit(fold_t)
    #    fold_t_norm = pd.DataFrame(scaler.transform(fold_t), columns = list(fold_t))
    #else:
    #    fold_t_norm = fold_t
    fold_t_norm = fold_t

    fold_s = pd.DataFrame()
    
    #monta o dataset de treinamento
    for name_s, fold_stmp in df_s_folds:
        if (name_s != name):
            fold_s = pd.concat([fold_s,fold_stmp])
    
    fold_s = fold_s[features[disciplina_s]]
    
    if (shuf==True):
        fold_s = shuffle(fold_s)
        
    target_s = fold_s.Evadido
    fold_s = fold_s.loc[:, fold_s.columns != 'Evadido']
    fold_s = fold_s.loc[:, fold_s.columns != 'CodigoTurma']
    
    #if (use_normalization_turma == True):
    #    scaler = preprocessing.StandardScaler().fit(fold_s)
    #    fold_s_norm = pd.DataFrame(scaler.transform(fold_s), columns = list(fold_s))
    #else:
    #    fold_s_norm = fold_s
    fold_s_norm = fold_s
    
    print('\tFold: %d: [Treino %d registros]' % (i, len(fold_s)))
    print('\t\tSucesso: %d (%.2f%%) / Insucesso: %d (%.2f%%)' % 
          (len(target_s[target_s == 0]), 
           len(target_s[target_s == 0]) / len(target_s) * 100, 
           len(target_s[target_s == 1]), 
           len(target_s[target_s == 1]) / len(target_s) * 100))
    
    print('\tFold: %d: [Teste %d registros]' % (i, len(fold_t)))
    print('\t\tSucesso: %d (%.2f%%) / Insucesso: %d (%.2f%%)' % 
          (len(target_t[target_t == 0]), 
           len(target_t[target_t == 0]) / len(target_t) * 100, 
           len(target_t[target_t == 1]), 
           len(target_t[target_t == 1]) / len(target_t) * 100))

    fold_s_norm = coral.correlation_alignment(fold_s_norm, fold_t_norm, lambda_par=coral_lambda)
    
    model.fit(fold_s_norm, target_s)    

    predicted = model.predict(fold_t_norm)
    
    accuracy = accuracy_score(target_t, predicted)                                
    
    result.set_value(i,'Coral',accuracy * 100)
    
    i += 1
    
#---- PLOT DO RESULTADO ----#

N = len(result)

ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig = plt.figure()                                                               
ax = fig.add_subplot(1,1,1)  

plt.title(disciplina_string)

plt.ylim(0,100)
plt.ylabel('Acurácia')

plt.xlabel('Turmas')

major_ticks = np.arange(0, 101, 10)                                              
minor_ticks = np.arange(0, 101, 2.5)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.set_xticks(ind + width / 2)
ax.set_xticklabels(result['Turma'])

#b1 = ax.bar(ind, result['TreinoSucesso'], width)
#b2 = ax.bar(ind + width, result['TreinoInsucesso'], width)
#b3 = ax.bar(ind + (width*2), result['TesteSucesso'], width)
#b4 = ax.bar(ind + (width*3), result['TesteInsucesso'], width)

l1 = ax.plot(ind + (width*3) / 2, result['Acur'], '#0033cc', marker='X')
l2 = ax.plot(ind + (width*3) / 2, result['Coral'], '#f44242', marker='X')

l3 = ax.plot(ind + (width*3) / 2, result['TreinoDesbalanceamento'], '#ff6600', marker='D')
l4 = ax.plot(ind + (width*3) / 2, result['TesteDesbalanceamento'], '#009933', marker='D')

#ax.legend((b1[0], b2[0], b3[0], b4[0], l1[0], l2[0]),
#          ('Treino Sucesso %', 'Treino Insucesso %', 'Teste Sucesso %', 'Teste Insucesso %', 'Acur. Original %', 'Acur. CORAL %'), loc=2, bbox_to_anchor=(1.05, 1))

ax.legend((l3[0], l4[0], l1[0], l2[0]),
          ('Treino Desbal. %', 'Teste Desbal. %', 'Acur. Original %', 'Acur. CORAL %'), 
          loc=4, bbox_to_anchor=(1.05, 1))

plt.xticks(ind + (width*3) / 2)

ax.grid(which='both')                                                            

# or if you want differnet settings for the grids:                               
ax.grid(which='minor', alpha=0.4)                                                
ax.grid(which='major', alpha=0.5)

plt.show()

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
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
disciplina_s = 60465
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

result = pd.DataFrame(columns=["Turma", "Acur", "Coral", "Sucesso", "Insucesso"])

for name, fold in df_s_folds:
    fold_t = fold[features[disciplina_s]]
    #fold_t = shuffle(fold_t)
    target_t = fold_t.Evadido
    fold_t = fold_t.loc[:, fold_t.columns != 'Evadido']
    fold_t = fold_t.loc[:, fold_t.columns != 'CodigoTurma']
    
    fold_s = pd.DataFrame()
    
    #monta o dataset de treinamento
    for name_s, fold_stmp in df_s_folds:
        if (name_s != name):
            fold_s = pd.concat([fold_s,fold_stmp])
    
    fold_s = fold_s[features[disciplina_s]]
    #fold_s = shuffle(fold_s)
    target_s = fold_s.Evadido
    fold_s = fold_s.loc[:, fold_s.columns != 'Evadido']
    fold_s = fold_s.loc[:, fold_s.columns != 'CodigoTurma']
    
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
    
    model.fit(fold_s, target_s)    

    predicted = model.predict(fold_t)
    
    cm_final += confusion_matrix(target_t, predicted);
    
    accuracy = accuracy_score(target_t, predicted)                                
    
    result.set_value(i,'Turma','T' + str(i))
    result.set_value(i,'Acur',accuracy * 100)
    result.set_value(i,'Sucesso',len(target_t[target_t == 0]) / len(target_t) * 100)
    result.set_value(i,'Insucesso',len(target_t[target_t == 1]) / len(target_t) * 100)
    
    i += 1

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
    #fold_t = shuffle(fold_t)
    target_t = fold_t.Evadido
    fold_t = fold_t.loc[:, fold_t.columns != 'Evadido']
    fold_t = fold_t.loc[:, fold_t.columns != 'CodigoTurma']
    
    fold_s = pd.DataFrame()
    
    #monta o dataset de treinamento
    for name_s, fold_stmp in df_s_folds:
        if (name_s != name):
            fold_s = pd.concat([fold_s,fold_stmp])
    
    fold_s = fold_s[features[disciplina_s]]
    #fold_s = shuffle(fold_s)
    target_s = fold_s.Evadido
    fold_s = fold_s.loc[:, fold_s.columns != 'Evadido']
    fold_s = fold_s.loc[:, fold_s.columns != 'CodigoTurma']
    
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
    
    fold_s = coral.correlation_alignment(fold_s, fold_t, lambda_par=coral_lambda)
    
    model.fit(fold_s, target_s)    

    predicted = model.predict(fold_t)
    
    cm_final += confusion_matrix(target_t, predicted);
    
    accuracy = accuracy_score(target_t, predicted)                                
    
    result.set_value(i,'Coral',accuracy * 100)
    
    i += 1


#util.show_confusion_matrix(cm_final, class_labels=['Sucesso', 'Insucesso'])

#Separa os dados de teste do atributo de predição
#features_test = df_t[df_t.columns.difference(['Evadido'])]
#target_test = df_t.Evadido

#predicted = model.predict(features_test)

#cm_final = confusion_matrix(target_test, predicted);

#util.show_confusion_matrix(cm_final, class_labels=['Sucesso', 'Insucesso'])

N = len(result)

ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig,ax = plt.subplots()

plt.title(disciplina_string)

plt.ylim(0,100)
plt.ylabel('Acurácia')

plt.xlabel('Turmas')

ax.set_xticks(ind + width / 2)
ax.set_xticklabels(result['Turma'])

b1 = ax.bar(ind, result['Sucesso'], width)
b2 = ax.bar(ind + width, result['Insucesso'], width)

l1 = ax.plot(ind + width / 2, result['Acur'], 'y')
l2 = ax.plot(ind + width / 2, result['Coral'], 'r:')

ax.legend((b1[0], b2[0], l1[0], l2[0]),
          ('Sucesso %', 'Insucesso %', 'Acur. Original %', 'Acur. CORAL %'), loc=2, bbox_to_anchor=(1.05, 1))

plt.xticks(ind + width / 2)

#for i,j in zip(ind - width,result['Acur']):
#    ax.annotate(str('%.2f' % (j)),xy=(i,j))

#for i,j in zip(ind + width,result['Coral']):
#    ax.annotate(str('%.2f' % (j)),xy=(i,j))

#result.plot()
#result.plot(result['Turma'], kind='bar')
#plt.plot(result['Acur'])
#plt.plot(result['Coral'])
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
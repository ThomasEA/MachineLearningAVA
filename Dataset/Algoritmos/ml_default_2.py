# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 00:58:56 2017

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

import algoritmos as algoritmos
import graficos as graficos

from sklearn.model_selection import StratifiedKFold
#Importando e configurando classificador (DecisionTree)
from sklearn.tree import DecisionTreeClassifier
#Importando gerador de parametros otimizados
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#-------------------------------------------------------
# Configuração de filtros para o dataset
disciplina = 60465
modulo = 6 #0 = ignora o módulo. Lembrando que só existem os módulos 3 e 6
periodo_letivo_source = [20120101,20120102,20120201,20120202,20130101]
periodo_letivo_test   = [20130102,20130201,20130202,20140101,20140102]
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

#Filtra o dataset conforme a configuração selecionada e faz alguns ajustes no dataset
df_s, df_t = filter.filter_dataset(
                            df, 
                            modulo, 
                            disciplinas, 
                            disciplina, 
                            periodo_letivo_source, 
                            periodo_letivo_test,
                            feature_normalization=True)

#df_s = coral.correlation_alignment(df_s, df_t, lambda_par=1)

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
clf_param = {'max_depth': range(3,10)}
clf = DecisionTreeClassifier()

model = GridSearchCV(clf, clf_param)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)

i = 1

cm_final = np.matrix('0 0; 0 0')

#Cria um Dataframe sem a coluna Evadido para ser utilizada no CV
df_source = df_s[df_s.columns.difference(['Evadido'])]
target = df_s.Evadido

for train, test in skf.split(df_source, target):

    fold_s = df_source.iloc[train]
    target_s = target.iloc[train]
    
    print('Fold: %d' % i)
    print('\tQtd. Registros: %d' % len(fold_s))
    print('\tSucesso.......: %d / %.2f%%' % (len(target_s[target_s == 0]), len(target_s[target_s == 0]) / len(target_s) * 100))
    print('\tInsucesso.....: %d / %.2f%%' % (len(target_s[target_s == 1]), len(target_s[target_s == 1]) / len(target_s) * 100))
    
    model.fit(fold_s, target_s)
    
    #Separa os dados de teste do atributo de predição
    fold_t   = df_source.iloc[test]
    target_t = target[test]
    
    predicted = model.predict(fold_t)
    
    cm_final += confusion_matrix(target_t, predicted);
    
    i += 1

util.show_confusion_matrix(cm_final, class_labels=['Sucesso', 'Insucesso'])

#Separa os dados de teste do atributo de predição
features_test = df_t[df_t.columns.difference(['Evadido'])]
target_test = df_t.Evadido

predicted = model.predict(features_test)

cm_final = confusion_matrix(target_test, predicted);

util.show_confusion_matrix(cm_final, class_labels=['Sucesso', 'Insucesso'])
"""
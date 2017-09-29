# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:06:48 2017

@author: Everton
@purpose: Script inicial para familiarização com python
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

train_df = pd.read_csv("m3_rac_logico_ext.csv", sep=';')

#print(train_df.describe())
#print(train_df.info())

#Mostra os valores distintos para a coluna Evadido
#print(train_df.Evadido.unique())

#Altera os valores da coluna Evadido para binário
train_df.Evadido = train_df.Evadido.map({'ReprEvadiu': 0, 'Sucesso': 1})

#Calcular z-Score para algumas features
z_score_features = ['Assignment_View_TempoUso_Somado',
                    'Chat_TempoUso_Somado',
                    'Forum_TempoUso_Somado',
                    'Questionario_TempoUso_Somado',
                    'Resource_View_Tempo_Somado',
                    'Turno_PercentualUsoMadrugada_Somado',
                    'Turno_PercentualUsoManha_Somado',
                    'Turno_PercentualUsoNoite_Somado',
                    'Turno_PercentualUsoTarde_Somado',
                    'Turno_TempoUsoMadrugada_Somado',
                    'Turno_TempoUsoManha_Somado',
                    'Turno_TempoUsoNoite_Somado',
                    'Turno_TempoUsoTarde_Somado',
                    'Turno_TempoUsoTotal_Somado',]
for col in z_score_features:
    train_df[col] = (train_df[col] - train_df[col].mean())/train_df[col].std(ddof=0)


#Verifica se existe alguma coluna com valor null
#print(train_df.isnull().any())

#separa em turmas
"""
turmas = train_df.groupby('CodigoTurma')
for key in turmas.groups.keys():
    print(key)
    print(turmas.get_group(name=key).info())
"""
#Importando classificador
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

clf_param = {'max_depth': range(3,10)}
clf = DecisionTreeClassifier()

dt = GridSearchCV(clf, clf_param)

print('*****************************************************')
print('Features in dataframe:')
print(train_df.info())
print('*****************************************************')

#Embaralha dataset
train_df = shuffle(train_df)

#Seleciona todas as colunas, exceto a coluna Evadido e outras que são irrelevantes
features = train_df[train_df.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
target = train_df.Evadido

print('*****************************************************')
print('Features used to predict:')
print(features.info())
print('*****************************************************')

#treina
dt.fit(features, target)

print('Score:')
print(dt.score(X = features, y = target))
                      
#features_test = test_df.loc[:, test_df.columns != 'Evadido']

#clf.predict(features_test)

#print(clf.score(X = features_test, y = target))



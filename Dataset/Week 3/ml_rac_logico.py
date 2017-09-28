# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:06:48 2017

@author: Everton
@purpose: Script inicial para familiarização com python
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

train_df = pd.read_csv("W3_Rac_logico.CSV", sep=';')
test_df = pd.read_csv("Teste_W6_Rac_Logico.CSV", sep=';')

#print(train_df.describe())
#print(train_df.info())

#Mostra os valores distintos para a coluna Evadido
#print(train_df.Evadido.unique())

#Altera os valores da coluna Evadido para binário
train_df.Evadido = train_df.Evadido.map({'ReprEvadiu': 0, 'Sucesso': 1})
test_df.Evadido = test_df.Evadido.map({'ReprEvadiu': 0, 'Sucesso': 1})

#Mostra os valores alterados
#print(train_df.Evadido.unique())

#Verifica se existe alguma coluna com valor null
#print(train_df.isnull().any())

#Importando classificador
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

#Seleciona todas as colunas, exceto a coluna Evadido
features = train_df.loc[:, train_df.columns != 'Evadido']
target = train_df.Evadido

#treina
clf.fit(features, target)

print(clf.score(X = features, y = target))
                      
features_test = test_df.loc[:, test_df.columns != 'Evadido']

clf.predict(features_test)

print(clf.score(X = features_test, y = target))



# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 15:32:22 2017

@author: Everton
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:06:48 2017

@author: Everton
@purpose: Script inicial para familiarização com python
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import util as util

import matplotlib



from sklearn import preprocessing

matplotlib.style.use('ggplot')

#train_df = pd.read_csv("m3_rac_logico_ext.CSV", sep=';')
train_df = pd.read_csv("m3_mat_adm.CSV", sep=';')

#Altera os valores da coluna Evadido para binário
train_df.Evadido = train_df.Evadido.map({'ReprEvadiu': 0, 'Sucesso': 1})

features = train_df[['Evadido']]

util.plot_dist('Distribuição', features)



#print(train_df.describe())
#print(train_df.info())

#Mostra os valores distintos para a coluna Evadido
#print(train_df.Evadido.unique())


#Verifica se existe alguma coluna com valor null
#print(train_df.isnull().any())

#separa em turmas
"""
turmas = train_df.groupby('CodigoTurma')
for key in turmas.groups.keys():
    print(key)
    print(turmas.get_group(name=key).info())
"""

"""
#print("np.nan=", np.where(np.isnan(train_df)))
#print("np.inf=", np.where(np.isinf(train_df)))

#Importando classificador
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

clf_param = {'max_depth': range(3,10)}
clf = DecisionTreeClassifier()

dt = GridSearchCV(clf, clf_param)

print('*****************************************************')
print('Features in dataframe:')
#print(train_df.info())
print('*****************************************************')

#Calcular z-Score para algumas features
features = train_df[train_df.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
features_target = train_df.Evadido

scaler = preprocessing.StandardScaler().fit(features)

train_df_std = pd.DataFrame(scaler.transform(features), columns = list(features))

#features_zscore = train_df[train_df.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
#train_df_std = pd.DataFrame(scaler.transform(features_zscore), columns = list(features_zscore))

#.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido']) #list(features)

#pd.DataFrame(scaler.transform(train_df), columns = features_zscore, copy = false)
"""


"""
#Embaralha dataset
#train_df = shuffle(train_df)

#Seleciona todas as colunas, exceto a coluna Evadido e outras que são irrelevantes
features = train_df[train_df.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]


#features_columns.remove('Assignment_Post_Quantidade_Somado')
print(features_columns)
features_normalized = pd.DataFrame(scaler.transform(features), columns = features_columns)

print('---- APOS CALCULO zScore ---------')
print(features_normalized.head(5))
print('-----------------------------------')


target = train_df.Evadido

print('*****************************************************')
print('Features used to predict:')
#print(features_normalized.info())
print('*****************************************************')

#treina
dt.fit(features_normalized, target)

print('Score:')
print(dt.score(X = features_normalized, y = target))
                      
#features_test = test_df.loc[:, test_df.columns != 'Evadido']

#clf.predict(features_test)

#print(clf.score(X = features_test, y = target))
"""


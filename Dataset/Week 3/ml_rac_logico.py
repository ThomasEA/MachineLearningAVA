# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:06:48 2017

@author: Everton
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import util as util

import graficos as graficos

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn import preprocessing
from sklearn.utils import shuffle

normalizar = False
plot_var_cov = True

s_disciplina = 'rac_logico'
t_disciplina = 'mat_adm'

df = pd.read_csv('m3_' + s_disciplina + '_ext.CSV', sep=';')

#Altera os valores da coluna Evadido para binário
df.Evadido = df.Evadido.map({'ReprEvadiu': 0, 'Sucesso': 1})

#Calcular z-Score para algumas features
#--------------------------------------
if (normalizar==True):
    features = df[df.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
    scaler = preprocessing.StandardScaler().fit(features)
    df_std = pd.DataFrame(scaler.transform(features), columns = list(features))
    
    df_std['Evadido'] = df.Evadido
    df_std['CodigoDisciplina'] = df.CodigoDisciplina
    df_std['CodigoTurma'] = df.CodigoTurma
    df_std['PeriodoLetivo'] = df.PeriodoLetivo
else:
    df_std = df
#--------------------------------------

if (plot_var_cov == True):
    #graficos.plot_corr_matrix(df_std.corr(), 'Correlação [' + s_disciplina + ']')
    graficos.plot_corr_matrix(df_std.cov(), 'Covariância [' + s_disciplina + ']')
    
    #Cria matriz identidade
    ID_S = np.eye(len(list(df_std)))
    C_S = df_std.corr() + ID_S
    graficos.plot_corr_matrix(C_S, 'Cs [' + s_disciplina + ']')
    
    #df_std = df_std * (C_S**(-1/2))
    #df_std = df_std * (C_S**4)#np.sqrt(C_T)
    #print(df_std.describe())
    print(C_S.describe())
    graficos.plot_corr_matrix(df_std.cov(), 'Covariância APÓS AJUSTE[' + s_disciplina + ']')

#Embaralha dataframe normalizado
df_normalized = shuffle(df_std)

#Importando e configurando classificador (DecisionTree)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

clf_param = {'max_depth': range(3,10)}
clf = DecisionTreeClassifier()

dt = GridSearchCV(clf, clf_param)

#Separa os folds por turma
folds = util.sep_folds(df_normalized, 'CodigoTurma')

qtd_folds = len(folds.groups)

#Cross-validation
cm_final = np.matrix('0 0; 0 0')

print('====================')
print('Coss-validation: (k = ' + str(qtd_folds) + ')')
for key in folds.groups.keys():
    fold_teste = folds.get_group(name=key)
    fold_treino = folds.filter(lambda x: x.name!=key)
    
    qtd_ex_teste = len(fold_teste)
    qtd_ex_treino = len(fold_treino)
    
    print('\tRegistros: ' + str(qtd_ex_teste + qtd_ex_treino) + ' / Treino: ' + str(qtd_ex_treino) + ' / Teste: ' + str(qtd_ex_teste))
 
    #Separa os dados de treino do atributo de predição
    features = fold_treino[fold_treino.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
    target = fold_treino.Evadido
    
    dt.fit(features, target)
    
    #Separa os dados de teste do atributo de predição
    features_test = fold_teste[fold_teste.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
    target_test = fold_teste.Evadido
    
    predicted = dt.predict(features_test)
    
    cm = confusion_matrix(target_test, predicted)
    
    cm_final = cm_final + cm

#Plota a matriz de confusão para o modelo
util.show_confusion_matrix(cm_final, class_labels=['Insucesso', 'Sucesso'])

#-----------------------------------------
#Aplica classificador em outra disciplina

df_target = pd.read_csv('m3_' + t_disciplina + '_ext.CSV', sep=';')

#Altera os valores da coluna Evadido para binário
df_target.Evadido = df_target.Evadido.map({'ReprEvadiu': 0, 'Sucesso': 1})

#Calcular z-Score para algumas features
#--------------------------------------
if (normalizar==True):
    features_target = df_target[df_target.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
    scaler = preprocessing.StandardScaler().fit(features_target)
    df_std_target = pd.DataFrame(scaler.transform(features_target), columns = list(features_target))
    
    df_std_target['Evadido'] = df_target.Evadido
    df_std_target['CodigoDisciplina'] = df_target.CodigoDisciplina
    df_std_target['CodigoTurma'] = df_target.CodigoTurma
    df_std_target['PeriodoLetivo'] = df_target.PeriodoLetivo
else:
    df_std_target = df_target
#--------------------------------------

if (plot_var_cov == True):
    graficos.plot_corr_matrix(df_std_target.corr(), 'Correlação [' + t_disciplina + ']')
    graficos.plot_corr_matrix(df_std_target.cov(), 'Covariância [' + t_disciplina + ']')

#Embaralha dataframe normalizado
df_normalized_target = shuffle(df_std_target)

#Cross-validation
cm_final_target = np.matrix('0 0; 0 0')

#Separa os dados de teste do atributo de predição
features_test_target = df_normalized_target[df_normalized_target.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
target_test_target = df_normalized_target.Evadido
    
predicted_target = dt.predict(features_test_target)
    
cm_final_target = confusion_matrix(target_test_target, predicted_target)
    
#Plota a matriz de confusão para o modelo
util.show_confusion_matrix(cm_final_target, class_labels=['Insucesso', 'Sucesso'])
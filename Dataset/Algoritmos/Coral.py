# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 05:58:49 2017

@author: Everton
@purpose: Teste de aplicação do método CORAL entre duas disciplinas
    Cs = cov(Ds) + eye(size(Ds,2))          
    Ct = cov(Dt) + eye(size(Dt,2))
    Ds = Ds * Cs^(-1/2)
    Ds = Ds * Ct^(1/2)
"""

# -*- coding: utf-8 -*-
from scipy import *

import numpy as np
import pandas as pd
import util as util

import graficos as graficos

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn import preprocessing
from sklearn.utils import shuffle

normalizar = True
plot_var_cov = True

modulo = '3'
s_disciplina = 'rac_logico'
#s_disciplina = 'mat_adm'

df_s_r = pd.read_csv('../Week 3/m' + modulo + '_' + s_disciplina + '_ext_reduzido.CSV', sep=';')
df_s = pd.read_csv('../Week 3/m' + modulo + '_' + s_disciplina + '_ext.CSV', sep=';')

#Limpa e organiza algumas features e normaliza com z-score
df_s_reduzido = util.clean_data(df_s_r, normalizar)
df_s_std = util.clean_data(df_s, normalizar)

df_s_reduzido = util.correlation_alignment(df_s_reduzido, df_s_std,1)

"""
#Embaralha dataframe normalizado
df_normalized = shuffle(df_s_std)

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
    fold_teste = folds.get_group(name=key).copy()
    fold_treino = folds.filter(lambda x: x.name!=key).copy()
    
    fold_treino = util.correlation_alignment(fold_treino, fold_teste,1)
    
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
"""

"""
#Cria matriz identidade
    ID_S = np.eye(len(list(df_std)))
    C_S = df_std.cov() + ID_S
    #graficos.plot_corr_matrix(C_S, 'Cs [' + s_disciplina + ']')
    
    df_std = df_std * (C_S**(-1/2))
    df_std = df_std * (C_S**(1/2))
    #print(df_std.describe())
    print(C_S.describe())
    graficos.plot_corr_matrix(df_std.cov(), 'Covariância APÓS AJUSTE[' + s_disciplina + ']')
"""
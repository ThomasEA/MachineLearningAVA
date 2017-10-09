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
s_disciplina = 'logica'
#s_disciplina = 'mat_adm'

df_s = pd.read_csv('../Week 3/m' + modulo + '_' + s_disciplina + '_ext_2012_01.csv', sep=',')
df_t = pd.read_csv('../Week 3/m' + modulo + '_' + s_disciplina + '_ext_2012_02_2014_01.csv', sep=',')

#Limpa e organiza algumas features e normaliza com z-score
df_s_std = util.clean_data(df_s, normalizar, plot_cov=False, title="Clean Data - Covariancia (Ds)")
df_t_std = util.clean_data(df_t, normalizar, plot_cov=False, title="Clean Data - Covariancia (Dt)")

df_s_std = util.correlation_alignment(df_s_std, df_t_std,1)

graficos.plot_cov_matrix(df_s_std,'Ds apos adaptaçao')

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
    
    #fold_treino = util.correlation_alignment(fold_treino, fold_teste,1)
    
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

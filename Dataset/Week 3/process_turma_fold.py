# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 00:47:54 2017

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

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def process(df_normalized):

    print(df_normalized['CodigoDisciplina'])
    
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
        features = fold_treino[fold_treino.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido','StatusMatricula'])]
        target = fold_treino.Evadido
        
        dt.fit(features, target)
        
        #Separa os dados de teste do atributo de predição
        features_test = fold_teste[fold_teste.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido','StatusMatricula'])]
        target_test = fold_teste.Evadido
        
        predicted = dt.predict(features_test)
        
        cm = confusion_matrix(target_test, predicted)
        
        cm_final = cm_final + cm
    
    #Plota a matriz de confusão para o modelo
    util.show_confusion_matrix(cm_final, class_labels=['Insucesso', 'Sucesso'])

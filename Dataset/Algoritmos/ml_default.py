# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 22:57:58 2017

@author: Everton

Modelo preditivo padrão
    - Cross validation
    - Balanceamento de classes
"""
import pandas as pd
import numpy as np
import filter as filter
import util as util
import algoritmos as algoritmos
import graficos as graficos

from sklearn.model_selection import StratifiedKFold
#Importando e configurando classificador (DecisionTree)
from sklearn.tree import DecisionTreeClassifier
#Importando gerador de parametros otimizados
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#-------------------------------------------------------
# Configuração de filtros para o dataset
disciplina = 60500
modulo = 3 #0 = ignora o módulo. Lembrando que só existem os módulos 3 e 6
periodo_letivo_source = [20120101,20120102,20120201,20120202]
periodo_letivo_test   = [20130101,20130102,20130201,20130202,20140101,20140102]
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

#Filtra o dataset conforme a configuração selecionada e faz alguns ajustes no
#dataset
df_s, df_t = filter.filter_dataset(
                            df, 
                            modulo, 
                            disciplinas, 
                            disciplina, 
                            periodo_letivo_source, 
                            periodo_letivo_test)

clf_param = {'max_depth': range(3,10)}
clf = DecisionTreeClassifier()

model = GridSearchCV(clf, clf_param)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)

i = 1

cm_final = np.matrix('0 0; 0 0')

for train, test in skf.split(df_s, df_s.Evadido):
    
    fold_s = df_s.iloc[train]
    target_s = fold_s.Evadido
    
    print('Fold: %d ' % i)
    print('\tQtd. Registros: %d' % len(fold_s))
    print('\tSucesso.......: %d / %.2f%%' % (len(fold_s[fold_s.Evadido == 1]), len(fold_s[fold_s.Evadido == 1]) / len(fold_s) * 100))
    print('\tInsucesso.....: %d / %.2f%%' % (len(fold_s[fold_s.Evadido == 0]), len(fold_s[fold_s.Evadido == 0]) / len(fold_s) * 100))
    
    model.fit(fold_s, target_s)
    
    #Separa os dados de teste do atributo de predição
    fold_t = df_s.iloc[test]
    target_t = fold_t.Evadido
    
    predicted = model.predict(fold_t)
    
    cm_final += confusion_matrix(target_t, predicted);
    
    i += 1

util.show_confusion_matrix(cm_final, class_labels=['Insucesso', 'Sucesso'])
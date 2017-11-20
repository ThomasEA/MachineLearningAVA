# -*- coding: utf-8 -*-
"""
Created on Sat Nov  11 05:02:23

@author: Everton

Modelo preditivo baseado em turma-fold
    - Cross validation
    - Balanceamento de classes
"""
import pandas as pd
import numpy as np
import ml_por_disciplina as mldisc

from sklearn.utils import shuffle
from sklearn import preprocessing
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

def sumarizar(i, disciplina, classificador, coral, df, result):
    result['Turma'] = df['Turma']
    #result.set_value(i,'Turma', df['Turma'])
    if (coral):
        result[classificador + ' [Coral]'] = df['Acur']
    else:
        result[classificador] = df['Acur']
    #result.set_value(i,classificador, df['Acur'].mean())
    #result.set_value(i,classificador + 'DP', df['Acur'].std(ddof=1))

plt.style.use('seaborn-colorblind')
plt.rcParams['figure.figsize'] = (11,7)

use_normalization = False
use_normalization_turma = True

#-------------------------------------------------------
# Configuração de filtros para o dataset
disciplina_s = 50404
modulo_s = 6 #0 = ignora o módulo. Lembrando que só existem os módulos 3 e 6

features = {
        50404: ['Questionario_Quantidade_Somado', 'Forum_TempoUso_Somado', 'Log_Post_Quantidade_Somado', 'Questionario_TempoUso_Somado', 'Login_Quantidade','Turno_TempoUsoTotal_Somado', 'Evadido','CodigoTurma'],
        60463: ['Turno_TempoUsoTotal_Somado', 'Login_Quantidade', 'Log_Post_Quantidade_Somado', 'Questionario_TempoUso_Somado', 'Log_View_Quantidade_Somado', 'Numero_Dias_Acessados_Modulo_Somado', 'Evadido','CodigoTurma'],
        60465: ['Log_Post_Quantidade_Somado', 'Login_Quantidade', 'Assignment_View_Quantidade_Somado', 'Turno_TempoUsoTotal_Somado', 'Numero_Dias_Acessados_Modulo_Somado','Log_View_Quantidade_Somado', 'Evadido','CodigoTurma'],
        60500: ['Assignment_View_Quantidade_Somado', 'Turno_TempoUsoTotal_Somado', 'Log_View_Quantidade_Somado', 'Login_Quantidade', 'Resource_View_Tempo_Somado','Forum_TempoUso_Somado', 'Evadido','CodigoTurma']
    }
            
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

result = pd.DataFrame()

use_coral = False

#-----------------------------------------
disciplina_string = str(disciplinas[disciplina_s])
d1 = mldisc.process(disciplina_s, modulo_s, 1, use_coral, use_normalization, use_normalization_turma)
sumarizar(1, disciplina_string, classificadores[1], use_coral, d1, result)
d2 = mldisc.process(disciplina_s, modulo_s, 2, use_coral, use_normalization, use_normalization_turma)
sumarizar(1, disciplina_string, classificadores[2], use_coral, d2, result)
d3 = mldisc.process(disciplina_s, modulo_s, 3, use_coral, use_normalization, use_normalization_turma)
sumarizar(1, disciplina_string, classificadores[3], use_coral, d3, result)
#-----------------------------------------

use_coral = True

#-----------------------------------------
disciplina_string = str(disciplinas[disciplina_s])
d1 = mldisc.process(disciplina_s, modulo_s, 1, use_coral, use_normalization, use_normalization_turma)
sumarizar(2, disciplina_string, classificadores[1], use_coral, d1, result)
d2 = mldisc.process(disciplina_s, modulo_s, 2, use_coral, use_normalization, use_normalization_turma)
sumarizar(2, disciplina_string, classificadores[2], use_coral, d2, result)
d3 = mldisc.process(disciplina_s, modulo_s, 3, use_coral, use_normalization, use_normalization_turma)
sumarizar(2, disciplina_string, classificadores[3], use_coral, d3, result)
#-----------------------------------------

result = result.reset_index()

#frames = [d1,d2,d3]
#result = pd.concat(frames)

N = len(result)

ind = np.arange(N)  # the x locations for the groups
width = 0.20       # the width of the bars
ymin=40

fig = plt.figure()                                                               
ax = fig.add_subplot(1,1,1)  

if (use_normalization == True):
    plt.title(disciplina_string + ' - Norm. toda variável')
elif (use_normalization_turma == True):
    plt.title(disciplina_string + ' - Norm. por turma')
else:
    plt.title(disciplina_string + ' - Sem normalização')

plt.ylabel('Acurácia')

plt.xlabel('Turmas')

major_ticks = np.arange(0, 101, 10)                                              
minor_ticks = np.arange(0, 101, 2.5)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

plt.xticks(ind + width / 2)
ax.set_xticklabels(result['Turma'])

l1 = ax.plot(ind, result['Naive Bayes'], '#4286f4', marker='D')
l2 = ax.plot(ind, result['Decision Tree'], '#ea2300', marker='X')
l3 = ax.plot(ind, result['SVM'], '#00ba06', marker='o')

l4 = ax.plot(ind, result['Naive Bayes [Coral]'], '#ffa100', marker='D')
l5 = ax.plot(ind, result['Decision Tree [Coral]'], '#c300ff', marker='X')
l6 = ax.plot(ind, result['SVM [Coral]'], '#050505', marker='o')

ax.set_ylim(ymin=ymin, ymax=100)

ax.legend((l1[0], l2[0], l3[0], l4[0], l5[0], l6[0]),
          ('Naive Bayes', 'Decision Tree', 'SVM', 'Naive Bayes [Coral]', 'Decision Tree [Coral]', 'SVM [Coral]'),bbox_to_anchor=(0.5,-0.08), loc='upper center', ncol=6)

#ax.legend((l1[0], l2[0], l3[0]),
#          ('Naive Bayes', 'Decision Tree', 'SVM'),bbox_to_anchor=(0.5,-0.08), loc='upper center', ncol=6)

ax.yaxis.grid(which="major", color='#000000', linestyle=':', linewidth=0.5)

ax.yaxis.grid(True)

plt.show()

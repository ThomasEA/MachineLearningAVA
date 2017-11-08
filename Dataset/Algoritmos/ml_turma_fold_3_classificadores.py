# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 07:54:01 2017

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

def sumarizar(i, disciplina, classificador, df, result):
    result.set_value(i,'Disciplina', disciplina)
    result.set_value(i,classificador, df['Acur'].mean())
    result.set_value(i,classificador + 'DP', df['Acur'].std(ddof=1))

plt.style.use('seaborn-colorblind')
plt.rcParams['figure.figsize'] = (11,7)

use_normalization = False
use_normalization_turma = True
use_coral = True

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

#-----------------------------------------
disciplina_s = 50404
disciplina_string = str(disciplinas[disciplina_s])
d1 = mldisc.process(disciplina_s, modulo_s, 1, use_coral, use_normalization, use_normalization_turma)
sumarizar(1, disciplina_string, classificadores[1], d1, result)
d2 = mldisc.process(disciplina_s, modulo_s, 2, use_coral, use_normalization, use_normalization_turma)
sumarizar(1, disciplina_string, classificadores[2], d2, result)
d3 = mldisc.process(disciplina_s, modulo_s, 3, use_coral, use_normalization, use_normalization_turma)
sumarizar(1, disciplina_string, classificadores[3], d3, result)
#-----------------------------------------
"""
#-----------------------------------------
disciplina_s = 60463
disciplina_string = str(disciplinas[disciplina_s])
d1 = mldisc.process(disciplina_s, modulo_s, 1, use_coral, use_normalization, use_normalization_turma)
sumarizar(2, disciplina_string, classificadores[1], d1, result)
d2 = mldisc.process(disciplina_s, modulo_s, 2, use_coral, use_normalization, use_normalization_turma)
sumarizar(2, disciplina_string, classificadores[2], d2, result)
d3 = mldisc.process(disciplina_s, modulo_s, 3, use_coral, use_normalization, use_normalization_turma)
sumarizar(2, disciplina_string, classificadores[3], d3, result)
#-----------------------------------------

#-----------------------------------------
disciplina_s = 60465
disciplina_string = str(disciplinas[disciplina_s])
d1 = mldisc.process(disciplina_s, modulo_s, 1, use_coral, use_normalization, use_normalization_turma)
sumarizar(3, disciplina_string, classificadores[1], d1, result)
d2 = mldisc.process(disciplina_s, modulo_s, 2, use_coral, use_normalization, use_normalization_turma)
sumarizar(3, disciplina_string, classificadores[2], d2, result)
d3 = mldisc.process(disciplina_s, modulo_s, 3, use_coral, use_normalization, use_normalization_turma)
sumarizar(3, disciplina_string, classificadores[3], d3, result)
#-----------------------------------------
"""
#-----------------------------------------
disciplina_s = 60500
disciplina_string = str(disciplinas[disciplina_s])
d1 = mldisc.process(disciplina_s, modulo_s, 1, use_coral, use_normalization, use_normalization_turma)
sumarizar(4, disciplina_string, classificadores[1], d1, result)
d2 = mldisc.process(disciplina_s, modulo_s, 2, use_coral, use_normalization, use_normalization_turma)
sumarizar(4, disciplina_string, classificadores[2], d2, result)
d3 = mldisc.process(disciplina_s, modulo_s, 3, use_coral, use_normalization, use_normalization_turma)
sumarizar(4, disciplina_string, classificadores[3], d3, result)
#-----------------------------------------

result = result.reset_index()

N = len(result)

ind = np.arange(N)  # the x locations for the groups
width = 0.20       # the width of the bars

fig = plt.figure()                                                               
ax = fig.add_subplot(1,1,1)  

plt.title('Disciplinas x Classificadores')

plt.ylabel('Acurácia')

plt.xlabel('Disciplinas')

major_ticks = np.arange(0, 101, 10)                                              
minor_ticks = np.arange(0, 101, 2.5)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.set_xticks(ind + width / 3)
#ax.set_xticklabels(result['Disciplina'])

b1 = ax.bar(ind, result['Naive Bayes'], width, color='#77b5e5')#, yerr=result['Naive BayesDP'])
(x, caps, _) = ax.errorbar(ind, result['Naive Bayes'], yerr=result['Naive BayesDP'], elinewidth=1, capsize=8.0, fmt='none')
#height = x.get_linewidth()
height = 0
for cap in caps:
    cap.set_markeredgewidth(1)
    height = cap.get_ydata(orig=True)[0]
i = 0
for rect in b1:
    val = result.iloc[i]['Naive BayesDP']
    #ax.text(ind+0.01,height, 't')
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2,1.02*height,"%.2f%%" % val)
    i = i + 1
            

b2 = ax.bar(ind + width, result['Decision Tree'], width, color='#0747b2')#, yerr=result['Decision TreeDP'])
(x, caps, _) = ax.errorbar(ind + width, result['Decision Tree'], yerr=result['Decision TreeDP'], elinewidth=1.0, capsize=8.0, fmt='none')            
height = 0
for cap in caps:
    cap.set_markeredgewidth(1)            
    height = cap.get_ydata(orig=True)[0]
i = 0
for rect in b2:
    val = result.iloc[i]['Decision TreeDP']
    #ax.text(ind+0.01 + width,height, '%.2f%%' % result.iloc[i]['Decision TreeDP'])
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2,1.02*height,"%.2f%%" % val)
    i = i + 1


b3 = ax.bar(ind + (width*2), result['SVM'], width, color='#cebe6f')#, yerr=result['Decision TreeDP'])
(x, caps, _) = ax.errorbar(ind + (width*2), result['SVM'], yerr=result['SVMDP'], elinewidth=1.0, capsize=8.0, fmt='none')            
height = 0
for cap in caps:
    cap.set_markeredgewidth(1)            
    height = cap.get_ydata(orig=True)[0]
i = 0
for rect in b3:
    val = result.iloc[i]['SVMDP']
    #ax.text(ind+0.01 + (width*2),height, '%.2f%%' % result.get_value(i,'SVMDP'))
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2,1.02*height,"%.2f%%" % val)
    i = i + 1
    




            
#b3 = ax.bar(ind + (width*2), result['SVM'], width, color='#cebe6f')#, yerr=result['Decision TreeDP'])
#(_, caps, _) = ax.errorbar(ind + (width*2), result['SVM'], yerr=result['SVMDP'], elinewidth=1.0, capsize=8.0)            
#for cap in caps:
#    cap.set_markeredgewidth(2)            
            


#b3 = ax.bar(ind + (width*2), result['TesteSucesso'], width, color='#cebe6f')
#b4 = ax.bar(ind + (width*3), result['TesteInsucesso'], width, color='#a37f00')


#l3 = ax.plot(ind + (width*3) / 2, result['TreinoDesbalanceamento'], '#ff6600', marker='D')
#l4 = ax.plot(ind + (width*3) / 2, result['TesteDesbalanceamento'], '#009933', marker='D')

#ax.legend((b1[0], b2[0], b3[0], b4[0], l1[0], l2[0]),
#          ('Treino Sucesso', 'Treino Insucesso', 'Teste Sucesso', 'Teste Insucesso', 'Acur. Original', 'Acur. CORAL'), loc=2, bbox_to_anchor=(1.05, 1))

#ax.legend((b1[0], b2[0], b3[0], b4[0], l1[0], l2[0]),
#          ('Treino Sucesso', 'Treino Insucesso', 'Teste Sucesso', 'Teste Insucesso', 'Acur. Original', 'Acur. CORAL'),bbox_to_anchor=(0., 0., 1., -0.09),ncol=6,mode="expand", borderaxespad=0.)

ax.set_ylim(ymin=40, ymax=100)

ax.legend((b1[0], b2[0], b3[0]),
          ('Naive Bayes', 'Decision Tree', 'SVM'),bbox_to_anchor=(0., 0., 1., -0.15),ncol=3, borderaxespad=0.)

#plt.legend(, loc=3,
#           )

#ax.legend((l3[0], l4[0], l1[0], l2[0]),
#          ('Treino Desbal. %', 'Teste Desbal. %', 'Acur. Original %', 'Acur. CORAL %'), 
#          loc=4, bbox_to_anchor=(1.05, 1))

plt.xticks(ind + width / 2)

#ax.grid(which='both')                                                            

# or if you want differnet settings for the grids:                               
#ax.grid(which='minor', alpha=0.4)                                                
ax.yaxis.grid(which="major", color='#000000', linestyle=':', linewidth=0.5)

ax.yaxis.grid(True)

#plot table
table_rows = ['Naive Bayes','Decision Tree','SVM']
table_cols = []
cell_text=[]

for idx, row in result.iterrows():
    cell_text.append([float("{0:.2f}".format(row['Naive Bayes']))])
    cell_text.append([float("{0:.2f}".format(row['Decision Tree']))])
    cell_text.append([float("{0:.2f}".format(row['SVM']))])
    table_cols.append([row['Disciplina']])

table = plt.table(cellText=cell_text,
                  rowLabels=table_rows,
                  colLabels=table_cols,
                  loc='bottom')

plt.show()

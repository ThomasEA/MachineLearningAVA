# -*- coding: utf-8 -*-
"""
Created on Fri Nov  17 09:56:35 2017

@author: Everton

Experimento final para discussao no artigo

Cenario:
        Normalizaçao pela vari

"""
import pandas as pd
import numpy as np
import preditor

import matplotlib.pyplot as plt

def sumarizar(i, disciplina, classificador, df, result, coral=False):

    """    
    result.set_value(i,'Classificador',classificadores[classificador])
    result.set_value(i,'Disciplina',disciplina_string)
    result.set_value(i,'Turma','T' + str(i))
    result.set_value(i,'Coral', use_coral)
    result.set_value(i,'Acur',accuracy * 100)
    result.set_value(i,'TreinoSucesso',len(target_s[target_s == 0]) / len(target_s) * 100)
    result.set_value(i,'TreinoInsucesso',len(target_s[target_s == 1]) / len(target_s) * 100)
    result.set_value(i,'TreinoDesbalanceamento',(len(target_s[target_s == 1]) / len(target_s) * 100) / (len(target_s[target_s == 0]) / len(target_s) * 100) * 5)
    result.set_value(i,'TesteSucesso',len(target_t[target_t == 0]) / len(target_t) * 100)
    result.set_value(i,'TesteInsucesso',len(target_t[target_t == 1]) / len(target_t) * 100)
    result.set_value(i,'TesteDesbalanceamento',(len(target_t[target_t == 1]) / len(target_t) * 100) / (len(target_t[target_t == 0]) / len(target_t) * 100) * 5)    
    """
    result.set_value(i,'Disciplina', disciplina)
    if (coral==True):
        result.set_value(i,classificador + 'Coral', df['Acur'].mean())
        result.set_value(i,classificador + 'CoralDP', df['Acur'].std(ddof=1))
    else:
        result.set_value(i,classificador, df['Acur'].mean())
        result.set_value(i,classificador + 'DP', df['Acur'].std(ddof=1))

#Carrega dataset
df = pd.read_csv('../../dataset_m3_m6.csv', sep=';')

plt.style.use('seaborn-colorblind')
plt.rcParams['figure.figsize'] = (11,7)

use_normalization = False
use_normalization_turma = True
use_coral = True

#-------------------------------------------------------
# Configuração de filtros para o dataset
modulo_s = 6 #0 = ignora o módulo. Lembrando que só existem os módulos 3 e 6
classificador = 3

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
disciplina_s = 60500
disciplina_string = str(disciplinas[disciplina_s])
d1 = preditor.process(df, disciplina_s, modulo_s, classificador, use_coral=False, use_normalization=True, use_normalization_turma=False)
sumarizar(1, disciplina_string, classificadores[classificador], d1, result)
d2 = preditor.process(df, disciplina_s, modulo_s, classificador, use_coral=True, use_normalization=True, use_normalization_turma=False)
sumarizar(1, disciplina_string, classificadores[classificador], d2, result, coral=True)

#-----------------------------------------

#-----------------------------------------
disciplina_s = 60463
disciplina_string = str(disciplinas[disciplina_s])
d1 = preditor.process(df, disciplina_s, modulo_s, classificador, use_coral=False, use_normalization=True, use_normalization_turma=False)
sumarizar(2, disciplina_string, classificadores[classificador], d1, result)
d2 = preditor.process(df, disciplina_s, modulo_s, classificador, use_coral=True, use_normalization=True, use_normalization_turma=False)
sumarizar(2, disciplina_string, classificadores[classificador], d2, result, coral=True)
#-----------------------------------------

#-----------------------------------------
disciplina_s = 60465
disciplina_string = str(disciplinas[disciplina_s])
d1 = preditor.process(df, disciplina_s, modulo_s, classificador, use_coral=False, use_normalization=True, use_normalization_turma=False)
sumarizar(3, disciplina_string, classificadores[classificador], d1, result)
d2 = preditor.process(df, disciplina_s, modulo_s, classificador, use_coral=True, use_normalization=True, use_normalization_turma=False)
sumarizar(3, disciplina_string, classificadores[classificador], d2, result, coral=True)
#-----------------------------------------

#-----------------------------------------
disciplina_s = 50404
disciplina_string = str(disciplinas[disciplina_s])
d1 = preditor.process(df, disciplina_s, modulo_s, classificador, use_coral=False, use_normalization=True, use_normalization_turma=False)
sumarizar(4, disciplina_string, classificadores[classificador], d1, result)
d2 = preditor.process(df, disciplina_s, modulo_s, classificador, use_coral=True, use_normalization=True, use_normalization_turma=False)
sumarizar(4, disciplina_string, classificadores[classificador], d2, result, coral=True)
#-----------------------------------------

result = result.reset_index()

N = len(result)

ind = np.arange(N)  # the x locations for the groups
width = 0.20       # the width of the bars
ymin=40

fig = plt.figure()                                                               
ax = fig.add_subplot(1,1,1)  

classif_str = classificadores[classificador]

plt.title('Semana {} - {}'.format(modulo_s, classif_str))

plt.ylabel('Acurácia')

plt.xlabel('Disciplinas')

major_ticks = np.arange(0, 101, 10)                                              
minor_ticks = np.arange(0, 101, 2.5)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.set_xticks(ind + width / 2)
ax.set_xticklabels(result['Disciplina'])

b1 = ax.bar(ind, result[classif_str], width, color='b', yerr=result[classif_str + 'DP'])
height = 0
i = 0
for rect in b1:
    val = result.iloc[i][classif_str]
    valDP = result.iloc[i][classif_str + 'DP']
    height = rect.get_height()
    ax.text(0.01 + rect.get_x() + rect.get_width() / 2,1.02*height,"%.2f%%" % valDP)
    ax.text(rect.get_x(),(height + ymin)/2,"%.2f%%" % val, color='w')
    i = i + 1
            
b2 = ax.bar(ind + width, result[classif_str + 'Coral'], width, color='r', yerr=result[classif_str + 'CoralDP'])
height = 0
i = 0
for rect in b2:
    val = result.iloc[i][classif_str + 'Coral']
    valDP = result.iloc[i][classif_str + 'CoralDP']
    height = rect.get_height()
    ax.text(0.01 + rect.get_x() + rect.get_width() / 2,1.02*height,"%.2f%%" % valDP)
    ax.text(rect.get_x(),(height + ymin)/2,"%.2f%%" % val)
    i = i + 1

ax.set_ylim(ymin=ymin, ymax=100)

ax.legend((b1[0], b2[0]),
          ('z-Score', 'CORAL'),bbox_to_anchor=(0.5,-0.10), loc='upper center', ncol=2)

#plt.legend(, loc=3,
#           )

#ax.legend((l3[0], l4[0], l1[0], l2[0]),
#          ('Treino Desbal. %', 'Teste Desbal. %', 'Acur. Original %', 'Acur. CORAL %'), 
#          loc=4, bbox_to_anchor=(1.05, 1))

#plt.xticks((ind + (width * 2)) / 2)

#ax.grid(which='both')                                                            

# or if you want differnet settings for the grids:                               
#ax.grid(which='minor', alpha=0.4)                                                
ax.yaxis.grid(which="major", color='#000000', linestyle=':', linewidth=0.5)

ax.yaxis.grid(True)

plt.show()

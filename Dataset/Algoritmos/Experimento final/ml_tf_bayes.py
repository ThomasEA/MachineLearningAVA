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
        result.set_value(i,classificador + 'PrecisionSucessoCoral', df['Precision Sucesso'].mean())
        result.set_value(i,classificador + 'PrecisionInsucessoCoral', df['Precision Insucesso'].mean())
        result.set_value(i,classificador + 'RecallSucessoCoral', df['Recall Sucesso'].mean())
        result.set_value(i,classificador + 'RecallInsucessoCoral', df['Recall Insucesso'].mean())
    else:
        result.set_value(i,classificador, df['Acur'].mean())
        result.set_value(i,classificador + 'DP', df['Acur'].std(ddof=1))
        result.set_value(i,classificador + 'PrecisionSucesso', df['Precision Sucesso'].mean())
        result.set_value(i,classificador + 'PrecisionInsucesso', df['Precision Insucesso'].mean())
        result.set_value(i,classificador + 'RecallSucesso', df['Recall Sucesso'].mean())
        result.set_value(i,classificador + 'RecallInsucesso', df['Recall Insucesso'].mean())

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
classificador = 1

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
"""
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
"""
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

ax.yaxis.grid(which="major", color='#000000', linestyle=':', linewidth=0.5)

ax.yaxis.grid(True)

plt.show()

#--------------------------------
#Plot Recall e Precision

ymin = 0

fig = plt.figure()                                                               
ax = fig.add_subplot(1,1,1)  

classif_str = classificadores[classificador]

plt.title('Precision - Semana {} - {}'.format(modulo_s, classif_str))

plt.ylabel('Precision')

plt.xlabel('Disciplinas')

major_ticks = np.arange(0, 101, 10)                                              
minor_ticks = np.arange(0, 101, 2.5)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.set_xticks(ind + (width * 3) / 2)
ax.set_xticklabels(result['Disciplina'])

b1 = ax.bar(ind, result[classif_str+'PrecisionInsucesso'], width, color='#0077d4')
height = 0
i = 0
for rect in b1:
    val = result.iloc[i][classif_str+'PrecisionInsucesso']
    height = rect.get_height()
    ax.text(rect.get_x(),(height + ymin)/2,"%.2f%%" % val, color='w')
    i = i + 1
            
b2 = ax.bar(ind + width, result[classif_str+'PrecisionSucesso'], width, color='#c6e2ff')
height = 0
i = 0
for rect in b2:
    val = result.iloc[i][classif_str+'PrecisionSucesso']
    height = rect.get_height()
    ax.text(rect.get_x(),(height + ymin)/2,"%.2f%%" % val)
    i = i + 1

b3 = ax.bar(ind + (width*2), result[classif_str+'PrecisionInsucessoCoral'], width, color='#b22222')
height = 0
i = 0
for rect in b3:
    val = result.iloc[i][classif_str+'PrecisionInsucessoCoral']
    height = rect.get_height()
    ax.text(rect.get_x(),(height + ymin)/2,"%.2f%%" % val, color='w')
    i = i + 1
            
b4 = ax.bar(ind + (width*3), result[classif_str+'PrecisionSucessoCoral'], width, color='#fa8072')
height = 0
i = 0
for rect in b4:
    val = result.iloc[i][classif_str+'PrecisionSucessoCoral']
    height = rect.get_height()
    ax.text(rect.get_x(),(height + ymin)/2,"%.2f%%" % val)
    i = i + 1


ax.set_ylim(ymin=ymin, ymax=100)

ax.legend((b1[0], b2[0], b3[0], b4[0]),
          ('Insucesso', 'Sucesso', 'Insucesso - CORAL', 'Sucesso - CORAL'),bbox_to_anchor=(0.5,-0.10), loc='upper center', ncol=4)

ax.yaxis.grid(which="major", color='#000000', linestyle=':', linewidth=0.5)

ax.yaxis.grid(True)

plt.show()

#-------------- PLOT RECALL -----------------#

ymin = 0

fig = plt.figure()                                                               
ax = fig.add_subplot(1,1,1)  

classif_str = classificadores[classificador]

plt.title('Recall - Semana {} - {}'.format(modulo_s, classif_str))

plt.ylabel('Recall')

plt.xlabel('Disciplinas')

major_ticks = np.arange(0, 101, 10)                                              
minor_ticks = np.arange(0, 101, 2.5)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.set_xticks(ind + (width * 3) / 2)
ax.set_xticklabels(result['Disciplina'])

b1 = ax.bar(ind, result[classif_str+'RecallInsucesso'], width, color='#0077d4')
height = 0
i = 0
for rect in b1:
    val = result.iloc[i][classif_str+'RecallInsucesso']
    height = rect.get_height()
    ax.text(rect.get_x(),(height + ymin)/2,"%.2f%%" % val, color='w')
    i = i + 1
            
b2 = ax.bar(ind + width, result[classif_str+'RecallSucesso'], width, color='#c6e2ff')
height = 0
i = 0
for rect in b2:
    val = result.iloc[i][classif_str+'RecallSucesso']
    height = rect.get_height()
    ax.text(rect.get_x(),(height + ymin)/2,"%.2f%%" % val)
    i = i + 1

b3 = ax.bar(ind + (width*2), result[classif_str+'RecallInsucessoCoral'], width, color='#b22222')
height = 0
i = 0
for rect in b3:
    val = result.iloc[i][classif_str+'RecallInsucessoCoral']
    height = rect.get_height()
    ax.text(rect.get_x(),(height + ymin)/2,"%.2f%%" % val, color='w')
    i = i + 1
            
b4 = ax.bar(ind + (width*3), result[classif_str+'RecallSucessoCoral'], width, color='#fa8072')
height = 0
i = 0
for rect in b4:
    val = result.iloc[i][classif_str+'RecallSucessoCoral']
    height = rect.get_height()
    ax.text(rect.get_x(),(height + ymin)/2,"%.2f%%" % val)
    i = i + 1


ax.set_ylim(ymin=ymin, ymax=100)

ax.legend((b1[0], b2[0], b3[0], b4[0]),
          ('Insucesso', 'Sucesso', 'Insucesso - CORAL', 'Sucesso - CORAL'),bbox_to_anchor=(0.5,-0.10), loc='upper center', ncol=4)

ax.yaxis.grid(which="major", color='#000000', linestyle=':', linewidth=0.5)

ax.yaxis.grid(True)

plt.show()

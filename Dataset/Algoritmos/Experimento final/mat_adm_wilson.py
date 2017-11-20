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

#Carrega dataset
df = pd.read_csv('../../dataset_m3_m6.csv', sep=';')

plt.style.use('seaborn-colorblind')
plt.rcParams['figure.figsize'] = (11,7)

use_normalization = False
use_normalization_turma = False
use_coral = False

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
disciplina_s = 60465
disciplina_string = str(disciplinas[disciplina_s])
d1 = preditor.process(df, disciplina_s, modulo_s, classificador, use_coral=False, use_normalization=True, use_normalization_turma=False)

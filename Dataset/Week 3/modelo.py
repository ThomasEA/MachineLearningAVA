# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:30:47 2017

@author: Everton
"""
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import util as util
import process_turma_fold as process

from sklearn import preprocessing
from sklearn.utils import shuffle

#Configuração de módulo
modulo = 3 #Módulos 3 e 6

#Configuração de disciplinas
#
#Rac. Lógico = 60463
#Mat. Administração = 60465
#Lógica = 60500
#Fund. Proc. Adm. = 50404
disciplina = 60463

#df = pd.read_csv("Week" + str(modulo) + ".CSV", sep=';')
#df_original = pd.read_csv("m3_rac_logico_ext.CSV", sep=';', encoding = 'latin1')
df_original = pd.read_csv("Week3.CSV", sep=';', encoding = 'latin1')

#Altera os valores da coluna Evadido para binário
df_original['Evadido'] = np.where(df_original['Evadido'] == 'Sucesso', 1, 0).copy()

#Filtra a disciplina
#df = df_original.loc[df_original['CodigoDisciplina'] == disciplina]
df_groups = df_original.groupby('CodigoDisciplina')

df = df_groups.get_group(name=disciplina).copy()
#df = df_groups.filter(lambda x: x.name==disciplina)

#df = util.remove_irrelevant_features(df_clean)

"""
list_to_remove = [
            'CodigoPessoa',
            'NumeroModulos',
            'NumeroModulo',
            'NumeroUltimoModulo',
            'Numero_Dias_Acessados_Modulo',
            'Turno_TempoUsoTotal',
            'QuantidadeMatriculasValidas',
            'QuantidadeMatriculasAprovadas',
            'MatriculasPercentualAproveitamento',
            'StatusMatricula',
            'Forum_Quantidade_Post',
            'Forum_Quantidade_Visualizacoes',
            'Chat_Quantidade_Mensagens',
            'Chat_TempoUso',
            'Turno_PercentualUsoMadrugada',
            'Turno_PercentualUsoManha',
            'Turno_PercentualUsoTarde',
            'Turno_PercentualUsoNoite',
            'Turno_TempoUsoMadrugada',
            'Turno_TempoUsoManha',
            'Turno_TempoUsoTarde',
            'Turno_TempoUsoNoite',
            'Assignment_Post_Quantidade',
            'Assignment_View_Quantidade',
            'Assignment_View_TempoUso',
            'Resource_View_Quantidade',
            'Resource_View_Tempo',
            'Questionario_Quantidade',
            'Questionario_TempoUso',
            'Log_Post_Quantidade',
            'Log_View_Quantidade',
            'Disciplina_Reprovado',
            ]
"""
#df = df[df.columns.difference(list_to_remove)]
#df = df.drop(list_to_remove, axis=1)

#Calcular z-Score para algumas features
#--------------------------------------
features = df[df.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido','StatusMatricula'])]

scaler = preprocessing.StandardScaler().fit(features)
df_std = pd.DataFrame(scaler.transform(features), columns = list(features))

df_std['Evadido'] = df.Evadido
df_std['CodigoDisciplina'] = df.CodigoDisciplina
df_std['CodigoTurma'] = df.CodigoTurma
df_std['PeriodoLetivo'] = df.PeriodoLetivo
#--------------------------------------

print('Antes shuffle')
print(df_std['CodigoDisciplina'])

#Embaralha dataframe normalizado
df_normalized = shuffle(df_std)

print(df_normalized['CodigoDisciplina'])
print(str(len(df_normalized)))

#process.process(df_normalized)


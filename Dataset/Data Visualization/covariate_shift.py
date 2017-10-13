# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 23:22:51 2017

@author: Everton

Visualização de covariate shift

"""
import sys
sys.path.insert(0, '../Algoritmos')

import filter as filter
import pandas as pd
import seaborn as sns

disciplinas = {
        50404: 'Fund. Proc. Administrativo', 
        60463: 'Ofic. Raciocínio Lógico',
        60465: 'Matemática Administração',
        60500: 'Lógica'
    }

#-------------------------------------------------------
# Configuração de filtros para o dataset
disciplina = 60465
modulo = 3 #0 = ignora o módulo. Lembrando que só existem os módulos 3 e 6
periodo_letivo_source = [20120101,20120102,20120201,20120202]
periodo_letivo_test   = [20130101,20130102,20130201,20130202,20140101,20140102]
#-------------------------------------------------------

s_periodo = str(periodo_letivo_source)
t_periodo = str(periodo_letivo_test)

#Carrega dataset
df = pd.read_csv('../dataset_m3_m6.csv', sep=';')

#Filtra o dataset conforme a configuração selecionada e faz alguns ajustes no dataset
df_s, df_t = filter.filter_dataset(
                            df, 
                            modulo, 
                            disciplinas, 
                            disciplina, 
                            periodo_letivo_source, 
                            periodo_letivo_test,
                            feature_normalization=True)

sns.distplot(df_s.Evadido)
sns.distplot(df_t.Evadido)

#g = sns.JointGrid(x="Forum_Quantidade_Post_Somado", y="Login_Quantidade", data=df_t) 
#g.plot_joint(sns.regplot, order=2) 
#g.plot_marginals(sns.distplot)

#sns.pairplot(df_s)
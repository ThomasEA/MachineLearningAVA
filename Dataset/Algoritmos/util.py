#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:27:13 2017

@author: kratos
"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import itertools
import pandas as pd
from scipy import linalg as linear_alg
import graficos as graficos

from sklearn import preprocessing

matplotlib.style.use('ggplot')


def calcular_z_score(dataframe, columns):
    df_ret = dataframe
    
    for col in columns:
        df_ret[col] = (df_ret[col] - df_ret[col].mean())/df_ret[col].std(ddof=0)
    return df_ret;

def plot_dist(title, dataframe):
    """Plota gráfico de distribuição de features.
    
    Parâmetros:
        
        title     -- Título do gráfico
    
        dataframe -- Dataframe com os dados
    """
    #fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6,5))
    fig, (ax1) = plt.subplots(ncols=1, figsize=(13,5))
    
    ax1.set_title(title)
    
    for col in dataframe:
        sns.kdeplot(dataframe[col], ax=ax1)
    
    #plt.show()
    
    
    #sns.distplot(dataframe)

def sep_folds(dataframe, column):
    """Separa dataframe em folds, baseado na coluna informada
    
    Parâmetros:
        
        dataframe -- Dataframe com os dados
        
        columns -- Coluna para agrupar os dados
    """
    #separa em turmas
    folds = dataframe.groupby(column)
    return folds;

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe correta')
    plt.xlabel('Classe predita')
    
def show_confusion_matrix(C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Classe Predita', fontsize=12)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('Classe real', fontsize=12, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'Verdadeiro Negativo: %d\n(Num. Negativos: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round'))

    ax.text(0,1,
            'Falso Negativo: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round'))

    ax.text(1,0,
            'Falso Positivo: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round'))


    ax.text(1,1,
            'Verdadeiro Positivo: %d\n(Num. Positivos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'Recall: %.2f'%(tn / (tn+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round'))

    ax.text(2,1,
            'Recall: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round'))

    ax.text(2,2,
            'Acurácia: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round'))

    ax.text(0,2,
            'Precision: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round'))

    ax.text(1,2,
            'Precision: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round'))


    plt.tight_layout()
    plt.show()
    
def remove_irrelevant_features(df):
    return df[df.columns.difference([
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
            ])];
    

def clean_data(df, standardize=True, plot_cov=True, title="Clean Data - Covariancia"):
    #Altera os valores da coluna Evadido para binário
    df.Evadido = df.Evadido.map({'ReprEvadiu': 0, 'Sucesso': 1})
    
    #Calcular z-Score para algumas features
    #--------------------------------------
    if (standardize==True):
        features = df[df.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','NumeroModulo','Evadido'])]
        scaler = preprocessing.StandardScaler().fit(features)
        df_std = pd.DataFrame(scaler.transform(features), columns = list(features))

        df_std['Evadido'] = df.Evadido
        df_std['CodigoDisciplina'] = df.CodigoDisciplina
        df_std['CodigoTurma'] = df.CodigoTurma
        df_std['PeriodoLetivo'] = df.PeriodoLetivo
    else:
        df_std = df
    #--------------------------------------

    if (plot_cov==True):
        graficos.plot_cov_matrix(df_std, title)

    return df_std;

def standardize(df):
    features = df[df.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
    scaler = preprocessing.StandardScaler().fit(features)
    df_std = pd.DataFrame(scaler.transform(features), columns = list(features))
    
    df_std['Evadido'] = df.Evadido
    df_std['CodigoDisciplina'] = df.CodigoDisciplina
    df_std['CodigoTurma'] = df.CodigoTurma
    df_std['PeriodoLetivo'] = df.PeriodoLetivo
    return df_std.copy();

def normalize(df, except_fields=None):
    df = df.reset_index(drop=True)
    
    if (except_fields==None):
        features = df.copy()
    else:
        features = df.copy()[df.columns.difference(except_fields)]
    
    scaler = preprocessing.StandardScaler().fit(features)
    df_std = pd.DataFrame(scaler.transform(features), columns = list(features))
    
    if (except_fields!=None):
    #    for f in except_fields:
    #        df_std[f] = 
        ret = pd.concat([df_std,df[except_fields]], axis=1)
        return ret;
    
    return df_std;


def covariancia_mais_diag(df, lambda_par=1):
    df_ = df.cov()
    df_.replace(np.inf, 0,inplace=True)
    df_.replace(np.nan, 0,inplace=True)
    ID_S = np.eye(len(list(df_)))
    np.fill_diagonal(ID_S, lambda_par)
    return df_ + ID_S;

def whitening_values(df, kernel):
    c = linear_alg.fractional_matrix_power(kernel, -0.5)
    df = df.dot(c)
    return df;

def recolor_values(df, kernel):
    c = linear_alg.fractional_matrix_power(kernel, 0.5)
    
    #c.replace(np.inf, 0,inplace=True)
    #c.replace(np.nan, 0,inplace=True)
    df = df.dot(c)
    #df.replace(np.inf, 0,inplace=True)
    #df.replace(np.nan, 0,inplace=True)
    return df;


def correlation_alignment(df_s, df_t, lambda_par=1):
    
    df_s_tmp = df_s[df_s.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
    df_t_tmp = df_t[df_t.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
    
    graficos.plot_cov_matrix(df_s_tmp, 'CORAL - Ds - Covariância Original')
    graficos.plot_cov_matrix(df_t_tmp, 'CORAL - Ts - Covariância Original')
    
    df_s_cov = covariancia_mais_diag(df_s_tmp, lambda_par=lambda_par)
    
    graficos.plot_cov_matrix(df_s_cov, 'CORAL - Ds - Covariância + Identidade')
    
    df_t_cov = covariancia_mais_diag(df_t_tmp, lambda_par=lambda_par)

    graficos.plot_cov_matrix(df_t_cov, 'CORAL - Ts - Covariância + Identidade')
    
    df_s_ = whitening_values(df_s_tmp, df_s_cov)
    
    graficos.plot_cov_matrix(df_s_, 'CORAL - Ds - Covariancia Whitening')
    
    df_s_ = recolor_values(df_s_, df_t_cov)
    
    graficos.plot_cov_matrix(df_s_, 'CORAL - Ds - Covariancia Re-color')
    
    df_s_['Evadido'] = df_s.Evadido
    df_s_['CodigoDisciplina'] = df_s.CodigoDisciplina
    df_s_['CodigoTurma'] = df_s.CodigoTurma
    df_s_['PeriodoLetivo'] = df_s.PeriodoLetivo
    
    return df_s_;
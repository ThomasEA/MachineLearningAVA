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

    fig = plt.figure(figsize=(8,8))
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
    ax.set_xlabel('Classe Predita', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('Classe real', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'Verdadeiro Negativo: %d\n(Num. Negativos: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'Falso Negativo: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'Falso Positivo: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'Verdadeiro Positivo: %d\n(Num. Positivos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'Precision: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'Precision: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Acurácia: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Recall: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Recall: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


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
    

def clean_data(df, standardize=True):
    #Altera os valores da coluna Evadido para binário
    df.Evadido = df.Evadido.map({'ReprEvadiu': 0, 'Sucesso': 1})
    
    #Calcular z-Score para algumas features
    #--------------------------------------
    if (standardize==True):
        features = df[df.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
        scaler = preprocessing.StandardScaler().fit(features)
        df_std = pd.DataFrame(scaler.transform(features), columns = list(features))
        
        df_std['Evadido'] = df.Evadido
        df_std['CodigoDisciplina'] = df.CodigoDisciplina
        df_std['CodigoTurma'] = df.CodigoTurma
        df_std['PeriodoLetivo'] = df.PeriodoLetivo
    else:
        df_std = df
    #--------------------------------------
    return df_std.copy();

def standardize(df):
    features = df[df.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
    scaler = preprocessing.StandardScaler().fit(features)
    df_std = pd.DataFrame(scaler.transform(features), columns = list(features))
    
    df_std['Evadido'] = df.Evadido
    df_std['CodigoDisciplina'] = df.CodigoDisciplina
    df_std['CodigoTurma'] = df.CodigoTurma
    df_std['PeriodoLetivo'] = df.PeriodoLetivo
    return df_std.copy();

def correlation_alignment(df_s, df_t):
    
    df_s_tmp = df_s[df_s.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
    df_t_tmp = df_t[df_t.columns.difference(['CodigoDisciplina','CodigoTurma','PeriodoLetivo','Evadido'])]
    
    ID_S = np.eye(len(list(df_s_tmp)))
    C_S = df_s_tmp.cov() + ID_S

    print(C_S)
                    
    ID_T = np.eye(len(list(df_t_tmp)))
    C_T = df_t_tmp.cov() + ID_T
    
    #df_s_tmp = df_s_tmp * (C_S**(-1/2))
    df_s_tmp = df_s_tmp * (C_T**(1/2))
    """
    df_s_tmp['Evadido'] = df_s.Evadido
    df_s_tmp['CodigoDisciplina'] = df_s.CodigoDisciplina
    df_s_tmp['CodigoTurma'] = df_s.CodigoTurma
    df_s_tmp['PeriodoLetivo'] = df_s.PeriodoLetivo
    
    return df_s_tmp.copy();
    """

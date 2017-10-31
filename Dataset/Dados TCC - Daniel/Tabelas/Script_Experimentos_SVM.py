import pandas as pd
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

#============================================ Variaveis
disciplina = "Rac_Logico"
#disciplina = "Mat_Administracao"
#disciplina = "Fund_Proc_Administrativo"
#disciplina = "Logica"
Modulo = 3
profundidadeArvore = 2
vFolds = 10

CSV = "%s_Modulo_%s.csv"%(disciplina, Modulo)
dados = pd.read_csv(CSV, sep=';')
#======================================================

pd.set_option('display.max_columns', None)

if disciplina == "Rac_Logico" :
    periodoLetivo = 20130201
else:
    periodoLetivo = 20130102

# 0 - Treino; 1 - Teste
for index, row in dados.iterrows():
    ifor_val = ""
    if row['PeriodoLetivo'] >= periodoLetivo:
        ifor_val = 1
    else:
        ifor_val = 0
            
    dados.set_value(index,'Origem', ifor_val)

dados_Teste = dados[dados.PeriodoLetivo >= periodoLetivo]
dados_Treino = dados[dados.PeriodoLetivo < periodoLetivo]

dados_Teste = shuffle(dados_Teste)
dados_Treino = shuffle(dados_Treino)

Teste_Suc = dados_Teste[dados_Teste.Sucesso == 1]
Teste_Suc = shuffle(Teste_Suc)

Teste_Insuc = dados_Teste[dados_Teste.Sucesso == 0]
Teste_Insuc = shuffle(Teste_Insuc)

print('\n ============ Dataset Teste: \n\n')

print('Tamanho dataset Teste:')
print(dados_Teste.shape)
print('Tamanho dataset Sucesso Teste:')
print(Teste_Suc.shape)
print('Tamanho dataset Insucesso Teste:')
print(Teste_Insuc.shape)

Treino_Suc = dados_Treino[dados_Treino.Sucesso == 1]
Treino_Suc = shuffle(Treino_Suc)

Treino_Insuc = dados_Treino[dados_Treino.Sucesso == 0]
Treino_Insuc = shuffle(Treino_Insuc)

print('\n ============ Dataset Treino: \n\n')

print('Tamanho dataset Treino:')
print(dados_Treino.shape)
print('Tamanho dataset Sucesso Treino:')
print(Treino_Suc.shape)
print('Tamanho dataset Insucesso Treino:')
print(Treino_Insuc.shape)

rTesteInsuc = len(Teste_Insuc.index)
rTesteSuc = len(Teste_Suc.index)
rTreinoInsuc = len(Treino_Insuc.index)
rTreinosuc = len(Treino_Suc.index)

listaValores = []
listaValores.append(rTesteInsuc)
listaValores.append(rTesteSuc)
listaValores.append(rTreinoInsuc)
listaValores.append(rTreinosuc)

print('Amostras mínimas de cada dataset.: ')
print(listaValores)

#deletando a coluna periodo letivo como teste da acurácia
#==============================================================================
del Teste_Insuc['PeriodoLetivo']
del Teste_Suc['PeriodoLetivo']
del Treino_Insuc['PeriodoLetivo']
del Treino_Suc['PeriodoLetivo'] 
#==============================================================================

print('\n\n ====== \n\n')

print(min(listaValores))

#Raciocínio Lógico:
#vAmostraSucesso = 16
#vAmostraInsucesso = 14

#Lógica
#vAmostraSucesso = 5
#vAmostraInsucesso = 9

#Matemática Para Administração
#vAmostraSucesso = 11
#vAmostraInsucesso = 20

#Fundamentos do Processo Administrativo
#vAmostraSucesso = 23
#vAmostraInsucesso = 11

#Para manter o balanceamento dos dados de treinamento e teste verifica a menor porção para poder fazer nos dois conjunto de dados
if rTesteInsuc < rTreinoInsuc:
    vAmostraInsucesso = int(rTesteInsuc / vFolds)
else:
    vAmostraInsucesso = int(rTreinoInsuc / vFolds)
    
if rTesteSuc < rTreinosuc:
    vAmostraSucesso = int(rTesteSuc / vFolds)
else:
    vAmostraSucesso = int(rTreinosuc / vFolds)

print('Amostra Sucesso.: ' + str(vAmostraSucesso))
print('Amostra Insucesso.: ' + str(vAmostraInsucesso))

#database.tail(x) remove as ultimas x linhas do database
frames = [Teste_Insuc.tail(vAmostraInsucesso), Teste_Suc.tail(vAmostraSucesso), Treino_Insuc.tail(vAmostraInsucesso), Treino_Suc.tail(vAmostraSucesso)]
FoldUm = pd.concat(frames)

print('Tamanho do dataset antes de deletar as linhas.:' + str(len(Teste_Insuc.index)))
Teste_Insuc = Teste_Insuc[:-vAmostraInsucesso]
Teste_Insuc = shuffle(Teste_Insuc)

Teste_Suc = Teste_Suc[:-vAmostraSucesso]
Teste_Suc = shuffle(Teste_Suc)

Treino_Insuc = Treino_Insuc[:-vAmostraInsucesso]
Treino_Insuc = shuffle(Treino_Insuc)

Treino_Suc = Treino_Suc[:-vAmostraSucesso]
Treino_Suc = shuffle(Treino_Suc)
print('Tamanho do dataset depois de deletar as linhas.:' + str(len(Teste_Insuc.index)))

print('\nMontou a fold 1')
print('Tamanho Fold 1.:' + str(FoldUm.shape))

print('===============================================')

#database.tail(x) remove as ultimas x linhas do database
frames = [Teste_Insuc.tail(vAmostraInsucesso), Teste_Suc.tail(vAmostraSucesso), Treino_Insuc.tail(vAmostraInsucesso), Treino_Suc.tail(vAmostraSucesso)]
FoldDois = pd.concat(frames)

print('Tamanho do dataset antes de deletar as linhas.:' + str(len(Teste_Insuc.index)))
Teste_Insuc = Teste_Insuc[:-vAmostraInsucesso]
Teste_Insuc = shuffle(Teste_Insuc)

Teste_Suc = Teste_Suc[:-vAmostraSucesso]
Teste_Suc = shuffle(Teste_Suc)

Treino_Insuc = Treino_Insuc[:-vAmostraInsucesso]
Treino_Insuc = shuffle(Treino_Insuc)

Treino_Suc = Treino_Suc[:-vAmostraSucesso]
Treino_Suc = shuffle(Treino_Suc)
print('Tamanho do dataset depois de deletar as linhas.:' + str(len(Teste_Insuc.index)))

print('\nMontou a fold 2')
print('Tamanho Fold 2.:' + str(FoldDois.shape))

print('===============================================')

#database.tail(x) remove as ultimas x linhas do database
frames = [Teste_Insuc.tail(vAmostraInsucesso), Teste_Suc.tail(vAmostraSucesso), Treino_Insuc.tail(vAmostraInsucesso), Treino_Suc.tail(vAmostraSucesso)]
FoldTres = pd.concat(frames)

print('Tamanho do dataset antes de deletar as linhas.:' + str(len(Teste_Insuc.index)))
Teste_Insuc = Teste_Insuc[:-vAmostraInsucesso]
Teste_Insuc = shuffle(Teste_Insuc)

Teste_Suc = Teste_Suc[:-vAmostraSucesso]
Teste_Suc = shuffle(Teste_Suc)

Treino_Insuc = Treino_Insuc[:-vAmostraInsucesso]
Treino_Insuc = shuffle(Treino_Insuc)

Treino_Suc = Treino_Suc[:-vAmostraSucesso]
Treino_Suc = shuffle(Treino_Suc)
print('Tamanho do dataset depois de deletar as linhas.:' + str(len(Teste_Insuc.index)))

print('\nMontou a fold 3')
print('Tamanho Fold 3.:' + str(FoldTres.shape))

print('===============================================')

frames = [Teste_Insuc.tail(vAmostraInsucesso), Teste_Suc.tail(vAmostraSucesso), Treino_Insuc.tail(vAmostraInsucesso), Treino_Suc.tail(vAmostraSucesso)]
FoldQuatro = pd.concat(frames)

print('Tamanho do dataset antes de deletar as linhas.:' + str(len(Teste_Insuc.index)))
Teste_Insuc = Teste_Insuc[:-vAmostraInsucesso]
Teste_Insuc = shuffle(Teste_Insuc)

Teste_Suc = Teste_Suc[:-vAmostraSucesso]
Teste_Suc = shuffle(Teste_Suc)

Treino_Insuc = Treino_Insuc[:-vAmostraInsucesso]
Treino_Insuc = shuffle(Treino_Insuc)

Treino_Suc = Treino_Suc[:-vAmostraSucesso]
Treino_Suc = shuffle(Treino_Suc)
print('Tamanho do dataset depois de deletar as linhas.:' + str(len(Teste_Insuc.index)))

print('\nMontou a fold 4')
print('Tamanho Fold 4.:' + str(FoldQuatro.shape))

print('===============================================')

frames = [Teste_Insuc.tail(vAmostraInsucesso), Teste_Suc.tail(vAmostraSucesso), Treino_Insuc.tail(vAmostraInsucesso), Treino_Suc.tail(vAmostraSucesso)]
FoldCinco = pd.concat(frames)

print('Tamanho do dataset antes de deletar as linhas.:' + str(len(Teste_Insuc.index)))
Teste_Insuc = Teste_Insuc[:-vAmostraInsucesso]
Teste_Insuc = shuffle(Teste_Insuc)

Teste_Suc = Teste_Suc[:-vAmostraSucesso]
Teste_Suc = shuffle(Teste_Suc)

Treino_Insuc = Treino_Insuc[:-vAmostraInsucesso]
Treino_Insuc = shuffle(Treino_Insuc)

Treino_Suc = Treino_Suc[:-vAmostraSucesso]
Treino_Suc = shuffle(Treino_Suc)
print('Tamanho do dataset depois de deletar as linhas.:' + str(len(Teste_Insuc.index)))

print('\nMontou a fold 5')
print('Tamanho Fold 5.:' + str(FoldCinco.shape))

print('===============================================')

frames = [Teste_Insuc.tail(vAmostraInsucesso), Teste_Suc.tail(vAmostraSucesso), Treino_Insuc.tail(vAmostraInsucesso), Treino_Suc.tail(vAmostraSucesso)]
FoldSeis = pd.concat(frames)

print('Tamanho do dataset antes de deletar as linhas.:' + str(len(Teste_Insuc.index)))
Teste_Insuc = Teste_Insuc[:-vAmostraInsucesso]
Teste_Insuc = shuffle(Teste_Insuc)

Teste_Suc = Teste_Suc[:-vAmostraSucesso]
Teste_Suc = shuffle(Teste_Suc)

Treino_Insuc = Treino_Insuc[:-vAmostraInsucesso]
Treino_Insuc = shuffle(Treino_Insuc)

Treino_Suc = Treino_Suc[:-vAmostraSucesso]
Treino_Suc = shuffle(Treino_Suc)
print('Tamanho do dataset depois de deletar as linhas.:' + str(len(Teste_Insuc.index)))

print('\nMontou a fold 6')
print('Tamanho Fold 6.:' + str(FoldSeis.shape))

print('===============================================')

frames = [Teste_Insuc.tail(vAmostraInsucesso), Teste_Suc.tail(vAmostraSucesso), Treino_Insuc.tail(vAmostraInsucesso), Treino_Suc.tail(vAmostraSucesso)]
FoldSete = pd.concat(frames)

print('Tamanho do dataset antes de deletar as linhas.:' + str(len(Teste_Insuc.index)))
Teste_Insuc = Teste_Insuc[:-vAmostraInsucesso]
Teste_Insuc = shuffle(Teste_Insuc)

Teste_Suc = Teste_Suc[:-vAmostraSucesso]
Teste_Suc = shuffle(Teste_Suc)

Treino_Insuc = Treino_Insuc[:-vAmostraInsucesso]
Treino_Insuc = shuffle(Treino_Insuc)

Treino_Suc = Treino_Suc[:-vAmostraSucesso]
Treino_Suc = shuffle(Treino_Suc)
print('Tamanho do dataset depois de deletar as linhas.:' + str(len(Teste_Insuc.index)))

print('\nMontou a fold 7')
print('Tamanho Fold 7.:' + str(FoldSete.shape))

print('===============================================')

frames = [Teste_Insuc.tail(vAmostraInsucesso), Teste_Suc.tail(vAmostraSucesso), Treino_Insuc.tail(vAmostraInsucesso), Treino_Suc.tail(vAmostraSucesso)]
FoldOito = pd.concat(frames)

print('Tamanho do dataset antes de deletar as linhas.:' + str(len(Teste_Insuc.index)))
Teste_Insuc = Teste_Insuc[:-vAmostraInsucesso]
Teste_Insuc = shuffle(Teste_Insuc)

Teste_Suc = Teste_Suc[:-vAmostraSucesso]
Teste_Suc = shuffle(Teste_Suc)

Treino_Insuc = Treino_Insuc[:-vAmostraInsucesso]
Treino_Insuc = shuffle(Treino_Insuc)

Treino_Suc = Treino_Suc[:-vAmostraSucesso]
Treino_Suc = shuffle(Treino_Suc)
print('Tamanho do dataset depois de deletar as linhas.:' + str(len(Teste_Insuc.index)))

print('\nMontou a fold 8')
print('Tamanho Fold 8.:' + str(FoldOito.shape))

print('===============================================')

frames = [Teste_Insuc.tail(vAmostraInsucesso), Teste_Suc.tail(vAmostraSucesso), Treino_Insuc.tail(vAmostraInsucesso), Treino_Suc.tail(vAmostraSucesso)]
FoldNove = pd.concat(frames)

print('Tamanho do dataset antes de deletar as linhas.:' + str(len(Teste_Insuc.index)))
Teste_Insuc = Teste_Insuc[:-vAmostraInsucesso]
Teste_Insuc = shuffle(Teste_Insuc)

Teste_Suc = Teste_Suc[:-vAmostraSucesso]
Teste_Suc = shuffle(Teste_Suc)

Treino_Insuc = Treino_Insuc[:-vAmostraInsucesso]
Treino_Insuc = shuffle(Treino_Insuc)

Treino_Suc = Treino_Suc[:-vAmostraSucesso]
Treino_Suc = shuffle(Treino_Suc)
print('Tamanho do dataset depois de deletar as linhas.:' + str(len(Teste_Insuc.index)))

print('\nMontou a fold 9')
print('Tamanho Fold 9.:' + str(FoldNove.shape))

print('===============================================')

frames = [Teste_Insuc.tail(vAmostraInsucesso), Teste_Suc.tail(vAmostraSucesso), Treino_Insuc.tail(vAmostraInsucesso), Treino_Suc.tail(vAmostraSucesso)]
FoldDez = pd.concat(frames)

print('Tamanho do dataset antes de deletar as linhas.:' + str(len(Teste_Insuc.index)))
Teste_Insuc = Teste_Insuc[:-vAmostraInsucesso]
Teste_Insuc = shuffle(Teste_Insuc)

Teste_Suc = Teste_Suc[:-vAmostraSucesso]
Teste_Suc = shuffle(Teste_Suc)

Treino_Insuc = Treino_Insuc[:-vAmostraInsucesso]
Treino_Insuc = shuffle(Treino_Insuc)

Treino_Suc = Treino_Suc[:-vAmostraSucesso]
Treino_Suc = shuffle(Treino_Suc)
print('Tamanho do dataset depois de deletar as linhas.:' + str(len(Teste_Insuc.index)))

print('\nMontou a fold 10')
print('Tamanho Fold 10.:' + str(FoldDez.shape))

vetFolds = [FoldUm, FoldDois, FoldTres, FoldQuatro, FoldCinco, FoldSeis, FoldSete, FoldOito, FoldNove, FoldDez]
vetAccuracy = []

parameters = {'kernel':('linear', 'rbf'), 'C': range(1, 20), 'cache_size': [1, 500000]}
Modelo_SVM = svm.SVC()
SVM = GridSearchCV(Modelo_SVM, parameters)

TN = 0
TP = 0
FN = 0
FP = 0

for i in range(0, 10):
    vetDtTreino = []
    for j in range(0,10):
        if(i == j):
            dtTeste = vetFolds[i]
        else:
            vetDtTreino.append(vetFolds[j])

    #concatena todas as folds de treino para montar um conjunto para treinar    
    dtTreino = pd.concat(vetDtTreino)
    print('\n========== Iteração.: ' + str(i))
    
#    print('Antes de normalizar (Treino):')
#    print(dtTreino.head(20).Sucesso)    
#    print('Antes de normalizar (Teste):')
#    print(dtTeste.head(20).Sucesso)
    
    #Normalizar usando z-scores (valor - média)%desviopadrao    
    colunas = list(dtTreino)
    scaler = preprocessing.StandardScaler().fit(dtTreino)
    
    dtTreinoNormalized = pd.DataFrame(scaler.transform(dtTreino), columns=colunas)
    dtTesteNormalized = pd.DataFrame(scaler.transform(dtTeste), columns=colunas)
    dtTreino = dtTreino.reset_index()
    dtTeste = dtTeste.reset_index()
    dtTreinoNormalized.Sucesso = dtTreino.Sucesso
    dtTreinoNormalized.Origem = dtTreino.Origem
    dtTesteNormalized.Sucesso = dtTeste.Sucesso
    dtTesteNormalized.Origem = dtTeste.Origem
    
#    print('\n=====================')
#    print('\nDepois de normalizar (Treino):')
#    print(dtTreinoNormalized.head(20).Sucesso)
#    print('\nDepois de normalizar (Teste):')    
#    print(dtTesteNormalized.head(20).Sucesso)
#    print('\n=====================')
    
    #embaralha os conjuntos
    dtTreinoNormalized = shuffle(dtTreinoNormalized)
    dtTesteNormalized = shuffle(dtTesteNormalized)
    
    #separa os dados de treino e teste das colunas a serem preditas
    Treino, target_treino = dtTreinoNormalized.iloc[:,:-1], dtTreinoNormalized.iloc[:, -1]
    Teste, target_teste = dtTesteNormalized.iloc[:,:-1], dtTesteNormalized.iloc[:, -1]
    print('Tamanho do conjunto de Treino.: '+ str(Treino.shape))
    print('Tamanho do conjunto de Teste.: '+ str(Teste.shape))
    arqCSV = str('SVM_' + disciplina + '_Fold'+ str(i)+'_Modulo_'+str(Modulo)+'.csv')
    
    #Treinando o modelo
    SVM.fit(Treino, target_treino)
    
    #prediz resultados com o modelo treinado
    predicted = SVM.predict(Teste)
    print("Best parameters set found on development set:")
    print(SVM)
    
    #transforma o array de predições em um dataframe para poder concatenar com o dataframe de testes
    prediction_df = pd.DataFrame(predicted, index=None)
    
    #faz uma cópia do dataframe usado para predizer para poder manuseá-lo
    tmpDfTeste = dtTesteNormalized.copy()
    tmpDfTeste = tmpDfTeste.reset_index()
    tmpDfTeste = tmpDfTeste.join(prediction_df)
    tmpDfTeste.to_csv(arqCSV, sep=';', index=False)

    accuracy = accuracy_score(target_teste, predicted)
    print('Acurácia.: ' + str(accuracy))
    vetAccuracy.append(accuracy)

    cm = confusion_matrix(target_teste, predicted)
    print(cm)
    TN = TN + cm[0][0]
    FN = FN + cm[1][0]
    TP = TP + cm[1][1]
    FP = FP + cm[0][1]
    
    #time.sleep(5)
    
#    fig = pl.figure()
#    ax = fig.add_subplot(111)
#    cax = ax.matshow(cm)
#    pl.title('Confusion matrix of the classifier')
#    fig.colorbar(cax)
#    pl.xlabel('Predicted')
#    pl.ylabel('True')
#    pl.show()

print('\n=====================')
sumAccuracy = sum(vetAccuracy)
qtAccuracy = len(vetAccuracy)
avgAccuracy = float(sumAccuracy / qtAccuracy)
print('Acurácia Média.: ' + str(avgAccuracy))
print('Verdadeiro Positivo.: ' + str(TP) + ' Verdadeiro Negativo.: ' + str(TN) + ' Falso Positivo.: ' + str(FP) + ' Falso Negativo.: ' + str(FN))
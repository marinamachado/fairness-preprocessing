import os
import pandas as pd
import numpy as np
import time

# seleção de modelos
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold

# medidas de desempenho
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from aif360.sklearn.metrics import generalized_fpr, selection_rate

# medidas de fairness
from aif360.sklearn.metrics import difference, statistical_parity_difference, disparate_impact_ratio, average_odds_difference, equal_opportunity_difference

# tela
from IPython.display import clear_output

measures_columns = ['accuracy', 'dif_accuracy', 'balanced_accuracy', 'dif_balanced_accuracy', 'recall', 'dif_recall', 
                    'precision', 'dif_precision', 'fpr', 'dif_fpr', 'selection_rate', 'dif_selection_rate', 
                    'dif_statistical_parity', 'dif_equal_opp', 'dif_avg_odds', 'disparate_impacto_ratio']

def fpr_score(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return fp/(fp+tn)

performance_measures = {
    'accuracy' : accuracy_score,
    'balanced_accuracy' : balanced_accuracy_score,
    'recall' : recall_score,
    'precision' : precision_score,
    'fpr' : fpr_score,
    'selection_rate' : selection_rate 
}

fairness_measures = {
    'dif_accuracy' : [difference, accuracy_score],
    'dif_balanced_accuracy' : [difference, balanced_accuracy_score],
    'dif_recall' : [difference, recall_score],
    'dif_precision' : [difference, precision_score],
    'dif_fpr' : [difference, fpr_score],
    'dif_selection_rate' : [difference, selection_rate],
    'dif_statistical_parity' : statistical_parity_difference, 
    'dif_equal_opp' : equal_opportunity_difference, 
    'dif_avg_odds' : average_odds_difference, 
    'disparate_impacto_ratio' : disparate_impact_ratio
}

def concat_results(*, relative_dir='', file_format='.csv', sep=';'):
    ''' Função que concatena aqruivos do tipo .csv em um mesmo frame.
    '''
    
    # concatena o diretório atual com o relativo passado no parâmetro relative_dir
    dirname = os.path.join(os.getcwd(), relative_dir)
    
    # verifica se é um diretório válido
    if not os.path.exists(dirname):
        raise ValueError('Erro: O seguinte diretório não existe: %s' % (dirname))
    
    results = None
    # percorre a lista dos arquivos e sub-diretórios 
    for item in os.listdir(dirname):
        
        # se não for um arquivo passa para o próximo item da lista
        if not os.path.isfile(os.path.join(dirname, item)):
            continue
        
        # copia o dir do arquivo junto ao nome do arquivo
        file = os.path.join(dirname, item)                   
        # recupera a extensão do arquivo
        ext = os.path.splitext(file)[1]

        # se o arquivo for no formato desejado
        if ext == file_format:
            # concatena os arquivos
            results = pd.concat([results, pd.read_csv(file, sep=sep)])
    
    return results

def convert_index(l, privileged_group):
    ''' Função que converte o valor do index para numérico, necessário para utilizar o AIF360
    '''
    if l == privileged_group:
        return 1
    else:
        return 0

''' Vetoriza a função convert_index
'''
convert_index = np.vectorize(convert_index)


def kfold(clf, X, y, weights, k=10):
    ''' Função que realiza o kfold estratificado e retorna a média de medidas de desempenho
    '''
    
    # instancia o KFold estratificado (sem utilizar todos os parametros)
    kf = StratifiedKFold(n_splits=k)
    
    # inicia um DataFrame para salvar os resultados 
    # (colunas são as medidas que irão no relatório (neste caso só tem acurácia), 
    # as linhas é o número de iterações)
    results = pd.DataFrame(index=np.arange(k), columns=measures_columns)

    
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        
        # Separa os conjuntos
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Treina o classificador
        if weights is None:
            clf.fit(x_train, y_train['target'])
        else:
            clf.fit(x_train, y_train['target'], weights[train_index])
        # Realiza as predições
        y_predict = clf.predict(x_test)
        
        # Calcula as medidas de desempenho
        for name, measure in performance_measures.items():
            results.loc[i, name] = measure(y_test['target'], y_predict)
            
        # Calcula as medidas de fairness
        for name, measure in fairness_measures.items():
            # verifica se é uma lista (medidas que precisam de duas funções)
            if isinstance(measure, list):
                results.loc[i, name] =  abs(measure[0](measure[1], y_test, y_predict, prot_attr='group', priv_group=1))
            else:
                results.loc[i, name] = abs(measure(y_test, y_predict, prot_attr='group', priv_group=1))     
                    
            
    # retorna o resultado como sendo a média de todas iterações
    return results.mean()

class Experiment:
    
    def __init__(self, classifiers_settings, *, k=10, dataset_name, preprocessing_name, privileged_group):
        
        self.classifiers_settings = classifiers_settings
        self.k = k
        self.dataset_name = dataset_name
        self.preprocessing_name = preprocessing_name
        self.report = pd.DataFrame(columns=['dataset', 'preprocessing', 'clf_type', 'params'] + measures_columns)
        self.counter = 0
        self.privileged_group = privileged_group
        self.n_classifiers = self.__get_number_classifiers()
        
    
    def display(self, clf_type, n_clf_type, counter_by_clf):
        """ Função que mostra o progresso no jupyter

        Args:
            clf_name (str): nome do classificador
        """
        clear_output()
        print('(%s - %s) - Classificador %s (%d/%d) - Progresso Geral (%d/%d)' % 
              (self.dataset_name, self.preprocessing_name, clf_type, counter_by_clf, n_clf_type, self.counter,
               self.n_classifiers))
                  
    def __get_number_classifiers(self):
        """
        Recupera o número de classificadores no experimento para auxiliar no progresso do experimento
        """
        n_classifiers = 0
        for _, (_, param_grid) in self.classifiers_settings.items():
            grid = ParameterGrid(param_grid)
            for _ in grid:
                n_classifiers += 1
        return n_classifiers
    
    def export_report(self, relative_path='', complement_name=''):
        filename = 'rep_' + self.dataset_name + '_' + self.preprocessing_name + '_' + complement_name + '.csv'
        self.report.to_csv(relative_path + filename, sep=';', index=False)
        
    def execute(self, X, y, weights=None):
        
        # transforma os indices do y para ficar compatível com o AIF360
        y = pd.DataFrame(y, columns=['target'])
        
        if not weights is None:
            weights = np.array(weights)
        
        # verifica se o grupo privilegiado está contido nos indices (se não estiver gera exceção)
        if isinstance(X.index, pd.MultiIndex):
            if not self.privileged_group in list(map('/'.join, list(X.index))):
                raise ValueError('Erro: privileged_group (%s) não está contido em X' % (self.privileged_group))
            else:
                y.index = convert_index(list(map('/'.join, list(X.index))), self.privileged_group)
                y.index.names = ['group']
                
        else:
            if not self.privileged_group in list(X.index):
                raise ValueError('Erro: privileged_group (%s) não está contido em X' % (self.privileged_group))
            else:
                y.index = convert_index(list(X.index), self.privileged_group)
                y.index.names = ['group']   
        

        for clf_type, (Classifier, param_grid) in self.classifiers_settings.items():
            # função que executa todas as configurações de um determinado algoritmo de classificação
            grid = ParameterGrid(param_grid)
            # contador para ver o progresso no algoritmo de classificação em questão
            counter_by_clf = 0
            # imprime o status
            self.display(clf_type, len(grid), counter_by_clf)
            for params in grid: 
                # instancia o classificador com os novos parâmetros
                clf = Classifier(**params)
                
                # realiza o kfold
                result = kfold(clf, X, y, weights)
                
                # salva o resultado no relatório
                self.report.loc[self.counter] = [self.dataset_name, self.preprocessing_name, clf_type, str(params)] + list(result)

                
                # incrementa os contadores e atualiza o status
                self.counter += 1
                counter_by_clf += 1
                self.display(clf_type, len(grid), counter_by_clf)


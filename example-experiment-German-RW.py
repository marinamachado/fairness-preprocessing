import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import aif360
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset,BankDataset

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover,LFR,OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aif360.sklearn.metrics import statistical_parity_difference,disparate_impact_ratio,equal_opportunity_difference,average_odds_difference,theil_index
from numpy import array
from dataset_loader import datasets_loader


def get_priv_unpriv_att(dataset_orig):
    
    attr = dataset_orig.protected_attribute_names[0]
    idx = dataset_orig.protected_attribute_names.index(attr)
    privileged_groups =  [{attr:dataset_orig.privileged_protected_attributes[idx][0]}] 
    unprivileged_groups = [{attr:dataset_orig.unprivileged_protected_attributes[idx][0]}] 

    return privileged_groups,unprivileged_groups



# função que realiza o kfold estratificado e retorna a média de medidas de desempenho
def kfold(clf, X, y, k=10):
    
    # instancia o KFold estratificado (sem utilizar todos os parametros)
    kf = StratifiedKFold(n_splits=k)
    
    # inicia um DataFrame para salvar os resultados 
    # (colunas são as medidas que irão no relatório (neste caso só tem acurácia), 
    # as linhas é o número de iterações)
    results = pd.DataFrame(index=np.arange(k), columns=['accuracy','statistical_parity_difference','disparate_impact_ratio',
                                                        'equal_opportunity_difference','average_odds_difference','theil_index'])

    
    for i, (train_index, test_index) in enumerate(kf.split(np.zeros(len(y)), y)):
        
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        
        y_train = y[train_index]
        y_test = y[test_index]

            
            # neste caso a estrutura de dados é uma matriz pq to usando um dataset do sistema
            # quando for um Frame usar .loc ou .iloc
            
            # Treina o classificador
        clf.fit(X_train, y_train)
            
            # Realiza as predições
        y_predict = clf.predict(X_test)       


        test = pd.DataFrame()
        test['credit'] = y_test
        test.index = X_test['sex']
        test.index.names = ['sex']
        
        pred = pd.DataFrame()
        pred['credit'] = y_predict
        pred.index = X_test['sex']
        pred.index.names = ['sex']
        
        results.loc[i, 'accuracy'] = accuracy_score(y[test_index], y_predict)
        results.loc[i, 'statistical_parity_difference'] = (statistical_parity_difference(test,pred, prot_attr =['sex']))
        results.loc[i, 'disparate_impact_ratio'] = (disparate_impact_ratio(test,pred, prot_attr =['sex']))
        results.loc[i, 'equal_opportunity_difference'] = (equal_opportunity_difference(test,pred, prot_attr =['sex']))
        results.loc[i, 'average_odds_difference'] = (average_odds_difference(test,pred, prot_attr =['sex']))

          
    # retorna o resultado como sendo a média de todas iterações
    return results.mean()

# classe que organiza executa os experimentos
class Experiment:
    
    def __init__(self, classifiers_settings, k=10):
        self.classifiers_settings = classifiers_settings
        self.k = k
        self.report = pd.DataFrame(columns=['clf_type', 'params', 'nome','accuracy','statistical_parity_difference','disparate_impact_ratio',
                                           'equal_opportunity_difference','average_odds_difference'])
        self.it = 0
        
    def execute(self, X, y, name):

        for clf_type, (Classifier, param_grid) in self.classifiers_settings.items():
            print('Executando -- %s --' % (clf_type))
            
            # função que executa todas as configurações de um determinado algoritmo de classificação
            grid = ParameterGrid(param_grid)
            for params in grid: 
                # instancia o classificador com os novos parâmetros
                clf = Classifier(**params)
                # realiza o kfold
                result = kfold(clf, X, y)
                # salva o resultado no relatório
                        
                self.report.loc[self.it] = [clf_type, str(params),name,result[0],result[1],result[2],result[3],result[4]] 
                self.it += 1


def main():

    d  = datasets_loader()
    d.load_german_dataset()
    dataset_orig = d.dataset
                    
    X = dataset_orig.convert_to_dataframe()[0]
    y = X['credit']
    X = X.drop(columns = ['credit'],axis =1)


    # nesse formato o conjunto de modelos a serem testados 
    classifiers_settings = {
    #     'Random Forest': [RandomForestClassifier,
    #           {'n_estimators' : np.arange(120, 220, 20)}],
        'Decision Tree' : [DecisionTreeClassifier,
            {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random']}]
    }


    # instancia o experimento
    experiment = Experiment(classifiers_settings)


    experiment.execute(X, y,'German')
    experiment.report


    privileged_groups,unprivileged_groups = get_priv_unpriv_att(dataset_orig)


    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
    RW.fit(dataset_orig)
    dataset_rw = RW.transform(dataset_orig)

    X_rw = dataset_rw.convert_to_dataframe()[0]
    y_rw = X_rw['credit']
    X_rw = X_rw.drop(columns = ['credit'],axis =1)

    experiment.execute(X_rw, y_rw,'German - RW')

    experiment.report.to_csv('German-RW.csv')


if __name__ == "__main__":
    main()


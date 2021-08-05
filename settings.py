import numpy as np

# algoritmos de classificação
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

hidden_layer = [5, 8, 15, (5, 5), (10, 10), (5, 5, 5)]

classifiers_settings_test = {
    'Random Forest': [RandomForestClassifier,
          {'n_estimators' : np.arange(120, 220, 20)}],
    'Naive Bayes': [GaussianNB, {}],
    'Decision Tree' : [DecisionTreeClassifier,
          {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random']}],
    'XGBoost' : [XGBClassifier,
                {'objective': ['binary:logistic']}],
    'MLP' : [MLPClassifier,
            {'hidden_layer_sizes' : hidden_layer, 'max_iter': [5000]}]
}

classifiers_settings_eniac = {
    'Decision Tree' : [DecisionTreeClassifier,
                      {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random']}],
    'KNN' : [KNeighborsClassifier,
            {'n_neighbors' : [1, 3, 5, 7]}],
    'Logistic Regression' : [LogisticRegression,
                            {'C': [0.75, 1, 1.25]}],
    'MLP' : [MLPClassifier,
            {'hidden_layer_sizes' : hidden_layer, 'max_iter': [5000]}], 
    'Naive Bayes' : [GaussianNB, {}],
    'Naive Bayes' : [BernoulliNB,{}],
    'Random Forest' : [RandomForestClassifier,
                      {'n_estimators' : np.arange(100, 600, 100)}],
    'SVM' : [SVC,
            {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [2, 3, 4]}],
    'XGBoost' : [XGBClassifier,
                {'objective': ['binary:logistic']}]
}

classifiers_settings_eniac_weights = {
    'Decision Tree' : [DecisionTreeClassifier,
                      {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random']}],
    'Logistic Regression' : [LogisticRegression,
                            {'C': [0.75, 1, 1.25]}], 
    'Naive Bayes' : [GaussianNB, {}],
    'Naive Bayes' : [BernoulliNB,{}],
    'Random Forest' : [RandomForestClassifier,
                      {'n_estimators' : np.arange(100, 600, 100)}],
    'SVM' : [SVC,
            {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [2, 3, 4]}]

}

# aceita peso RF, DT, NB, SVM, LR
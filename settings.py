import numpy as np

# algoritmos de classificação
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

classifiers_settings_test = {
    'Random Forest': [RandomForestClassifier,
          {'n_estimators' : np.arange(120, 220, 20)}],
    'Decision Tree' : [DecisionTreeClassifier,
          {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random']}]
}


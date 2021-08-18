import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline 
import math

#Transformador que aplica o método de préprocessamento Massaging nos dados
class Massaging( BaseEstimator, TransformerMixin ):
    
    
    #Class Constructor 
    def __init__( self, dataset):
        self.dataset = dataset
        self.label = dataset.label_names[0]
        self.attr =  dataset.protected_attribute_names[0] # atributo protegido
        self.m = None # nro de modificações
        
        # array auxiliar para ajudar no processo, 
        # baseado na probabilidade de ser classificado como positivo
        self.diff = None  
        
        
    def calc_disc(self,X_train):
        
        qnt_priv_pos = len(X_train[(X_train[self.attr]==1) & (X_train[self.label]==1)])
        qnt_unpriv_pos = len(X_train[(X_train[self.attr]==0) & (X_train[self.label]==1)])

        qnt_priv = len(X_train[(X_train[self.attr]==1)])
        qnt_unpriv = len(X_train[(X_train[self.attr]==0)])

        disc = (qnt_priv_pos/qnt_priv) - (qnt_unpriv_pos/qnt_unpriv)
        m = math.ceil((disc * qnt_priv * qnt_unpriv)/len(X_train))
        
        return disc,m
    
    # X dataframe
    # y array-like
    def fit( self, X, y = None ):
        

        X_train = X.copy()
        y_train = y.copy()
    

        clf = GaussianNB()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_train)
    
        train = X_train.copy()
        train[self.label] = y_train
        train['positive_probability'] = probs[:,1]
    
    
        disc, self.m = self.calc_disc(train)
        self.diff = (train['positive_probability'] - np.float64(0.5))
        
        return self 
    
    # X dataframe
    # y array-like
    def transform(self, X, y = None ):
        
        
        new_dataset = X.copy()
        new_dataset[self.label] = y
        new_dataset['diff'] = self.diff
        
        new_dataset = new_dataset.sort_values('diff', ascending = False)
        index_mudar_p_1 = new_dataset[(new_dataset['diff'] < 0) & (new_dataset[self.attr]== 0)  & (new_dataset[self.label]== 0)].iloc[:self.m].index
        new_dataset.loc[index_mudar_p_1,self.label] = 1
    
        new_dataset = new_dataset.sort_values('diff', ascending = True)
        index_mudar_p_0 = new_dataset[(new_dataset['diff'] > 0) & (new_dataset[self.attr]==1)  & (new_dataset[self.label]==1)].iloc[:self.m].index
        new_dataset.loc[index_mudar_p_0,self.label] = 0
        disc,_ = self.calc_disc(new_dataset)
        
        
        return new_dataset.drop(columns=['diff']), disc

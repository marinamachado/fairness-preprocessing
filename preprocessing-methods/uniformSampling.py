
import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import math

#Transformador que aplica o método de préprocessamento Uniform Sampling dados
class UniformSampling( BaseEstimator, TransformerMixin ):
    
    
    #Class Constructor 
    def __init__( self,attr,label):
        self.attr = attr
        self.label = label
        self.grupos = {'DP':{'attr':0,'label':1},'FP':{'attr':1,'label':1},'DN':{'attr':0,'label':0},'FN':{'attr':1,'label':0}}
        self.lista = ['DP','DN','FP','FN'] 

    # X dataframe completo
    def fit( self, X, y = None ):
        
        tam_data = len(X)
        for li in self.lista:
            
            grupo = self.grupos[li]
            l = X[(X[self.attr] == grupo['attr']) & (X[self.label] == grupo['label'])]

            change = int(len(l) - (tam_data/4))
            
            grupo['to_change'] = change


        return self 
    
    # X dataframe completo
    def transform(self, X, y = None ):
        new_df = pd.DataFrame()

        for li in self.lista:

            grupo = self.grupos[li]
            l = X[(X[self.attr] == grupo['attr']) & (X[self.label] == grupo['label'])]
            
            change = grupo['to_change']
            
            if(change < 0):
                aux = change * -1

                while(aux > 0):
                    size = min(len(l),aux)
                    aux = aux - size
                    new = l.sample(n = size)
                    l = pd.concat([l,new])
            elif(change > 0):
                new_size = len(l) - change
                l = l.sample(n = new_size)

            new_df = pd.concat([new_df,l])
            grupo['valor_final'] = len(l)
            
        return new_df,self.grupos
        
        
        


import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB
import math

#Transformador que aplica o método de préprocessamento Preferential Sampling dados
class PreferentialSampling( BaseEstimator, TransformerMixin ):
    
    
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

#             change = int(len(l) - (tam_data/4))
            change = math.ceil(len(l)) - math.ceil((tam_data/4))

            
            grupo['to_change'] = change


        return self 
    
    # X dataframe completo
    def transform(self, X, y = None ):
        new_df = X
        

        
        for li in self.lista:

            grupo = self.grupos[li]
            l = X[(X[self.attr] == grupo['attr']) & (X[self.label] == grupo['label'])]
            
            change = grupo['to_change']
            
            print(li)
            
            if(change < 0):
                
                aux = change * -1
                    
                while(aux > 0):
                   

                    size = min(len(l),aux)
                    aux = aux - size

                    y = new_df[self.label]

                    y = y.ravel()
                    X = new_df.drop(columns = [self.label],axis =1)

                    clf = GaussianNB()
                    clf.fit(X, y)

                    probs = clf.predict_proba(X)

                    X['positive_probability'] = probs[:,1]

                    X[self.label] = y
                    
                    if(grupo['label'] == 1):
                        X = X.sort_values('positive_probability', ascending = True)
                        new_lines = X[(X[self.attr]==grupo['attr'])  & (X[self.label]==grupo['label'])].iloc[:size].index
                        new = X.loc[new_lines]
                        new = new.drop(columns = ['positive_probability'],axis =1)
                        new_df = pd.concat([new_df ,new])

                    elif(grupo['label'] == 0):
                        X = X.sort_values('positive_probability', ascending = False)
                        new_lines = X[(X[self.attr]==grupo['attr'])  & (X[self.label]==grupo['label'])].iloc[:size].index
                        new = X.loc[new_lines]
                        new = new.drop(columns = ['positive_probability'],axis =1)
                        new_df = pd.concat([new_df ,new])

                    new_df = new_df.reset_index(drop = True)


            elif(change > 0):
                new_size = change
                y = new_df[self.label]

                y = y.ravel()
                X = new_df.drop(columns = [self.label],axis =1)

                clf = GaussianNB()
                clf.fit(X, y)


                probs = clf.predict_proba(X)

                X['positive_probability'] = probs[:,1]

                X[self.label] = y


                if(grupo['label'] == 1):
                    X = X.sort_values('positive_probability', ascending = True)
                    new_lines = X[(X[self.attr]==li['attr'])  & (X[self.label]==li['label'])].iloc[:new_size].index
                    new_df = new_df.drop(new_lines)

                elif(grupo['label'] == 0):
                    X = X.sort_values('positive_probability', ascending = False)
                    new_lines = X[(X[self.attr]==grupo['attr'])  & (X[self.label]==grupo['label'])].iloc[:new_size].index
                    new_df = new_df.drop(new_lines)
                new_df = new_df.reset_index(drop = True)
                
            l = new_df[(new_df[self.attr]==grupo['attr']) & (new_df[self.label]==grupo['label'])]
            print(len(l))
                
            
        return new_df
        

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 03:55:58 2022

@author: Hussein Aly
"""
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split



class LoadUCI:
    
    def __init__(self , dataset_name , data_loc = "data", test_size=0.2 , random_state=41) : 
        self.data_loc = data_loc
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.test_size = test_size
        self.dataset_maps = {
            "abalone" : self.process_abalone,
            "bank" : self.process_bank , 
            "musk" : self.process_musk , 
            }

    
    def data(self) :    
        return self.dataset_maps[self.dataset_name]()   
        

    def process_abalone(self): 
        data = pd.read_csv(f"{self.data_loc}/abalone/abalone.csv")
        X = data.iloc[:,1:].values
        y = data.iloc[:,0].values
    
        ohe = OneHotEncoder(sparse=False)
        y= ohe.fit_transform(y.reshape(-1 , 1))
        X_train , y_train , X_test , y_test = train_test_split(X , y , test_size=self.test_size , random_state=self.random_state)
        
        return X_train , y_train , X_test , y_test
    
    
    def process_bank(self):
        data = pd.read_csv(f"{self.data_loc}/bank/bank.csv", delimiter=";")
        y = data['y'].values
        data.drop(columns='y' , inplace=True)
        cat_index = data.dtypes == "object"
        cat_data = data.iloc[:,np.where(cat_index)[0]]
        numric_data = data.iloc[:,np.where(cat_index == False)[0]]
        ohe = OneHotEncoder(sparse=False)
        one_hot = ohe.fit_transform(cat_data.values)
        combined = np.concatenate((one_hot , numric_data.values) , axis=1)
        y = ohe.fit_transform(y.reshape(-1 , 1))
        
        X_train , y_train , X_test , y_test = train_test_split(combined , y ,
                                                               test_size=self.test_size ,
                                                               random_state=self.random_state)
        return X_train , y_train , X_test , y_test
        
    
    def process_musk(self): 
        data = pd.read_csv(f"{self.data_loc}/musk/clean2.data", delimiter="," , header=None)
        X = data.iloc[:,2:-1].values
        y =  data.iloc[:,-1].values
        ohe = OneHotEncoder(sparse=False)
        y = ohe.fit_transform(y.reshape(-1 , 1))
        X_train , y_train , X_test , y_test = train_test_split(X , y ,
                                                               test_size=self.test_size ,
                                                               random_state=self.random_state)
        return X_train , y_train , X_test , y_test

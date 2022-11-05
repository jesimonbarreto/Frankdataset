from os import execv
import pickle
import sys
from abc import ABCMeta, abstractmethod
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from scipy.io import loadmat
import random
import csv, glob


#Class name pattern - use gerund with the first capital letter
#Examples: Walking, Standing, Upstairs
#TODO create a file with all activities already used 
class Model(metaclass=ABCMeta):
    def __init__(self):
        self.encoder = None
    
    def code_y(self, y):
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder.fit(y.reshape(-1,1))
        return self.encoder.transform(y.reshape(-1,1))
    
    def uncode_y(self, y):
        if self.encoder!=None:
            y = self.encoder.inverse_transform(y.reshape(-1,1))
        return y
    
    @abstractmethod
    def model_use(self, data_input_file):
        pass
    @abstractmethod
    def get_details(self):
        pass


class ManagerModels(object):
    def __init__(self, models):
        self.models = models
    
    def run_models(self, dirmodels):
        result = {} 
        data_input_files = glob.glob(dirmodels)
        for model in self.models:
            for file_ in data_input_files:
                try:
                    name_data = file_.split('/')[-1]
                    res = model.model_use(file_)
                    result[model.get_details()+'-'+name_data].append(res)
                except:
                    print('\n\nModel: '+ model.get_details())
                    print('dataset: '+ file_)
        return result
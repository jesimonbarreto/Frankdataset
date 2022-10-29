import pickle
import sys
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from scipy.io import loadmat
import random
import csv


#Class name pattern - use gerund with the first capital letter
#Examples: Walking, Standing, Upstairs
#TODO create a file with all activities already used 
class Model(metaclass=ABCMeta):
    @abstractmethod
    def model_use(self, dir_files):
        pass


class ManagerModels(object):
    def __init__(self, models):
        self.models = models
    
    def run_models(self, dirmodels):
        for model in self.models:
            model.model_use(dirmodels)
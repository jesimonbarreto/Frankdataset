import numpy as np
import csv, sys, glob, os
import pandas as pd
from enum import Enum
from Dataset.Pamap2 import PAMAP2
from Dataset.Pamap2 import SignalsPAMAP2 as sp
from Process.Manager import preprocess_datasets
from Process.Protocol import Loso
from Models import ManagerModels
from Models import *
np.random.seed(12227)

if __name__ == "__main__":
    file_pm = dir_base+'PAMAP2_Dataset/' 
    dir_datasets = dir_base+'preprocessed/'
    dir_save_file = dir_base+'generated/'
    freq = 20
    time_wd=5
    type_interp= 'cubic'
    norm = False
    replace = False
    
    p2 = PAMAP2('Pamap2', file_pm, dir_datasets)

    sig_pm = [sp.acc1_dominant_wrist_X, sp.acc1_dominant_wrist_Y, sp.acc1_dominant_wrist_Z,
              sp.gyr_dominant_wrist_X, sp.gyr_dominant_wrist_Y, sp.gyr_dominant_wrist_Z   
            ]
    p2.set_signals_use(sig_pm)
    
    datasets = [p2]
    preprocess_datasets(datasets, replace=replace)            
    generate_ev = Loso(datasets, overlapping = 0.0, transforms=[]
                time_wd=time_wd,
                type_interp=tip,
                replace=replace)
    file_generate = generate_ev.simple_generate(dir_save_file, new_freq = freq)
    
    c = Catal()
    models = [c]
    result = ManagerModels(models).run_models(file_generate)

    print(result)

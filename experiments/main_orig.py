import numpy as np
import csv, sys, glob, os
import pandas as pd
from enum import Enum
from Dataset.Wisdm import Wisdm
from Dataset.Wisdm import SignalsWisdm as sw
from Dataset.Utdmhad1 import UTDMHAD1
from Dataset.Utdmhad1 import SignalsUtdmhad1 as su
from Dataset.Mhealth import MHEALTH
from Dataset.Mhealth import SignalsMHEALTH as sm
from Dataset.Pamap2 import PAMAP2
from Dataset.Pamap2 import SignalsPAMAP2 as sp
from Dataset.Uschad import USCHAD
from Dataset.Uschad import SignalsUSCHAD as sus
from Process.Manager import preprocess_datasets
from Dataset.Nonsense19 import NonSense
from Process.Protocol import Loso
from Models import ManagerModels
from Models import *
from Signal.Transform import dataset_to_datasets
np.random.seed(12227)

def verifydir(dirlist):
    for directory in dirlist:
        if not os.path.exists(directory):
            os.makedirs(directory)

if __name__ == "__main__":
    
    #list_name_file = ['../','../']
    if len(sys.argv) > 10:
        file_wisdm = sys.argv[1]
        dir_datasets = sys.argv[2]
        dir_save_file = sys.argv[3]
    else:
        #dir_base = '/home/jesimon/Documents/Project_sensors_dataset/'#'./data/'
        dir_base = sys.argv[1] #'./data/'
        dir_data = 'dataset/'
        file_wisdm = dir_base+dir_data+'wisdm/debug.txt'
        file_utd1 = dir_base+dir_data+'Inertial/'
        file_pm = dir_base+dir_data+'PAMAP2_Dataset/'
        file_mh = dir_base+dir_data+'MHEALTHDATASET/'
        file_us = dir_base+dir_data+'USC-HAD/'
        dir_datasets = dir_base+'preprocessed/'
        dir_save_file = dir_base+'generated/'
        dir_save_file_all = dir_base+'generatedAll/'
        dataset_exp = sys.argv[2].split('-')
        model_exp = sys.argv[3].split('-')
        freqs = [20,50,100]
        time_wd=5
        type_interp= ['slinear', 'quadratic', 'cubic']
        norm = False
        replace = False
    
    dirlist = [dir_datasets,dir_save_file,dir_save_file_all]
    verifydir(dirlist)
    #Creating datasets
    #name, dir_dataset, dir_save, freq = 100, trial_per_file=100000
    #w = Wisdm('Wisdm', file_wisdm, dir_datasets, freq = 20, trials_per_file = 1000000)
    #utd = UTDMHAD1('UTD1', file_utd1, dir_datasets, freq = 50, trials_per_file = 1000000)
    p2 = PAMAP2('Pamap2', file_pm, dir_datasets, freq = 100, trials_per_file = 10000)
    #mh = MHEALTH('Mhealth', file_mh, dir_datasets, freq = 100, trials_per_file = 10000)
    us = USCHAD('Uschad', file_us, dir_datasets, freq = 100, trials_per_file = 10000)

    #Define signals of each dataset
    #sig_w = [sw.acc_front_pants_pocket_X, sw.acc_front_pants_pocket_Y, sw.acc_front_pants_pocket_Z]
    #w.set_signals_use(sig_w)

    #sig_utd = [su.acc_right_wrist_X, su.acc_right_wrist_Y, su.acc_right_wrist_Z,
    #            su.gyr_right_wrist_X, su.gyr_right_wrist_Y, su.gyr_right_wrist_Z]
    #utd.set_signals_use(sig_utd)

    sig_pm = [sp.acc1_dominant_wrist_X, sp.acc1_dominant_wrist_Y, sp.acc1_dominant_wrist_Z,
            sp.gyr_dominant_wrist_X, sp.gyr_dominant_wrist_Y, sp.gyr_dominant_wrist_Z   
            ]
    p2.set_signals_use(sig_pm)
    
    sig_us = [sus.acc_front_right_hip_X,sus.acc_front_right_hip_Y,sus.acc_front_right_hip_Z,
                sus.gyr_front_right_hip_X, sus.gyr_front_right_hip_Y, sus.gyr_front_right_hip_Z
            ]
    us.set_signals_use(sig_us)

    #sig_mh = [sm.acc_right_lower_arm_X, sm.acc_right_lower_arm_Y, sm.acc_right_lower_arm_Z,
    #            sm.gyr_right_lower_arm_X, sm.gyr_right_lower_arm_Y, sm.gyr_right_lower_arm_Z
    #        ]
    #mh.set_signals_use(sig_mh)
    
    #list datasets
    datasets = []
    for de in dataset_exp:
        if de == 'PAMA':
            datasets.append(p2)
        elif de == 'USC':
            datasets.append(us)
    
    
    files_run = []
    for freq in freqs:  
        for tip in type_interp:
            for dataset in datasets:
                #preprocessing
                preprocess_datasets([dataset], replace=replace)
                #Creating Loso evaluate generating
                generate_ev = Loso([dataset], overlapping = 0.0,
                            time_wd=time_wd,
                            type_interp=tip,
                            replace=replace)
                #Save name of dataset in variable y
                #generate_ev.set_name_act()
                #function to save information e data
                #files = glob.glob(dir_datasets+'*.pkl')
                file_ = generate_ev.simple_generate(dir_save_file, new_freq = freq)
                files_run.append(file_)
        #try:
        #dataset_to_datasets(join, dir_save_file_all, replace = True, norm = norm)
        #except:
        #    print('erro')
        #    print(join)
    
    print('CLASSIFICATION')
    #colocar tudo interno no costrutor e so precisar passar string com os codigos
    #cl = Catal()
    sn = simpleNet()
    #cx = ChenXue()
    #hc = HaChoi()
    #ht = Haetal()
    #jy = JiangYin()
    kz = Kwapisz()
    #pl = Panwaretal()
    sen = Sena()

    
    models = []
    
    for me in model_exp:
        if me == 'Sen':
            models.append(sen)
        elif me == 'Jor':
            models.append(sn)
        elif me == 'Kwa':
            models.append(kz)

    mm = ManagerModels(models)
    result = mm.run_models(files_run)
    print(result)

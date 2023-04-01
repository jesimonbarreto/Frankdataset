import numpy as np
import csv, sys, glob, os
import pandas as pd
from enum import Enum
from Dataset.Pamap2 import PAMAP2
from Dataset.Pamap2 import SignalsPAMAP2 as sp
from Dataset.Pamap2 import SignalsPAMAP2 as sp
from Dataset.Mhealth import SignalsMHEALTH as sm
from Dataset.Mhealth import MHEALTH
from Dataset.Uschad import SignalsUSCHAD as su
from Dataset.Uschad import USCHAD
from Dataset.Wharf import SignalsWharf as sw
from Dataset.Wharf import WHARF
from Dataset.Wisdm import SignalsWisdm as swi
from Dataset.Wisdm import Wisdm
from Dataset.Utdmhad2 import SignalsUtdmhad2 as su
from Dataset.Utdmhad2 import UTDMHAD2
from Process.Manager import preprocess_datasets
from Process.Protocol import Loso
from Models import ManagerModels
from Models import *
np.random.seed(12227)

def process_dataset(name,
                    dir_data,
                    dir_system,
                    dir_save,
                    freq,
                    new_freq,
                    sensors,
                    replace,
                    overlapping,
                    transforms,
                    time_wd,
                    type_interp,
                    select_activities
                    ):
    
    datasets={
        'Pamap2':PAMAP2,
        'Mhealth':MHEALTH,
        'Uschad':USCHAD,
        'Wharf':WHARF,
        'Wisdm':Wisdm,
        'Utdmhad2':UTDMHAD2
    }

    cod_sensors={
        'Pamap2':sp,
        'Mhealth':sm,
        'Uschad':su,
        'Wharf':sw,
        'Wisdm':swi,
        'Utdmhad2':su
    }

    data = datasets[name](name, dir_data, dir_system, freq=freq)  
    sensors_cod = []
    for s in sensors:
        vs = getattr(cod_sensors[name],s)
        sensors_cod.append(vs)
    data.set_signals_use(sensors_cod)
    print('\n\nProcessing Dataset: '+name)
    preprocess_datasets([data], replace=replace)            
    generate_ev = Loso([data], 
                overlapping=overlapping,
                transforms=transforms,
                time_wd=time_wd,
                type_interp=type_interp,
                replace=replace
            )
    generate_ev.set_select_activity(select_activities)
    file_generate = generate_ev.simple_generate(dir_save, new_freq=new_freq)
    print(file_generate)

def run_models():
    models_={
        'catal':Catal
    }
    
    c = Catal()
    models = [c]
    result = ManagerModels(models).run_models(file_generate)

    print(result)
import os
import numpy as np
from .Datasets import Dataset
from enum import Enum
from tqdm import tqdm

class SignalsWharf(Enum):
    acc_right_wrist_X = 0
    acc_right_wrist_Y = 1
    acc_right_wrist_Z = 2


actNameWHARF = {
    1:  'BrushTeeth',
    2:  'ClimbingStairs',
    3:  'CombHair',
    4:  'DescendingStairs',
    5:  'Drinking',
    6:  'Eating',
    7:  'EatingSoup',
    8:  'GetupBed',
    9:  'LiedownBed',
    10: 'PourWater',
    11: 'Sitting',
    12: 'StandupChair',
    13: 'Walking',
    14: 'UseTelephone'
}


class WHARF(Dataset):
    def print_info(self):
        return """
                device: IMU
                frequency: 32Hz
                positions: right wrist 
                sensors: acc
                """

    def preprocess(self):
        txt_files = []
        for root, dirs, files in os.walk(self.dir_dataset):
            if len(dirs) == 0:
                txt_files.extend([os.path.join(root, f) for f in files if '.txt' in f])

        print(len(txt_files))

        for trial_id, filepath in tqdm((txt_files)):
            filename = filepath.split(os.sep)[-1]
            act, subject = filename.split('-')[-2:]
            act = act.replace("_", " ")
            subject = subject.replace('.txt', '')
            with open(filepath, 'r') as inp:
                instances = [list(map(float, line.split())) for line in inp.read().splitlines()]

            trial = []
            for instance in instances:
                trial.append(instance)

            trial = np.array(trial)
            data = []
            for d in self.signals_use:
                data.append(trial[:,d.value])
            trial = np.column_stack(data).astype('float64')
            self.add_info_data(act, subject, trial_id, trial, self.dir_save)
            #print('file_name:[{}] s:[{}]'.format(filepath, subject))

        self.save_data(self.dir_save)




import os
from scipy.io import loadmat
from Dataset.Datasets import Dataset
from enum import Enum
import numpy as np
from tqdm import tqdm


class SignalsUSCHAD(Enum):
    acc_front_right_hip_X = 0
    acc_front_right_hip_Y = 1
    acc_front_right_hip_Z = 2
    gyr_front_right_hip_X = 3
    gyr_front_right_hip_Y = 4
    gyr_front_right_hip_Z = 5


actNameUSCHAD = {
    1:  'Walking',
    2:  'WalkingLeft',
    3:  'WalkingRight',
    4:  'WalkingUp',
    5:  'WalkingDown',
    6:  'Running',
    7:  'Jumping',
    8:  'Sitting',
    9:  'Standing',
    10: 'Sleeping',
    11: 'ElevatorUp',
    12: 'ElevatorDown',
}


class USCHAD(Dataset):
    def print_info(self):
        return """
                device: IMU
                frequency: 100Hz
                positions: front-right-hip
                sensors: acc and gyr
                """

    def fix_name_act(self, act):
        if "running" in act:
            act = act.replace("running", "run")
        if "jumping" in act:
            act = act.replace("jumping", "jump")
        if "sitting" in act:
            act = act.replace("sitting", "sit")
        if "standing" in act:
            act = act.replace("standing", "stand")
        if "downstairs" in act:
            act = act.replace("downstairs", "down")
        if "walking" in act:
            act = act.replace("walking", "walk")
        if "upstairs" in act:
            act = act.replace("upstairs", "up")
        return act

    def preprocess(self):
        mat_files = []
        for root, dirs, files in os.walk(self.dir_dataset):
            if len(dirs) == 0:
                mat_files.extend([os.path.join(root, f) for f in files])

        for filepath in tqdm(mat_files)):
            mat_file = loadmat(filepath)
            act = mat_file['activity'][0]
            subject = int(mat_file['subject'][0])
            trial_id = int(mat_file['trial'][0])
            trial_data = mat_file['sensor_readings'].astype('float64')

            data = []
            for d in self.signals_use:
                data.append(trial_data[:, d.value])
            trial = np.column_stack(data).astype('float64')
            act = act.replace("-", " ")
            self.add_info_data(act, subject, trial_id, trial, self.dir_save)

        self.save_data(self.dir_save)

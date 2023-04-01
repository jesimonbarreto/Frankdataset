import sys
import json
import argparse
from FrankDataset import process_dataset
from FrankDataset import run_models

def main(config_name_file):
    
    with open(config_name_file) as user_file:
        file_contents = user_file.read()
    configs = json.loads(file_contents)
    
    for config in configs:
        name = config['name']
        dir_data = config['dir_data']
        dir_system = config['dir_system']
        dir_save = config['dir_save']
        freq = config['freq']
        new_freq = config['new_freq']
        time_wd = config['time_wd']
        type_interp = config['type_interp']
        norm = config['norm']
        replace = config['replace']
        overlapping = config['overlapping']
        transforms = config['transforms']
        sensors = config['sensors']
        select_activities = config['select_activities']

        process_dataset(name,
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
        )
    
if __name__ == "__main__":
    '''parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config file used to execution', required=True)
    args = parser.parse_args()

    if args.config.split('.')[-1] != 'json':
        print('Erro, the system needs a json file!')
        sys.exit()
    
    main(args.config)'''
    main('input_frankdataset.json')
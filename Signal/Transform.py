import numpy as np                  # for working with tensors outside the network
import pandas as pd                 # for data reading and writing
import math, os
import scipy.stats as st
from scipy.interpolate import interp1d
#from fastpip import pip
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def sampling_rate(data, rate_reduc):
    number_samp = int(data[0].shape[-2])
    samples_slct = list(range(0,number_samp,rate_reduc))
    new_data = [data[0][:,:,samples_slct,:]]
    return new_data

#recebe uma lista de samples 
def interpolate_sensors(samples, type_interp, n_samples, plot=False):
    samples = np.array(samples)
    shape = samples.shape
    if len(shape) > 3:
        samples = np.squeeze(samples)
        shape = samples.shape
    new_samples = []
    for smp in samples:
        s = []
        for sig in range(shape[-1]):
            size = int(smp.shape[0])
            x = np.linspace(0, size-1, num=size, endpoint=True)
            y = smp[:,sig]
            f = interp1d(x, y, kind=type_interp)
            x_new = np.linspace(0, size-1, num=n_samples, endpoint=True)
            y_new = f(x_new)
            s.append(y_new.reshape(-1,1))
            if plot:
                plt.plot(x, y, '--', x_new, y_new, 'o')
                plt.xticks(x_new)
                plt.show()
        s = np.concatenate(s,axis=-1)
        new_samples.append(s)
    #new_samples = np.array(new_samples)
    return new_samples

def pip_sample(data, n_samples):
    number_samp = int(data[0].shape[-2])
    number_s = int(data[0].shape[0])
    final = []
    for samp in data[0]:
        samp = samp[0]
        x_pip = []
        y_pip = []
        z_pip = []
        for j in range(number_samp):
            x_pip.append((j,samp[:,0][j]))
            y_pip.append((j,samp[:,1][j]))
            z_pip.append((j,samp[:,2][j]))
        x_pip = pip(x_pip,n_samples)
        y_pip = pip(y_pip,n_samples)
        z_pip = pip(z_pip,n_samples)
        x_pip = np.array(list(zip(*x_pip))[1]).reshape(-1,1)
        y_pip = np.array(list(zip(*y_pip))[1]).reshape(-1,1)
        z_pip = np.array(list(zip(*z_pip))[1]).reshape(-1,1)
        final.append(np.concatenate((x_pip,y_pip,z_pip),axis=1))

    data_final = [np.array(final).reshape(number_s,1,n_samples,3)]
    return data_final


def select_features(X, y, d_act):
    y = np.argmax(y, axis=1)
    X_new = []
    y_new = []
    for idx,act in enumerate(d_act):
        ind = np.where(y==act)
        if idx == 0:
            X_new = X[ind]
            y_new = np.array([idx]*len(ind[0]))
        else:
            X_new = np.concatenate((X_new,X[ind]),axis=0)
            y_new = np.concatenate((y_new,np.array([idx]*len(ind[0]))),axis=0)

    y_new = to_categorical(y_new)
    return X_new, y_new

#debug - generate data -  classify
def dataset_to_datasets(datasets_names, dir_save, replace = False, norm = False, just_first=True):
    
    for i in range(len(datasets_names)):
        name = os.path.join(dir_save, datasets_names[i].split('/')[-1].split('.')[0]+'all')
        if not replace and os.path.exists(name+'.npz'):
            print('File '+ name+'.npz exist')
            continue
        name_list = datasets_names.copy()
        tmp = np.load(name_list[i], allow_pickle=True)
        X = tmp['X']
        y = tmp['y']
        folds = tmp['folds']
        len_x = len(X)
        del name_list[i]
        #define normalization
        if norm:
            scaler = MinMaxScaler()
            shp = X.shape
            X = scaler.fit_transform(X.reshape(shp[0], shp[1]*shp[2]*shp[3]))
            X = X.reshape(shp)
        for data_name in name_list:
            tmp = np.load(data_name, allow_pickle=True)
            X_ = tmp['X']
            y_ = tmp['y']
            #define normalization
            if norm:
                shp = X_.shape
                scaler = MinMaxScaler()
                X_ = scaler.fit_transform(X_.reshape(shp[0], shp[1]*shp[2]*shp[3]))
                X_ = X_.reshape(shp)
            X = np.concatenate([X,X_])
            y = np.concatenate([y,y_])
            for j in range(0, len(folds)):
                ##colocar todos os dados do dataset X_ no treino de folds...
                n = np.concatenate([folds[j][0], [*range(len_x, len_x + len(X_))]])
                folds[j][0] = n
            len_x += len(X_)
        name = os.path.join(dir_save, datasets_names[i].split('/')[-1].split('.')[0]+'all')
        np.savez_compressed(name, X=X, y=y, folds=folds)
        print('Create file '+ name+'.npz')
        if just_first:
            break
    #return name+'.npz'



if __name__ == '__main__':
    np.random.seed(12227)
    
    tp = 'cubic'
    data_input_file = '/home/jesimon/Documents/others/sensors2017/SavedFeatures/LOSO/UTD-MHAD2_1s.npz'


    print('Dataset {};'.format(data_input_file))
    print('info: Type amp {}'.format(tp))
    tmp = np.load(data_input_file, allow_pickle=True)

    X = tmp['X']
    y = tmp['y']
    folds = tmp['folds']
    dataset_name = data_input_file.split('/')[-1]
    print('Loaded {} with success'.format(dataset_name))

    data = interpolate_sensors(X, tp, 100)


import numpy as np

from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys
import custom_model as cm
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import sys
#import cv2
from keras.layers import Input, Dense
from keras.models import Model
import keras, copy
from keras import backend as K
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

K.set_image_data_format('channels_first')



def _stream(inp, n_filters, kernel, n_classes):
    hidden = keras.layers.Conv2D(
        filters= 64, kernel_size=(5,1), activation='relu', kernel_initializer='glorot_normal',
        padding='same')(inp)
    hidden = keras.layers.MaxPooling2D(pool_size=(2, 1))(hidden)
    hidden = keras.layers.Conv2D(
        filters= 64, kernel_size=(5,1), activation='relu', kernel_initializer='glorot_normal',
        padding='same')(hidden)
    hidden = keras.layers.Conv2D(
        filters= 64, kernel_size=(5,1), activation='relu', kernel_initializer='glorot_normal',
        padding='same')(hidden)
    hidden = keras.layers.MaxPooling2D(pool_size=(2, 1))(hidden)

    hidden = keras.layers.Flatten()(hidden)

    activation_dense = 'relu'
    kernel_init_dense = 'glorot_normal'
    n_neurons = 512
    dropout_rate = 0.1

    # -------------- output layer --------------------------------------------

    out = keras.layers.Dense(n_neurons)(hidden)
    out = keras.layers.Dropout(0.5)(out)

    return out

def activation_layer(activation, tensor):
    """Activation layer"""
    import keras
    if activation == 'selu':
        hidden = keras.layers.core.Activation(activation)(tensor)
        hidden = keras.layers.normalization.BatchNormalization()(hidden)
    else:
        hidden = keras.layers.core.Activation(activation)(tensor)
        hidden = keras.layers.normalization.BatchNormalization()(hidden)
    return hidden


def _kernelmlfusion(n_classes, inputs, kernel_pool):
    width = (16, 32)

    streams_models = []
    for inp in inputs:
            streams_models.append(_stream(inp, width, kernel_pool, n_classes))

    if len(kernel_pool) > 1:
        concat = keras.layers.concatenate(streams_models, axis=-1)
    else:
        concat = streams_models[0]

    n_neurons = 512

    hidden = keras.layers.Dense(n_neurons)(concat)
    hidden = keras.layers.Dropout(0.5)(hidden)
    # -------------- output layer --------------------------------------------

    hidden = keras.layers.Dense(n_classes)(hidden)
    out = keras.layers.core.Activation('softmax')(hidden)

    # -------------- model buider  --------------------------------------------
    model = keras.models.Model(inputs=inputs, outputs=out)
    opt = SGD(lr=0.00001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model

if __name__ == '__main__':
    np.random.seed(12227)

    batch_size = 10

    if len(sys.argv) > 2:
        data_input_file = sys.argv[1]
        batch = int(sys.argv[2])
    else:
        data_input_file = '/home/jesimon/Documents/sensors2017/SavedFeatures/LOSO/UTD-MHAD2_1s.npz'

    #mode:
    #for sensor-wise -> 'sensor';
    #for device-wise -> 'device';
    #for normal -> normal. 
    mode = 'sensor'

    tmp = np.load(data_input_file, allow_pickle=True)

    X = tmp['X']
    y = tmp['y']
    folds = tmp['folds']
    dataset_name = data_input_file.split('/')[-1]

    data = []

    if dataset_name == 'MHEALTH.npz':
        if mode == 'device':
            sensor1 = np.concatenate((X[:, :, :, 0:3],X[:, :, :, 8:11], X[:, :, :,11:14]), axis=-1)
            sensor2 = np.concatenate((X[:, :, :, 5:8],X[:, :, :, 17:20], X[:, :, :, 20:23]), axis=-1)
            sensor3 = X[:, :, :, 14:17]
            data.append(sensor1)
            data.append(sensor2)
            data.append(sensor3)
        elif mode == 'sensor' :
            acc = np.concatenate((X[:, :, :, 0:3],X[:, :, :, 5:8], X[:, :, :, 14:17]), axis=-1)
            gyr = np.concatenate((X[:, :, :, 8:11],X[:, :, :, 17:20]), axis=-1)
            mag = np.concatenate((X[:, :, :, 11:14],X[:, :, :, 20:23]), axis=-1)
    
    n_class = y.shape[1]

    avg_acc = []
    avg_recall = []
    avg_f1 = []

    pool = [(1, 5), (1, 5)]

    print('Rueda et al. 2018 {}'.format(data_input_file))

    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]
        X_test = []
        X_train = []

        for x in data:
            X_test.append(x[test_idx])
            X_train.append(x[train_idx])

        inputs = []

        for x in data:
            inputs.append(keras.layers.Input((x.shape[1], x.shape[2], x.shape[3])))

        model = _kernelmlfusion(n_class, inputs, pool)

        sys.stdout.flush()
        model.fit(X_train, y[train_idx], batch, cm.n_ep, verbose=0,
                   callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)],
                   validation_data=(X_train, y[train_idx]))

        test_idx = folds[i][1]


        y_pred = model.predict(X_test)

        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y[test_idx], axis=1)

        acc_fold = accuracy_score(y_true, y_pred)
        avg_acc.append(acc_fold)

        recall_fold = recall_score(y_true, y_pred, average='macro')
        avg_recall.append(recall_fold)

        f1_fold = f1_score(y_true, y_pred, average='macro')
        avg_f1.append(f1_fold)

        print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold, i))
        print('______________________________________________________')

    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
    ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))
    print('Mean Accuracy {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
    print('Mean Recall {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
    print('Mean F1 {:.4f}|[{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))

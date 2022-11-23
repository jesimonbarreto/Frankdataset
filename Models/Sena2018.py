import numpy as np

from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys

import custom_model as cm
#import cv2
from keras.models import model_from_json
import numpy
import os
import keras
from keras import backend as K
from keras.models import load_model

K.set_image_data_format('channels_first')

def _stream(inp, n_filters, kernel, n_classes):
    hidden = keras.layers.Conv2D(
        filters=n_filters[0], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal',
        padding='same')(inp)
    hidden = keras.layers.MaxPooling2D(pool_size=(2, 1))(hidden)
    hidden = keras.layers.Conv2D(
        filters=n_filters[1], kernel_size=kernel, activation='relu', kernel_initializer='glorot_normal',
        padding='same')(hidden)
    hidden = keras.layers.MaxPooling2D(pool_size=(2, 1))(hidden)

    hidden = keras.layers.Flatten()(hidden)

    activation_dense = 'selu'
    kernel_init_dense = 'glorot_normal'
    n_neurons = 200
    dropout_rate = 0.1

    # -------------- second hidden FC layer --------------------------------------------
    if kernel_init_dense == "":
        hidden = keras.layers.Dense(n_neurons)(hidden)
    else:
        hidden = keras.layers.Dense(n_neurons, kernel_initializer=kernel_init_dense)(hidden)

    hidden = activation_layer(activation_dense, dropout_rate, hidden)

    # -------------- output layer --------------------------------------------

    hidden = keras.layers.Dense(n_classes)(hidden)
    out = keras.layers.core.Activation('softmax')(hidden)

    return out

    # pylint: disable=R0201


def activation_layer(activation, dropout_rate, tensor):
    """Activation layer"""
    import keras
    if activation == 'selu':
        hidden = keras.layers.core.Activation(activation)(tensor)
        hidden = keras.layers.normalization.BatchNormalization()(hidden)
        hidden = keras.layers.noise.AlphaDropout(dropout_rate)(hidden)
    else:
        hidden = keras.layers.core.Activation(activation)(tensor)
        hidden = keras.layers.normalization.BatchNormalization()(hidden)
        hidden = keras.layers.core.Dropout(dropout_rate)(hidden)
    return hidden


def _kernelmlfusion(n_classes, inputs, kernel_pool):
    width = (16, 32)

    streams_models = []
    for inp in inputs:
        for i in range(len(kernel_pool)):
            streams_models.append(_stream(inp, width, kernel_pool[i], n_classes))

    # pasa o input -1 pra um mlp ou não
    # passa um softmax
    # streams_models append o softmax

    if len(kernel_pool) > 1:
        concat = keras.layers.concatenate(streams_models, axis=-1)
    else:
        concat = streams_models[0]

    hidden = activation_layer('selu', 0.1, concat)
    # -------------- output layer --------------------------------------------

    hidden = keras.layers.Dense(n_classes)(hidden)
    out = keras.layers.core.Activation('softmax')(hidden)
    # -------------- model buider  --------------------------------------------
    model = keras.models.Model(inputs=inputs, outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSProp',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    np.random.seed(12227)
    #batch_size = 10
    if len(sys.argv) > 2:
        data_input_file = sys.argv[1]
        batch = int(sys.argv[2])
    else:
        data_input_file = '/home/jesimon/Documents/sensors2017/SavedFeatures/LOSO/PAMAP2P.npz'
        batch = 200

    tmp = np.load(data_input_file, allow_pickle=True)
    X = tmp['X']
    y = tmp['y']
    #print('SHAPE X {}'.format(X.shape))
    #sys.exit()

    folds = tmp['folds']
    dataset_name = data_input_file.split('/')[-1]
    data = []

    data = select_sensors(dataset_name, 0, 0)


    #del data
    n_class = y.shape[1]

    avg_acc = []
    avg_recall = []
    avg_f1 = []
    acc_class = []

    # ----------------------------variables of model -----------------
    pool = [(2, 2), (3, 3), (5, 2), (12, 2), (25, 2)]
    #batch = 10
    print('Sena 2018 {}'.format(data_input_file))
    print('Pool  = {}'.format(pool))
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


        _model = _kernelmlfusion(n_class, inputs, pool)
        print('Training...')
        _model.fit(X_train, y, batch, cm.n_ep, verbose=0,
                    callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)],
                    validation_data=(X_train, y))

        sys.stdout.flush()

        """print('Saving Model')
        # serialize model to JSON
        model_json = _model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        _model.save_weights("model.h5")
        print("Saved model to disk")
        """

        y_pred = _model.predict(X_test)

        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y, axis=1)

        fold_acc_class = []

        acc_fold = accuracy_score(y_true, y_pred)
        avg_acc.append(acc_fold)

        recall_fold = recall_score(y_true, y_pred, average='macro')
        avg_recall.append(recall_fold)

        f1_fold = f1_score(y_true, y_pred, average='macro')
        avg_f1.append(f1_fold)

        print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold, i))
        sys.stdout.flush()
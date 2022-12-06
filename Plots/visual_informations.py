import numpy as np
import keras
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys
import custom_model as cm

from keras import backend as K
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def get_cmap(n, name='gist_rainbow'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

K.set_image_data_format('channels_first')


def _stream2D(inp, n_filters, kernel, n_classes):
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
    n_neurons = 50
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




if __name__ == '__main__':
    np.random.seed(12227)

    if len(sys.argv) > 1:
        data_input_file = sys.argv[1]
    else:
        data_input_file = '/mnt/users/jessica/datasets/LOSO/MHEALTH.npz'
        #data_input_file = 'Z:/datasets/LOSO/MHEALTH.npz'

    tmp = np.load(data_input_file, allow_pickle=True)
    X = tmp['X']
    # For sklearn methods X = X[:, 0, :, :]
    y = tmp['y']
    folds = tmp['folds']

    X_modas = []

    # ------------------------------------------------------------------------------------
    # split dataset into modalities
    # ------------------------------------------------------------------------------------
    dataset_name = data_input_file.split('/')[-1]
    if dataset_name == 'UTD-MHAD2_1s.npz' or dataset_name == 'UTD-MHAD1_1s.npz' or dataset_name == 'USCHAD.npz':
        X_modas.append(X[:, :, :, 0:3])
        X_modas.append(X[:, :, :, 3:6])
    elif dataset_name == 'WHARF.npz' or dataset_name == 'WISDM.npz':
        X_modas.append(X)
        data = []
    elif dataset_name == 'MHEALTH.npz':
        X_modas.append(X[:, :, :, 5:8])  # ACC right-lower-arm
        X_modas.append(X[:, :, :, 17:20])  # GYR right-lower-arm
        X_modas.append(X[:, :, :, 20:23])  # MAG right-lower-arm

    n_class = y.shape[1]

    #for i in range(0, len(folds)):
    i=0
    train_idx = folds[i][0]
    test_idx = folds[i][1]
    X_train = []
    X_test = []
    for x in X_modas:
        X_train.append(x[train_idx])
        X_test.append(x[test_idx])

    model = load_model('Sena2018_model_' + dataset_name.split(".")[0] + '_fold0.h5')

    feature_layer = K.function([model.layers[0].input, model.layers[1].input, model.layers[2].input, K.learning_phase()], [model.layers[168].output])

    features = feature_layer([X_train[0], X_train[1], X_train[2], 0])[0]
    # features = []
    # for v in X_features:
    #     features.append(v.flatten())
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(features)
    y_one = [np.argmax(x) for x in y[train_idx]]
    # lda = LDA(n_components=2)
    # principalComponents = lda.fit_transform(features, y_one)
    # principalComponents = lda.transform(features)
    # lda = LDA(n_components=2)
    # principalComponents = lda.fit(features, y_one).transform(features)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['First Component', 'Second Component'])

    y_pd = pd.DataFrame(data=y_one, columns=['target'])

    finalDf = pd.concat([principalDf, y_pd], axis=1)


    ######### 2D ##############

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('First Component', fontsize=15)
    ax.set_ylabel('Second Component', fontsize=15)
    ax.set_title('MHEALTH Activities Features', fontsize=20)
    cmap = get_cmap(n_class)
    targets = []
    colors = []
    for i in range(n_class):
        targets.append(i)
        colors.append(cmap(i))
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'First Component']
                   , finalDf.loc[indicesToKeep, 'Second Component']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    #plt.show()
    plt.savefig('Sena2018_model_layer168' + dataset_name.split(".")[0] + '_fold0.png')
    plt.clf()

    ######### 3D ##############
    # features = []
    # for v in X_train[0]:
    #     features.append(v.flatten())
    #
    # pca = PCA(n_components=3)
    # principalComponents = pca.fit_transform(features)
    # lda = LDA(n_components=3)
    # principalComponents = lda.fit_transform(features, y_one)
    # principalDf = pd.DataFrame(data=principalComponents, columns=['First Component', 'Second Component', 'Third Component'])
    #
    # y_pd = pd.DataFrame(data=y_one, columns=['target'])
    #
    # finalDf = pd.concat([principalDf, y_pd], axis=1)
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # cmap = get_cmap(n_class)
    # targets = []
    # colors = []
    # for i in range(n_class):
    #     targets.append(i)
    #     colors.append(cmap(i))
    # for target, color in zip(targets, colors):
    #     indicesToKeep = finalDf['target'] == target
    #     ax.scatter(finalDf.loc[indicesToKeep, 'First Component']
    #                , finalDf.loc[indicesToKeep, 'Second Component'],  finalDf.loc[indicesToKeep, 'Third Component']
    #                , c=color
    #                , s=50)
    # ax.legend(targets)
    # ax.grid()
    # plt.savefig('Sena2018_model_layer168' + dataset_name.split(".")[0] + '_fold0_3D.png')
    # plt.clf()
    # plt.show()



    # output in train mode = 1
    #layer_output = get_3rd_layer_output([x, 1])[0]

   # Your testing goes here. For instance:
   # y_pred = _model.predict(X_test)

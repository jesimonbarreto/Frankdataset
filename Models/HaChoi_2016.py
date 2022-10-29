import sys
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import scipy.stats as st

from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Activation, Concatenate, merge
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras.models import Model
from keras import backend as K
K.set_image_data_format('channels_first')

import glob
from .Model import Model
from .custom_model import custom_model as cm
from .custom_model import custom_stopping

class HaChoi(Model):

    def custom_model_(self,inp, n_classes):
        #Original architecture
        H1 = Conv2D(filters=5, kernel_size=(5, 5))(inp)
        H1 = Activation('relu')(H1)
        H1 = MaxPooling2D(pool_size=(4, 4))(H1)

        H2 = Conv2D(filters=5, kernel_size=(5, 5))(inp)
        H2 = Activation('relu')(H2)
        H2= MaxPooling2D(pool_size=(4, 4))(H2)

        shape_1 = int(H2.shape[1].value)
        shape_2 = int(H2.shape[2].value)
        shape_3 = int(H2.shape[3].value)
        inp_zeros = Input((shape_1, shape_2, shape_3))
        H = merge([H1, inp_zeros, H2], mode='concat', concat_axis=3)

        H = Conv2D(filters=10, kernel_size=(5, 5))(H)
        H = Activation('relu')(H)
        H = MaxPooling2D(pool_size=(2, 2))(H)

        H = Flatten()(H)
        H = Dense(120)(H)
        H = Activation('relu')(H)

        H = Dense(n_classes)(H)
        H = Activation('softmax')(H)

        model = Model([inp, inp_zeros], H)

        return model, (shape_1, shape_2, shape_3)

    def zero_padding(self, X, idx):
        #'The number of zero-padded columns is set to one less vertical size of 2D convolutional kernel'.
        #Therefore, we need to add 2 zero padding after every selected column

        output = []
        v_kernel = 3-1
        for sample in X:
            sample = sample[0] #or sample = sample[0,:,:]
            sample = np.insert(sample, idx, 0, axis=1)
            output.append([sample])

        output = np.array(output)
        return output

    def model_use(self, dir_files):
        #Paper: Multi-modal convolutional neural networks for activity recognition
        np.random.seed(12227)

        data_input_files = glob.glob(dir_files)

        result = {}

        for data_input_file in data_input_files:

            tmp = np.load(data_input_file, allow_pickle=True)
            X = tmp['X']
            y = tmp['y']
            folds = tmp['folds']
            y = self.code_y(y)
            n_class = y.shape[1]

            #Groups accelerometers and gyroscopes 0, 1, 2, 5, 6, 7, 14, 15, 16 = acc index (-1), 8, 9, 10, 17, 18, 19 = gyro index (-1)
            #idx = [0, 1, 2, 5, 6, 7, 14, 15, 16, 8, 9, 10, 17, 18, 19]
            #X = X[:, :, :, idx]

            #idx = [3] #For UTD-MHAD, USCHAD, WISDM
            idx = [3, 5, 8, 11, 14, 17, 20] #For MHEALTH
            idx = [val for val in idx for _ in (0, 1)] #Vertical Kernel-1

            X = self.zero_padding(X, idx)
            _, _, img_rows, img_cols = X.shape
            avg_acc = []
            avg_recall = []
            avg_f1 = []

            print('Ha and Choi 2016 {}'.format(data_input_file))

            for i in range(0, len(folds)):
                train_idx = folds[i][0]
                test_idx = folds[i][1]

                X_train = X[train_idx]
                X_test = X[test_idx]

                inp = Input((1, img_rows, img_cols))
                model, inp_zeros = self.custom_model(inp, n_classes=n_class)
                zeros_mat = np.zeros((X_train.shape[0], inp_zeros[0], inp_zeros[1], inp_zeros[2]))

                model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adadelta')
                model.fit([X_train, zeros_mat], y[train_idx], batch_size=cm.bs, epochs=cm.n_ep,
                        verbose=0, callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)], validation_data=([X_train, zeros_mat], y[train_idx]))

                zeros_mat = np.zeros((X_test.shape[0], inp_zeros[0], inp_zeros[1], inp_zeros[2]))

                y_pred = model.predict([X_test, zeros_mat])
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
                del model

            ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
            ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
            ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))
            print('Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
            print('Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
            print('Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))
            result[data_input_file] = [[np.mean(avg_acc), ic_acc[0], ic_acc[1]],[np.mean(avg_recall), ic_recall[0], ic_recall[1]],[np.mean(avg_f1), ic_f1[0], ic_f1[1]]]
        print('Final -------------------------------------------------------')

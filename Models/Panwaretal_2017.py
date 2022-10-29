import sys
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import scipy.stats as st

from keras.layers import Input, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Activation, AveragePooling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras.models import Model
from keras import backend as K
K.set_image_data_format('channels_first')
from custom_model import custom_model as cm
from custom_model import custom_stopping
import glob
from Models import Model


class Panwaretal(Model):
    def custom_model_(self, inp, n_classes):
        #Architecture1 from paper
        activation = 'relu'

        H = Conv2D(filters=5, kernel_size=(9, 3), padding = 'same')(inp)
        H = Conv2D(filters=5, kernel_size=(5, 3), padding = 'same')(H)
        H = MaxPooling2D(pool_size=(2, 1))(H)


        H = Activation(activation)(H)

        H = Flatten()(H)
        H = Dense(n_classes)(H)
        H = Activation('softmax')(H)

        model = Model([inp], H)

        return model

    def model_use(self, dir_files):
        #Paper: CNN based approach for activity recognition using a wrist-worn accelerometer
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
            _, _, img_rows, img_cols = X.shape
            avg_acc = []
            avg_recall = []
            avg_f1 = []

            print('Panwar et al. {}'.format(data_input_file))

            for i in range(0, len(folds)):
                train_idx = folds[i][0]
                test_idx = folds[i][1]

                X_train = X[train_idx]
                X_test = X[test_idx]

                inp = Input((1, img_rows, img_cols))
                model = self.custom_model_(inp, n_classes=n_class)

                model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='SGD')
                model.fit(X_train, y[train_idx], batch_size=cm.bs, epochs=cm.n_ep,
                        verbose=0, callbacks=[cm.custom_stopping(value=cm.loss, verbose=1)], validation_data=(X_train, y[train_idx]))

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
                del model

            ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
            ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
            ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))
            print('Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
            print('Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
            print('Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))
            result[data_input_file] = [[np.mean(avg_acc), ic_acc[0], ic_acc[1]],[np.mean(avg_recall), ic_recall[0], ic_recall[1]],[np.mean(avg_f1), ic_f1[0], ic_f1[1]]]
        print('Final -------------------------------------------------------')

        return result
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

from .Model import Model as MD
from .custom_model import custom_model as cm
from .custom_model import custom_stopping


class simpleNet(MD):
    def custom_model2(self, inp, n_classes=None):
        H = Conv2D(filters=24, kernel_size=(2, 2), padding='same')(inp)
        H = Activation('relu')(H)
        #H = MaxPooling2D(pool_size=1)(H)

        H = Conv2D(filters=36, kernel_size=(3, 3), padding='same')(H)
        H = Activation('relu')(H)
        #H = MaxPooling2D(pool_size=1)(H)

        H = Flatten()(H)
        H = Dense(512)(H)
        H = Dense(256)(H)
        H = Dense(n_classes)(H)
        H = Activation('softmax')(H)
        model = Model([inp], H)

        return model

    def model_use(self, data_input_file):
        #Paper: A Deep Learning Approach to Human Activity Recognition Based on Single Accelerometer
        np.random.seed(12227)

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

        print('Jordao et al. 2 layers {}'.format(data_input_file))

        for i in range(0, len(folds)):
            train_idx = folds[i][0]
            test_idx = folds[i][1]

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            print(y_train)

            print(y_train.shape)
            print(X_train.shape)
            print(X_test.shape)
            sp = X_train.shape
            inp = Input((sp[1], sp[2], sp[3]))
            model = self.custom_model2(inp, n_classes=n_class)

            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adadelta')
            model.fit(X_train, y_train, batch_size=cm.bs, epochs=cm.n_ep,
                    verbose=0, callbacks=[custom_stopping(value=cm.loss, verbose=1)], validation_data=(X_train, y_train))

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

        result = [[np.mean(avg_acc), ic_acc[0], ic_acc[1]],[np.mean(avg_recall), ic_recall[0], ic_recall[1]],[np.mean(avg_f1), ic_f1[0], ic_f1[1]]]
        print('Final -------------------------------------------------------')
        return result

    def get_details(self):
        return "Simple net"
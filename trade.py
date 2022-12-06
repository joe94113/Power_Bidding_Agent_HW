import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout

class Trade:
    def __init__(self):
        self.model_con_name = 'model_con.hdf5'
        self.model_gen_name = 'model_gen.hdf5'
        self.buy_price = 2.4
        self.sell_price = 2

    def data_helper(self):
        training_data = os.listdir('training_data')
        x_con_train = []
        x_con_test = []
        y_con_train = []
        y_con_test = []

        x_gen_train = []
        x_gen_test = []
        y_gen_train = []
        y_gen_test = []
        for i, target in enumerate(training_data):
            df = pd.read_csv('training_data/' + target)
            if i <= 40:
                for i in range(int(df.shape[0] / 24 - 7)):
                    x_con_train.append(df.loc[i * 24 : (i + 7) * 24 - 1, 'consumption'].tolist())
                    y_con_train.append(df.loc[(i + 7) * 24 : (i + 8) * 24 - 1, 'consumption'].tolist())
                    
                    x_gen_train.append(df.loc[i * 24 : (i + 7) * 24 - 1, 'generation'].tolist())
                    y_gen_train.append(df.loc[(i + 7) * 24 : (i + 8) * 24 - 1, 'generation'].tolist())
            else:
                for i in range(int(df.shape[0] / 24 - 7)):
                    x_con_test.append(df.loc[i * 24 : (i + 7) * 24 - 1, 'consumption'].tolist())
                    y_con_test.append(df.loc[(i + 7) * 24 : (i + 8) * 24 - 1, 'consumption'].tolist())
                    
                    x_gen_test.append(df.loc[i * 24 : (i + 7) * 24 - 1, 'generation'].tolist())
                    y_gen_test.append(df.loc[(i + 7) * 24 : (i + 8) * 24 - 1, 'generation'].tolist())
        return x_con_train, x_con_test, y_con_train, y_con_test, x_gen_train, x_gen_test, y_gen_train, y_gen_test

    def build_model(self):
        model = Sequential()
        model.add(Dense(168, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(336, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(48, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(24, activation='relu'))
        model.compile(loss="mse", optimizer="adam", metrics=['mse'])
        return model

    def fit_model(self, model, x_con_train, y_con_train, x_gen_train, y_gen_train):
        x_con_train = np.array(x_con_train)
        y_con_train = np.array(y_con_train)
        x_gen_train = np.array(x_gen_train)
        y_gen_train = np.array(y_gen_train)

        history_con = model.fit(
            x_con_train,
            y_con_train,
            batch_size=64,
            epochs=1000,
            validation_split=0.2
        )
        model.save(self.model_con_name)
        # self.draw_loss(history_con)
        
        history_con = model.fit(
            x_gen_train,
            y_gen_train,
            batch_size=64,
            epochs=1000,
            validation_split=0.2
        )
        model.save(self.model_gen_name)
        # self.draw_loss(history_con)

    def draw_loss(self, history):
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='Answer')
        plt.legend();
        plt.show()

    def draw_pre_ans(self, x_con_test, y_con_test, x_gen_test, y_gen_test):
        model_con = keras.models.load_model(self.model_con_name)
        pred = model_con.predict(x_con_test)

        plt.plot(pred[-1],color='red', label='Prediction')
        plt.plot(y_con_test[-1],color='blue', label='Answer')
        plt.legend(loc='best')
        plt.show()

        model_gen = keras.models.load_model(self.model_gen_name)
        pred = model_gen.predict(x_gen_test)

        plt.plot(pred[-1],color='red', label='Prediction')
        plt.plot(y_gen_test[-1],color='blue', label='Answer')
        plt.legend(loc='best')
        plt.show()

    def get_model(self):
        if not os.path.exists(self.model_con_name) or not os.path.exists(self.model_gen_name):
            x_con_train, x_con_test, y_con_train, y_con_test, x_gen_train, x_gen_test, y_gen_train, y_gen_test = self.data_helper()
            # print('data_helper done')
            model = self.build_model()
            # print('build_model done')
            self.fit_model(model, x_con_train, y_con_train, x_gen_train, y_gen_train)
        
        model_con = keras.models.load_model(self.model_con_name)
        model_gen = keras.models.load_model(self.model_gen_name)

        return model_con, model_gen
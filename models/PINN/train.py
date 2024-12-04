import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Dropout, TimeDistributed, Activation, Flatten, SimpleRNN, GRU
# from keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import data as dt
from custom_layers import Newell_Layer
from datetime import datetime
import os
import argparse


def combined_loss(w_phy):
    def loss_fn(y_true, y_pred):
        phy_pred = y_pred[:, :20]
        lstm_pred = y_pred[:, 20:]

        phy_loss = K.mean(K.square(phy_pred - lstm_pred))
        lstm_loss = K.mean(K.square(lstm_pred - y_true))
        return w_phy * phy_loss + (1-w_phy) * lstm_loss

    return loss_fn

def build_PINN_model(train_x_shape, train_y_shape):
    vi_input = Input(shape=(1,), name='vi_input')  # 一个形状为(N, 1)的数组
    delta_v_input = Input(shape=(1,), name='delta_v_input')
    delta_d_input = Input(shape=(1,), name='delta_d_input')
    x_input = Input(shape=(None, 14), name='x_input')  # 一个形状为(N, T, 5)的数组，T是序列的长度。Assuming you've 6 features for LSTM

    # IDM Layer 输出
    # idm_output = IDM_Layer(forward_steps=10)([vi_input, delta_v_input, delta_d_input])
    # idm_output = Flatten()(idm_output)

    # Newell Layer 输出
    phy_output = Newell_Layer(forward_steps=20)([vi_input, delta_v_input, delta_d_input, x_input])
    phy_output = Flatten()(phy_output)

    # LSTM Layer
    forward = 20
    lstm_output_1 = LSTM(128, return_sequences=True)(x_input)
    lstm_output_1 = Dropout(0.2)(lstm_output_1)
    lstm_output_2 = LSTM(128, return_sequences=False)(lstm_output_1)
    lstm_output_2 = Dropout(0.2)(lstm_output_2)
    dense_output = Dense(forward, activation='relu')(lstm_output_2)
    dense_output = Dropout(0.2)(dense_output)
    final_output = Dense(forward)(dense_output)
    lstm_output = Activation("relu", name='my_activation_layer')(final_output)

    # GRU Layer
    # lstm_output = GRU(64, return_sequences=True)(x_input)
    # lstm_output = Dropout(0.2)(lstm_output)
    # lstm_output = GRU(64)(lstm_output)
    # lstm_output = Dense(10)(lstm_output)
    # lstm_output = Activation("relu")(lstm_output)

    combined_output = tf.keras.layers.concatenate([phy_output, lstm_output])

    return [vi_input, delta_v_input, delta_d_input, x_input], phy_output, lstm_output, combined_output

def create_training_model(inputs, combined_output, w_phy):
    training_model = Model(inputs=inputs, outputs = combined_output)
    training_model.compile(optimizer=Adam(learning_rate=0.001), loss=combined_loss(w_phy=w_phy))  #w_phy是物理模型比例
    return training_model

def create_prediction_model(inputs, phy_output, lstm_output, alpha):
    prediction_model = Model(inputs=inputs, outputs=lstm_output)
    return prediction_model


def train_model(train_x, train_y, val_x, val_y, w_phy):
    DataName = "NGSIM_US101"
    backward = 50
    forward = 20
    feature_num = 14

    # 准备数据
    # train_x, val_x, _, train_y, val_y, _, _, _, _ = dt.load_data(num_samples, scenario)

    # 将train_x解构为IDM和LSTM的输入部分
    vi_data = np.array([item[0][0] for item in train_x])
    delta_v_data = np.array([item[0][1] for item in train_x])
    delta_d_data = np.array([item[0][2] for item in train_x])
    x_data = np.array([item[1] for item in train_x])
    x_data = x_data.reshape((-1, backward, 14))
    train_y = np.array(train_y)

    vi_val_data = np.array([item[0][0] for item in val_x])
    delta_v_val_data = np.array([item[0][1] for item in val_x])
    delta_d_val_data = np.array([item[0][2] for item in val_x])
    x_val_data = np.array([item[1] for item in val_x])
    x_val_data = x_val_data.reshape((-1, backward, 14))
    val_y = np.array(val_y)


    # 创建模型
    inputs, phy_output, lstm_output, combined_output = build_PINN_model(x_data.shape, train_y.shape[1])
    training_model = create_training_model(inputs, combined_output, w_phy)

    # 添加收敛检查
    early_stopping = EarlyStopping(monitor='loss', patience=20, min_delta=0.00001, verbose=2)

    # 保存收敛过程
    history = training_model.fit(
        [vi_data, delta_v_data, delta_d_data, x_data], train_y,
        validation_data=([vi_val_data, delta_v_val_data, delta_d_val_data, x_val_data], val_y),
        epochs=1000, batch_size=256, verbose=2, callbacks=[early_stopping]
    )

    combined_loss_history = np.column_stack((history.history['loss'], history.history['val_loss']))
    # os.makedirs(f'./results_{DataName}', exist_ok=True)
    # np.savetxt(f"./results_{DataName}/r1_{num_samples}_{k}_convergence_rate.csv", combined_loss_history, delimiter=",",
    #            header="train_loss,val_loss")

    # 保存模型
    training_model.save(f"./model/{DataName}.h5")


    # 输出训练得到的IDM参数
    # idm_layer = [layer for layer in training_model.layers if isinstance(layer, IDM_Layer)][0]
    # print("vf:", K.get_value(idm_layer.vf))
    # print("A:", K.get_value(idm_layer.A))
    # print("b:", K.get_value(idm_layer.b))
    # print("s0:", K.get_value(idm_layer.s0))
    # print("T:", K.get_value(idm_layer.T))

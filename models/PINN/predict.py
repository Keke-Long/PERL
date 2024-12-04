import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime
import tensorflow as tf
import data as dt
from custom_layers import Newell_Layer
from train import combined_loss, build_PINN_model


def plot_and_save_prediction(A_real, A_PINN, sample_id, DataName):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(A_real)), A_real, color='b', markersize=1, label='Real-world')
    plt.plot(np.arange(len(A_real)), A_PINN, color='r', markersize=1, label='LSTM')
    plt.xlabel('Index')
    plt.ylabel('Acceleration error $(m/s^2)$')
    plt.ylim(-4, 4)
    plt.title(f'Sample ID: {sample_id}')
    plt.legend()
    plt.savefig(f'./results_{DataName}/plots/predict_result_{sample_id}.png')
    plt.close()


physical_model = "Newell"
#physical_model = "FVD"

def predict_model(test_x, test_y_real, A_min, A_max, test_chain_ids, scenario, num_samples, w_phy):
    DataName = "NGSIM_US101"
    forward = 20
    backward = 50
    feature_num = 14

    os.makedirs(f'./results_{DataName}', exist_ok=True)

    # _, _, test_x, _, _, test_y_real, A_min, A_max, test_chain_ids = dt.load_data(num_samples, scenario)
    vi_data_test = np.array([item[0][0] for item in test_x])
    delta_v_data_test = np.array([item[0][1] for item in test_x])
    delta_d_data_test = np.array([item[0][2] for item in test_x])
    x_data_test = np.array([item[1] for item in test_x])
    x_data_test = x_data_test.reshape((-1, backward, 14))

    test_y_real = np.array(test_y_real)  # Convert to numpy array

    # inputs, phy_output, lstm_output, combined_output = build_PINN_model(x_data_test.shape, test_y_real.shape[1])
    # loss_function = combined_loss(w_phy=w_phy)

    # 1. 加载整个模型
    entire_model = load_model(f"./model/{DataName}.h5",
                              custom_objects={'Newell_Layer': Newell_Layer, 'loss_fn': combined_loss(w_phy=w_phy)})
    # entire_model.summary()

    # 2. 获取特定层的输出
    desired_output = entire_model.get_layer('my_activation_layer').output

    # 3. 创建新的预测模型
    inputs = [entire_model.input[i] for i in range(len(entire_model.input))]
    prediction_model = Model(inputs=inputs, outputs=desired_output)

    # 使用预测模型进行预测
    test_y_predict = prediction_model.predict([vi_data_test, delta_v_data_test, delta_d_data_test, x_data_test])

    A_real = test_y_real
    A_PINN = test_y_predict.tolist()

    # 反归一化
    A_real = np.array(A_real) * (A_max - A_min) + A_min
    A_PINN = np.array(A_PINN) * (A_max - A_min) + A_min

    # 找到原始数据作为对比
    df = pd.read_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/{DataName}_{physical_model}_results_5.csv")
    indices = []
    for chain_id in test_chain_ids:
        chain_df = df[df['chain_id'] == chain_id]
        indices.extend(chain_df.index[-forward:])
    # 使用这些索引从A_IDM中提取数据
    A_array = df['a'].iloc[indices].to_numpy()
    n_samples = len(A_array) // forward
    A = A_array.reshape(n_samples, forward)

    V_array = df['v'].iloc[indices].to_numpy()
    V = V_array.reshape(n_samples, forward)

    Y_array = df['y'].iloc[indices].to_numpy()
    Y = Y_array.reshape(n_samples, forward)

    V_LSTM = np.zeros_like(V)
    V_LSTM[:, 0] = V[:, 0]
    for i in range(1, forward):
        V_LSTM[:, i] = V_LSTM[:, i - 1] + A_PINN[:, i] * 0.1

    Y_LSTM = np.zeros_like(Y)
    Y_LSTM[:, 0:2] = Y[:, 0:2]
    for i in range(2, forward):
        Y_LSTM[:, i] = Y_LSTM[:, i - 1] + V_LSTM[:, i] * 0.1 + A_PINN[:, i] * 0.005


    # 保存结果
    # pd.DataFrame(test_chain_ids).to_csv(f'./results_{DataName}/test_chain_ids.csv', index=False)
    pd.DataFrame(A_PINN).to_csv(f'./results_{DataName}/A.csv', index=False)
    # pd.DataFrame(V_LSTM).to_csv(f'./results_{DataName}/V.csv', index=False)
    # pd.DataFrame(Y_LSTM).to_csv(f'./results_{DataName}/Y.csv', index=False)


    # 计算MSE，保存
    a_mse = mean_squared_error(A, A_PINN)
    a_mse_first = mean_squared_error(A[:, 0], A_PINN[:, 0])
    v_mse = mean_squared_error(V, V_LSTM)
    v_mse_first = mean_squared_error(V[:, 1], V_LSTM[:, 1])
    y_mse = mean_squared_error(Y, Y_LSTM)
    y_mse_first = mean_squared_error(Y[:, 2], Y_LSTM[:, 2])
    with open(f"./results_{DataName}/predict_MSE_results.txt", 'a') as f:
        # now = datetime.now()
        # current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        # f.write(f'{current_time}\n')
        # f.write(f'num_samples ={num_samples}\n')

        # scenario, training_size, w_phy, a_mse, v_mse, a_mse_first, v_mse_first
        f.write(f'{scenario},{int(num_samples*0.6)},{w_phy},{a_mse:.4f},{v_mse:.4f},{a_mse_first:.4f},{v_mse_first:.4f}\n')

    # os.makedirs(f'./results_{DataName}/plots', exist_ok=True)
    # for i in range(len(A_real)):
    #     plot_and_save_prediction(A_real[i], A_PINN[i], i, DataName)
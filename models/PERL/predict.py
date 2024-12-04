import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import argparse
from datetime import datetime
import os
import data as dt

DataName = "NGSIM_US101"
# physical_model = "IDM"
# physical_model = "FVD"
physical_model = "Newell"


def predict_function(num_samples, seed, feature_num):
    backward = 50
    forward = 50

    # 准备数据
    _, _, test_x, _, _, test_y, a_residual_min, a_residual_max, test_chain_ids = dt.load_data(num_samples, seed)
    test_x = test_x.reshape(test_x.shape[0], backward, feature_num)

    # 加载模型
    model = load_model(f"./model/{DataName}.h5")

    # 在测试集上进行预测
    A_residual_hat = model.predict(test_x)

    # 反归一化
    A_residual_hat = A_residual_hat * (a_residual_max - a_residual_min) + a_residual_min
    # A_residual_real = test_y
    # A_residual_real = A_residual_real * (a_residual_max - a_residual_min) + a_residual_min

    # 找到原始数据作为对比
    df = pd.read_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/{DataName}_{physical_model}_results_4.6.csv")
    indices = []
    for chain_id in test_chain_ids:
        chain_df = df[df['chain_id'] == chain_id]
        indices.extend(chain_df.index[-forward:])
    # 使用这些索引从A_IDM中提取数据
    A_phy_array = df[f'a_{physical_model}'].iloc[indices].to_numpy()
    n_samples = len(A_phy_array) // forward
    A_phy = A_phy_array.reshape(n_samples, forward)

    A_array = df['a'].iloc[indices].to_numpy()
    A = A_array.reshape(n_samples, forward)

    V_array = df['v'].iloc[indices].to_numpy()
    V = V_array.reshape(n_samples, forward)

    # Y_array = df['y'].iloc[indices].to_numpy()
    # Y = Y_array.reshape(n_samples, forward)

    # 计算A_PERL, V_PERL, Y_PERL
    A_PERL = A_phy - A_residual_hat

    V_PERL = np.zeros_like(V)
    V_PERL[:, 0] = V[:, 0]
    for i in range(1, forward):
        V_PERL[:, i] = V_PERL[:, i - 1] + A_PERL[:, i] * 0.1

    # Y_PERL = np.zeros_like(Y)
    # Y_PERL[:, 0:2] = Y[:, 0:2]
    # for i in range(2, forward):
    #     Y_PERL[:, i] = Y_PERL[:, i - 1] + V_PERL[:, i] * 0.1 + A_PERL[:, i] * 0.005


    # 计算MSE，保存
    a_mse = mean_squared_error(A, A_PERL)
    a_mse_first = mean_squared_error(A[:, 0], A_PERL[:, 0])
    v_mse = mean_squared_error(V, V_PERL)
    v_mse_first = mean_squared_error(V[:, 1], V_PERL[:, 1])
    # y_mse = mean_squared_error(Y, Y_PERL)
    # y_mse_first = mean_squared_error(Y[:, 2], Y_PERL[:, 2])

    # physics的结果也留着
    # A_phy_array = df['a_Newell'].iloc[indices].to_numpy()
    # A_phy = A_phy_array.reshape(n_samples, forward)
    # V_phy_array = df['v_Newell'].iloc[indices].to_numpy()
    # V_phy = V_phy_array.reshape(n_samples, forward)

    # a_mse_phy = mean_squared_error(A[:,:30], A_phy[:,:30])
    # a_mse_phy_first = mean_squared_error(A[:, 0], A_phy[:, 0])
    # v_mse_phy = mean_squared_error(V[:,:30], V_phy[:,:30])
    # v_mse_phy_first = mean_squared_error(V[:, 1], V_phy[:, 1])


    # 保存结果
    # pd.DataFrame(test_chain_ids).to_csv(f'./results_{DataName}/test_chain_ids.csv', index=False)
    pd.DataFrame(A_PERL).to_csv(f'./results_{DataName}/A.csv', index=False)
    # pd.DataFrame(V_PERL).to_csv(f'./results_{DataName}/V.csv', index=False)
    # pd.DataFrame(Y_PERL).to_csv(f'./results_{DataName}/Y.csv', index=False)

    return a_mse, a_mse_first, v_mse, v_mse_first


# if __name__ == '__main__':
#     predict_function(20000,3,11)
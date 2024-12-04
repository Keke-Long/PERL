import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import mean_squared_error


def predict(filepath, w):
    data = pd.read_csv(filepath)

    # Re-initializing the predicted speeds list to cover the entire dataset
    begin = 20
    predicted_v = [None] * begin  # First several entries will be None
    predicted_a = [None] * begin

    # 遍历未知数据的每一行
    for index in range(begin, len(data)):
        row = data.iloc[index]
        found_speed = False
        # 遍历上游的车辆，从veh_ID-1开始
        for i in range(1, 5):
            # 根据shockwave计算应该使用的上游车辆的时刻
            time_difference = (row[f'Y-{i}']-row['Y0']) / (w+row['v-1'])
            if time_difference < 0:
                # 如果时间差为负数，这意味着上游车辆的位置在主车辆之后，我们应该跳过这辆车
                continue
            # 获取这个时刻的上游车辆的速度
            refer_index = index - int(time_difference*10)
            if refer_index >= 0:
                predicted_v.append(data.iloc[refer_index][f'v-{i}'])
                predicted_a.append(data.iloc[refer_index][f'a-{i}'])
                found_speed = True
                break

        # 如果未找到任何上游车辆的信息
        if not found_speed:
            #print(index, found_speed)
            predicted_v.append(predicted_v[-1])
            predicted_a.append(predicted_a[-1])

    # 计算预测的加速度
    # predicted_a += [0] + [(predicted_v[i] - predicted_v[i-1]) for i in range(101, len(predicted_v))]

    data['v0_Newell'] = predicted_v
    data['a0_Newell'] = predicted_a
    data['a0_residual_Newell'] = data['a0_Newell'] - data['a0']
    data['v0_residual_Newell'] = data['v0_Newell'] - data['v0']
    data.to_csv(filepath, index=False)


def plot_predicted_value(data):
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(data['t'], data['v0'], label='Actual Speed', color='blue')
    plt.plot(data['t'], data['v0_Newell'], label='Predicted Speed', linestyle='--', color='red')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.title('Actual vs. Predicted Speed')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(data['t'], data['a0'], label='Actual acceleration', color='blue')
    plt.plot(data['t'], data['a0_Newell'], label='Predicted acceleration', linestyle='--', color='red')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title('Actual vs. Predicted Acceleration')
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate(filepath):
    data = pd.read_csv(filepath)
    v_mse = ((data['v0'] - data['v0_Newell'])**2).mean()
    a_mse = ((data['a0'] - data['a0_Newell'])**2).mean()
    return v_mse, a_mse


def evaluate_model(combined_df):
    combined_df = combined_df.dropna(subset=['a0', 'a0_Newell'])
    subset = combined_df.groupby('chain_id').apply(lambda x: x.iloc[51:100]).reset_index(drop=True)
    mse_a = mean_squared_error(subset['a0'], subset['a0_Newell'])
    mse_v = mean_squared_error(subset['v0'], subset['v0_Newell'])
    return mse_a, mse_v


# Prediction
path = "/home/ubuntu/Documents/PERL/data/NGSIM_haotian/chains/"
all_files = [f for f in os.listdir(path) if f.endswith('.csv')]

# 随机选择500个文件
selected_files = random.sample(all_files, 500)

# 遍历w从3到6以0.1为步长
ws = np.arange(4, 4.6, 0.005)

# 对每个文件和每个w值进行预测

for w in ws:
    print('w=', round(w,3), end=' ')
    chain_id = 0
    for file in selected_files:
        predict(os.path.join(path, file), w)

    # 组合所有预测结果到一个DataFrame
    dfs = []
    for file in selected_files:
        df = pd.read_csv(os.path.join(path, file))
        df['chain_id'] = chain_id
        dfs.append(df)
        chain_id += 1
    combined_df = pd.concat(dfs, ignore_index=True)

    # 加载组合后的数据
    mse_a, mse_v = evaluate_model(combined_df)
    print(f'MSE_a, MSE_v (rows 51-100): {mse_a:.5f}, {mse_v:.5f}')

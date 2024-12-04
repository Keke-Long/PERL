import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import math


def predict_Newell_approximate(filepath):
    data = pd.read_csv(filepath)
    predicted_a = [np.nan] * 50

    # 计算前车相对于veh 0的时间差
    time_diffs = []
    for i in range(1, 5):
        time_diff = (data.iloc[50][f'Y-{i}'] - data.iloc[50]['Y0']) / (w + data.iloc[50][f'v-{i}'])
        time_diffs.append(time_diff)

    # 从第51行开始预测
    for index in range(50, len(data)):
        found_speed = False
        # 遍历每个前车
        for i in range(1, 5):
            if index >= 50 + math.floor(time_diffs[i - 1]*10):
                continue
            refer_index = index - int(time_diffs[i - 1] * 10)
            print('i=',i,int(time_diffs[i - 1] * 10))
            if refer_index >= 0 and refer_index < 51:
                predicted_a.append(data.iloc[refer_index][f'a-{i}'])
                found_speed = True
                break
        if not found_speed:
            predicted_a.append(predicted_a[-1])

    data['a0_Newell'] = predicted_a
    # predicted_a_df = pd.DataFrame(predicted_a, columns=['a0_Newell'])
    # predicted_a_smooth = predicted_a_df.rolling(window=7, min_periods=1).mean()
    # data['a0_Newell'] = predicted_a_smooth['a0_Newell']
    data['a0_residual_Newell'] = data['a0_Newell'] - data['a0']
    return data



def predict_Newell(filepath, w):
    data = pd.read_csv(filepath)
    predicted_a = [np.nan] * 50

    # 计算波动位置
    def calculate_wave_positions(w, time_step):
        return data.iloc[50]['Y0'] + w * time_step * np.arange(50, 0, -1)

    # 计算与波动位置相交的index差
    def calculate_index_diffs(w):
        wave_positions = calculate_wave_positions(w, 0.1)
        index_diffs = {}
        for i in range(1, 5):
            vehicle_column = f'Y-{i}'
            diffs = np.abs(wave_positions - data[vehicle_column][:50].values)
            min_diff_index = np.argmin(diffs)
            index_diffs[vehicle_column] = 50 - min_diff_index
        return index_diffs

    # 预先计算所有前车的index差
    index_diffs = calculate_index_diffs(w)
    start_smoothing_index = len(data)

    # 从第51行开始预测
    for index in range(50, len(data)):
        found_speed = False
        # 遍历每个前车
        for i in range(1, 5):
            vehicle_column = f'Y-{i}'
            refer_index = index - index_diffs[vehicle_column]
            if refer_index >= 0 and refer_index < 50:
                predicted_a.append(data.iloc[refer_index][f'a-{i}'])
                found_speed = True
                if i == 2 and index < start_smoothing_index:
                    start_smoothing_index = index
                break
        if not found_speed:
            time_diff = (data.iloc[50][f'Y-{i}'] - data.iloc[50]['Y0']) / (w + data.iloc[50][f'v-{i}'])
            refer_index = index - int(time_diff * 10)
            if refer_index >= 0 and refer_index < 50:
                predicted_a.append(data.iloc[refer_index][f'a-{i}'])
            else:
                predicted_a.append(predicted_a[-1])
            # predicted_a.append(predicted_a[-1])

    filt = True
    if filt:
        predicted_a_df = pd.DataFrame(predicted_a, columns=['a0_Newell'])
        predicted_a_df = predicted_a_df.rolling(window=20, min_periods=1).mean()
        # predicted_a_df.iloc[start_smoothing_index:] = predicted_a_df.iloc[start_smoothing_index:].rolling(window=20,
        #                                                                                                       min_periods=1).mean()
        data['a0_Newell'] = predicted_a_df['a0_Newell']
    else:
        data['a0_Newell'] = predicted_a
    data['a0_residual_Newell'] = data['a0_Newell'] - data['a0']
    return data



# 检验单个sample的预测
# path = "/home/ubuntu/Documents/PERL/data/NGSIM_haotian/chain/"
# # path = "/home/ubuntu/Documents/PERL/data/HIGHSIM/chains/"
# w = 4.45
# df = predict_Newell('/home/ubuntu/Documents/PERL/data/NGSIM_haotian/chain/lane1_veh23_650.csv', w)
# df.to_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_Newell_result2.csv", index=False)




w = 5
path = "/home/ubuntu/Documents/PERL/data/NGSIM_haotian/chain/"
all_files = [f for f in os.listdir(path) if f.endswith('.csv')]
dfs = []
chain_id = 0  # 初始化子序列编号

for file in tqdm(all_files, desc="Processing files"):
    filepath = os.path.join(path, file)
    df = predict_Newell(filepath, w)
    if all(df['Y-1']-df['Y0'] < 200) and all(df['Y-2']-df['Y-1'] < 200) and all(df['Y-3']-df['Y-2'] < 200) \
            and all(df['Y-1']-df['Y0'] > 3) \
            and all(df['a0_residual_Newell'][-50:] < 3) and all(df['a0_residual_Newell'][-50:] > -3):
        df['chain_id'] = chain_id  # 添加子序列编号
        chain_id += 1
        dfs.append(df)
print('Total number of chains', chain_id)

combined_df = pd.concat(dfs, ignore_index=True)

cols_to_convert = ['veh_ID0', 'veh_ID-1', 'veh_ID-2', 'veh_ID-3', 'veh_ID-4']
for col in cols_to_convert:
    combined_df[col] = combined_df[col].astype('int32')

column_mapping = {'Y0': 'y',
                  'v0': 'v',
                  'a0': 'a',
                  'Y-1': 'y-1',
                  'a0_Newell': 'a_Newell',
                  'a0_residual_Newell': 'a_residual_Newell'}
combined_df.rename(columns=column_mapping, inplace=True)


# 计算整体MSE
df_filtered = combined_df.dropna(subset=['a', 'a_Newell'])
mse = np.mean((df_filtered['a'] - df_filtered['a_Newell'])**2)
print(f"Average MSE: {mse}")


combined_df.to_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_Newell_results_{w}.csv", index=False)

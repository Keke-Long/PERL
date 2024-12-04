import matplotlib.pyplot as plt
import pandas as pd
from IDM import IDM
import numpy as np
from sklearn.metrics import mean_squared_error
import glob

arg = (25.6331, 0.7852, 2.8029, 3.8367, 0.7696)
folder_path = '../Data/NGSIM/chain/'
file_paths = glob.glob(folder_path + '*.csv')

for file_path in file_paths:
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['pre_V'])

    for i, row in df.iterrows():
        #if row['chain_id'] == 0:
        V = row['V']
        pre_V = row['pre_V']
        pre_Y = row['pre_Y']
        df.at[i, 'A_hat'] = IDM(arg, V, V - pre_V, pre_Y - row['Local_Y'])

    mse = mean_squared_error(df.loc[df['chain_id'] == 0, 'A'], df.loc[df['chain_id'] == 0, 'A_hat'])
    print(f'MSE when predicting acceleration for {file_path}:', mse)
    df.to_csv(file_path, index=False)


# df = pd.read_csv('../Data/NGSIM/chain/i-80_3_chain1.csv')
# # # 绘制所有车辆的轨迹
# # vehicle_ids = df['chain_id'].unique()
# # fig, ax = plt.subplots()
# # for vehicle_id in vehicle_ids:
# #     vehicle_data = df[df['chain_id'] == vehicle_id]
# #     ax.plot(vehicle_data['Global_Time'], vehicle_data['Local_Y'], label=f"Vehicle {vehicle_id}")
# # ax.set_xlabel('Global_Time')
# # ax.set_ylabel('Local_Y')
# # ax.legend()
# # plt.show()
#
#
# arg = (25.6331, 0.7852, 2.8029, 3.8367, 0.7696)
#
# # 对最后一辆车（'chain_id'=0）进行加速度预测，
# for i, row in df.iterrows():
#     if row['chain_id'] == 0:
#         V = row['V']
#         pre_V = row['pre_V']
#         pre_Y = row['pre_Y']
#         df.at[i, 'A_hat'] = IDM(arg, V, V - pre_V, pre_Y - row['Local_Y'])
#
# mse = mean_squared_error(df.loc[df['chain_id'] == 0, 'A'], df.loc[df['chain_id'] == 0, 'A_hat'])
# print('MSE when predicting acceleration:', mse)
#
#
# # 绘制结果图
# # last_vehicle_data = df[df['chain_id'] == 0]
# # x = range(len(last_vehicle_data))
# # plt.plot(x, last_vehicle_data['A'], label='Real a')
# # plt.plot(x, last_vehicle_data['A_hat'], label='Predicted a')
# # plt.xlabel('Index')
# # plt.ylabel('Acceleration (m/s^2)')
# # plt.ylim(-3, 3)
# # plt.title('MSE: {:.4f}'.format(mse))
# # plt.legend()
# # plt.show()



'''
FVD有两种计算模式，
第一种是评价FVD的预测能力，在每个sample内，对forward部分进行累计预测，即使用上一个时刻的FVD预测值作为输入，以保证与其他模型一致
第二种是计算residual，作为PERL模型的输入，是对所有数据进行单步预测，即使用上一个时刻的真实值作为输入
'''
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import load_data_fun
from FVD import FVD


# arg = (0.107, 0.537, 22.971, 9.411, 5.013) # FVD_US101第一次标定
arg = (0.11, 0.537, 17.09, 11.929, 2.067) # FVD_US101第二次标定


# FVD_US101 用清理后的数据标定的
backward = 50
forward = 10
t_chain = backward + forward

#
# # 读取数据，形成chain
# df = load_data_fun.load_data()
#
# filtered_data = []
# chain_id = 0  # 初始化子序列编号
#
# for _, group in tqdm(df.groupby(['id', 'Lane_ID']), total=df['id'].nunique()):
#     group = group.sort_values(by='t')  # 按时间排序
#     for i in range(0, len(group) - t_chain + 1, t_chain):
#         sub_group = group.iloc[i:i+t_chain].copy()  # 选取子序列并创建副本
#         sub_group.reset_index(drop=True, inplace=True)  # 重置索引
#         sub_group['chain_id'] = chain_id  # 添加子序列编号
#
#         # 第一种计算模式，计算 a，v，y， 用于记录FVD model的误差
#         # for j in range(backward, t_chain):
#         #     if j == backward:
#         #         sub_group.loc[j, 'a_FVD'] = FVD(arg,
#         #                                         sub_group.loc[j-1, 'v'],
#         #                                         sub_group.loc[j-1, 'v']-sub_group.loc[j-1, 'v-1'],
#         #                                         sub_group.loc[j-1, 'y-1']-sub_group.loc[j-1, 'y'])
#         #         sub_group.loc[j, 'v_FVD'] = sub_group.loc[j-1, 'v'] + 0.1 * sub_group.loc[j-1, 'a']
#         #         sub_group.loc[j, 'y_FVD'] = sub_group.loc[j-1, 'y'] + 0.1 * sub_group.loc[j-1, 'v'] + 0.5 * 0.01 * sub_group.loc[j-1, 'a']
#         #     else:
#         #         sub_group.loc[j, 'a_FVD'] = FVD(arg,
#         #                                         sub_group.loc[j-1, 'v_FVD'],
#         #                                         sub_group.loc[j-1, 'v_FVD']-sub_group.loc[j-1, 'v-1'],
#         #                                         sub_group.loc[j-1, 'y-1']-sub_group.loc[j-1, 'y_FVD'])
#         #         sub_group.loc[j, 'v_FVD'] = sub_group.loc[j-1, 'v_FVD'] + 0.1 * sub_group.loc[j-1, 'a_FVD']
#         #         sub_group.loc[j, 'y_FVD'] = sub_group.loc[j-1, 'y_FVD'] + 0.1 * sub_group.loc[j-1, 'v_FVD'] + 0.5 * 0.01 * sub_group.loc[j-1, 'a_FVD']
#         # sub_group['a_FVD'] = sub_group['a_FVD'].astype(np.float32).round(3)
#         # sub_group['v_FVD'] = sub_group['v_FVD'].astype(np.float32).round(3)
#         # sub_group['y_FVD'] = sub_group['y_FVD'].astype(np.float32).round(3)
#
#
#         # 第二计算模式，计算 a_residual
#         sub_group.loc[0, 'a_residual_FVD'] = 0
#         sub_group.loc[0, 'a_FVD_2'] = sub_group.loc[0, 'a']
#         for j in range(1, t_chain):
#             sub_group.loc[j, 'a_FVD_2'] = FVD(arg,
#                                               sub_group.loc[j - 1, 'v'],
#                                               sub_group.loc[j - 1, 'v'] - sub_group.loc[j - 1, 'v-1'],
#                                               max(0, sub_group.loc[j - 1, 'y-1'] - sub_group.loc[j - 1, 'y']) )
#             sub_group.loc[j, 'a_residual_FVD'] = round(sub_group.loc[j, 'a_FVD_2'] - sub_group.loc[j, 'a'], 3)
#
#         filtered_data.append(sub_group)
#         chain_id += 1
#
# filtered_data_df = pd.concat(filtered_data)
# filtered_data_df.to_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_FVD_results_origin.csv", index=False)


# 根据FVD结果筛选数据
df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_FVD_results_origin.csv")
print('Before filtering len(df)=', len(df))

# 遍历所有的 chain_id
filtered_indices = []

unique_chain_ids = df['chain_id'].unique()
for chain_id in unique_chain_ids:
    print(chain_id)
    chain_df = df[df['chain_id'] == chain_id].copy()
    chain_df_1 = df[df['chain_id'] == chain_id].copy()
    chain_df_1.dropna(subset=['a_FVD_2'], inplace=True)
    if all(chain_df_1['a_FVD_2'] >= -3.5) \
        and all(chain_df_1['Space_Headway'] < 150)\
        and all(chain_df_1['Space_Headway'] > 4): # 排除离谱加速度的数据
        filtered_indices.extend(chain_df.index)

# and all(chain_df_1['a_residual_FVD'] > -3.5) \
#     and all(chain_df_1['a_residual_FVD'] < 3.5) \
df_filtered = df.loc[filtered_indices]
df_filtered['a_residual_FVD'] = df_filtered['a_residual_FVD'].astype(np.float32).round(3)
df_filtered.to_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_FVD_results.csv", index=False)
print('After filtering len(df_filtered)=', len(df_filtered))


# Analysis FVD predciton 记录FVD model的误差
# filtered_data_df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_FVD_results PERL.csv")
# filtered_data_df.dropna(subset=['a_FVD_2'], inplace=True)
# mse_a_single = mean_squared_error(filtered_data_df['a'], filtered_data_df['a_FVD_2'])
# filtered_data_df.dropna(subset=['a_FVD'], inplace=True)
# mse_a = mean_squared_error(filtered_data_df['a'], filtered_data_df['a_FVD'])
# mse_v = mean_squared_error(filtered_data_df['v'], filtered_data_df['v_FVD'])
# mse_y = mean_squared_error(filtered_data_df['y'], filtered_data_df['y_FVD'])
# with open("./results/US101_predict_MSE_results.txt", 'a') as f:
#     now = datetime.now()
#     current_time = now.strftime("%Y-%m-%d %H:%M:%S")
#     f.write(f'{current_time}\n')
#     f.write(f'MSE when predict a of single-steps: {mse_a_single:.5f}\n')
#     f.write(f'MSE when predict a of multi-steps: {mse_a:.5f}\n')
#     f.write(f'MSE when predict v of multi-steps: {mse_v:.5f}\n')
#     f.write(f'MSE when predict y of multi-steps: {mse_y:.5f}\n')
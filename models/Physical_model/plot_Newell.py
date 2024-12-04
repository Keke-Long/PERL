import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os



def plot_chain(chain_id, chain_data):
    # 计算MSE，先过滤掉含有NaN的行
    filtered_chain_data = chain_data.dropna(subset=['a', 'a_Newell'])
    mse = np.mean((filtered_chain_data['a'] - filtered_chain_data['a_Newell']) ** 2)

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.title(f"Chain {chain_id} Acceleration Over Time - MSE: {mse:.4f}")
    plt.xlabel("Time (Row Number)")
    plt.ylabel("Acceleration")
    plt.ylim([-4, 4])

    # 画出vehicle0的历史值和真实值
    plt.plot(chain_data.index, chain_data['a'], label='Vehicle 0 Acceleration', color='black')

    # 画出0到50行的所有4辆车的加速度
    if len(chain_data) > 50:
        grey_colors = ['darkgrey', 'grey', 'lightgrey', 'gainsboro']
        for i, color in zip(range(-1, -5, -1), grey_colors):
            plt.plot(chain_data.index[:51], chain_data[f'a{i}'][:51], label=f'Veh {i}', color=color)

        # 从50行之后只画出预测的加速度值
        plt.plot(chain_data.index[50:], chain_data['a_Newell'][50:], 'o', label='Newell Prediction of Veh 0',
                 color='blue', markersize=2)

    plt.legend()
    plt.savefig(f"./NGSIM plot_Newell/chain_{chain_id}_acceleration.png")  # 保存图像的路径
    plt.close()




# 加载数据
path_to_results = "/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_Newell_results_4.45.csv"
# path_to_results = "/home/ubuntu/Documents/PERL/data/HIGHSIM/HIGHSIM_Newell_results_5.csv"
df = pd.read_csv(path_to_results)

# 遍历每个独特的chain_id
# for chain_id in df['chain_id'].unique():
#     # 获取当前chain的数据
#     chain_data = df[df['chain_id'] == chain_id]
#     # 确保链中有至少52个元素（0到51）
#     if len(chain_data) > 51:
#         # 检查 V[51] 是否等于 V[50] + A[50] * 0.1
#         v_51 = chain_data.iloc[51]['v']
#         v_50 = chain_data.iloc[50]['v']
#         a_50 = chain_data.iloc[50]['a']
#         if v_51 == v_50 + a_50 * 0.1:
#             pass
#         else:
#             print(f"Chain ID {chain_id}: V[51] does not equal V[50] + A[50] * 0.1")


# 遍历前60个chain画图
for chain_id in range(60):
    chain_data = df[df['chain_id'] == chain_id]
    plot_chain(chain_id, chain_data)


# 绘制a_residual_Newell的直方图
# plt.figure(figsize=(10, 6))
# plt.hist(df['a_residual_Newell'], bins=50, color='blue', edgecolor='black')
# plt.title("Distribution of a_residual_Newell")
# plt.xlabel("a_residual_Newell")
# plt.ylabel("Frequency")
# plt.savefig("a_residual_Newell_histogram.png")
# plt.close()

# 计算整体MSE
# df_filtered = df.dropna(subset=['a', 'a_Newell'])
# mse = np.mean((df_filtered['a'] - df_filtered['a_Newell'])**2)
# print(f"Average MSE: {mse}")


# 画有极值的chain
# for chain_id in df['chain_id'].unique():
#     chain_data = df[df['chain_id'] == chain_id]
#     if (chain_data['a_residual_Newell'] < -2.5).any() or (chain_data['a_residual_Newell'] > 2.5).any():
#         plot_chain(chain_id, chain_data)


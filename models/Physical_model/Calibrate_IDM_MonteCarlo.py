import pandas as pd
import random
from IDM import IDM
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import mean_squared_error


def monte_carlo_optimization(df, num_iterations):
    best_mse = 100000
    best_arg = None

    df = df.sort_values(by=['id', 't'])  # 假设这里的 'id' 是车辆标识，'time' 是时间标识
    df['v_previous_step'] = df.groupby('id')['v'].shift(1)

    # 使用tqdm显示进度条
    with tqdm(total=num_iterations, desc='Iterations', postfix={'Best RMSE': float('inf')}) as pbar:
        for _ in range(num_iterations):
            vf = random.uniform(22, 23) # (22, 23)
            A = random.uniform(0.5, 2) # (0.9, 1.3)
            b = random.uniform(2.0, 3) #(2.8, 3)
            s0 = random.uniform(1.0, 1.9) #(1.5, 1.7)
            T = random.uniform(1, 1.9) #(1, 1.3)
            arg = (round(vf, 3), round(A, 3), round(b, 3), round(s0, 3), round(T, 3))

            df['a_hat'] = df.apply(lambda row: IDM(arg, row['v'], row['v'] - row['v-1'], row['y-1'] - row['y']),axis=1)
            df['a_error'] = df['a_hat'] - df['a']
            mse = mean_squared_error(df['a'], df['a_hat'])

            df_valid = df.dropna(subset=['v_previous_step']).copy()
            df_valid.loc[:, 'V_hat'] = df_valid['v_previous_step'] + df_valid['a_hat'] * 0.1  # 使用 .loc 来修改
            df_valid.loc[:, 'V_error'] = df_valid['V_hat'] - df_valid['v']
            mse_v = np.mean(df_valid['V_error'] ** 2)

            if mse_v < best_mse:
                best_mse = mse_v
                best_arg = arg

            # 更新最小MSE的值
            pbar.set_postfix_str({'Best MSE': round(best_mse, 3), 'best_arg': best_arg})
            pbar.update(1)

    return best_arg, best_mse


# Load data reconstructed by Dr. Haotian Shi
# 加载原始数据
# df = load_data_fun.load_data()

# Load cleaned data 即已经删除了不合理a_IDM_2的数据，防止异常值对标定结果的影响
df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_IDM_results.csv")


# 筛选
df = df[df['Preceding'] != 0]
df = df.dropna(subset=['v', 'v-1'])
print('Before filtering len(df)=', len(df))
df = df[(df['Space_Headway'] > 4) & (df['Space_Headway'] < 150)] # 这个阈值直接决定了标定结果
print('After filtering  len(df)=', len(df))


# 随机选取一部分数据进行标定，
df = df.sample(n=300*50*4, random_state=1)  #样本量×时间窗×4个跟驰规则
print('After sampling len(df_sampled)=', len(df))

# 标定
best_arg, best_rmse = monte_carlo_optimization(df, num_iterations = 5000)

# 结果保存

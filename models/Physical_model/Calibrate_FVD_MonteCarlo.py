import pandas as pd
import random
from FVD import FVD
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def monte_carlo_optimization(df, num_iterations):
    best_mse_a = 100000
    best_mse_v = 100000
    best_arg = None

    df = df.sort_values(by=['id', 't'])  # 假设这里的 'id' 是车辆标识，'time' 是时间标识
    df['v_previous_step'] = df.groupby('id')['v'].shift(1)

    # 使用tqdm显示进度条
    with tqdm(total=num_iterations, desc='Iterations', postfix={'Best MSE': float('inf')}) as pbar:
        for _ in range(num_iterations):
            alpha = random.uniform(0, 0.1) #(0.1, 0.2)
            lamda = random.uniform(0.1, 1)
            v_0 = random.uniform(20, 28)
            b = random.uniform(5, 10)
            beta = random.uniform(0.1, 5)
            arg = (round(alpha, 3), round(lamda, 3), round(v_0, 3), round(b, 3), round(beta, 3))

            df['a_hat'] = df.apply(lambda row: FVD(arg, row['v'], row['v'] - row['v-1'], row['y-1']-row['y']),axis=1)
            df['a_error'] = df['a_hat'] - df['a']
            mse_a = mean_squared_error(df['a'], df['a_hat'])

            df_valid = df.dropna(subset=['v_previous_step']).copy()
            df_valid.loc[:, 'V_hat'] = df_valid['v_previous_step'] + df_valid['a_hat'] * 0.1  # 使用 .loc 来修改
            df_valid.loc[:, 'V_error'] = df_valid['V_hat'] - df_valid['v']
            mse_v = np.mean(df_valid['V_error'] ** 2)

            if mse_a < best_mse_a:
                best_mse_v = mse_v
                best_arg = arg
                best_mse_a = mse_a

            # 更新最小MSE的值
            pbar.set_postfix_str({'Best MSE_v': round(best_mse_v, 3), 'Best MSE_a': round(best_mse_a, 3),'best_arg': best_arg})
            pbar.update(1)

    # plt.hist(df['A_error'], bins=20, color='blue', alpha=0.5)
    # plt.title('A_error Distribution')
    return best_arg, best_mse


# 加载原始数据
import sys
sys.path.append('/home/ubuntu/Documents/PERL/models')  # 将 load_data.py 所在的目录添加到搜索路径
# import load_data_fun
# df = load_data_fun.load_data()

# Load cleaned data 即已经删除了不合理a_IDM_2的数据，防止异常值对标定结果的影响
df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_FVD_results_origin.csv")


# 筛选
df = df[df['Preceding'] != 0]
df = df.dropna(subset=['v', 'v-1', 'Space_Headway'])
print('Before filtering len(df)=', len(df))
df = df[(df['Space_Headway'] > 4) & (df['Space_Headway'] < 150)]
print('After filtering  len(df)=', len(df))


# 随机选取一部分数据进行标定，
df = df.sample(n=10000*50*4, random_state=1)  # Fixing the random state for reproducibility
print('After sampling len(df_sampled)=', len(df))

# 标定
best_arg, best_mse = monte_carlo_optimization(df, num_iterations = 10000)

# 结果保存
# {'Best RMSE': 5.586, 'best_arg': (0.676, 0.761, 25.11, 22.423, 2.299)}]
# {'Best RMSE': 2.952, 'best_arg': (0.19, 0.989, 25.403, 6.247, 7.295)}]
# {'Best RMSE': 2.243, 'best_arg': (0.159, 0.541, 25.183, 9.147, 7.256)}]
# {'Best RMSE': 2.121, 'best_arg': (0.15, 0.528, 17.131, 6.87, 2.779)}]
# {'Best RMSE': 1.805, 'best_arg': (0.107, 0.537, 22.971, 9.411, 5.013)}]

# 清洗数据之后：
# {'Best RMSE': 1.465, 'best_arg': (0.11, 0.537, 17.09, 11.929, 2.067)}]


# 11.18
# {'Best MSE': 8.377, 'best_arg': (0.181, 0.874, 26.538, 6.111, 3.344)}]
# {'Best MSE': 4.386, 'best_arg': (0.193, 0.526, 24.921, 19.835, 0.926)}
# {'Best MSE': 4.418, 'best_arg': (0.202, 0.168, 22.571, 17.217, 0.579)}]
# {'Best MSE': 4.484, 'Best MSE_a': 19.494, 'best_arg':  (0.208, 0.469, 24.972, 12.734, 0.916)}]
# {'Best MSE': 1.724, 'Best MSE_a': 1.724, 'best_arg':   (0.144, 0.192, 15.701, 12.643, 0.249)}]
# {'Best MSE_v': 4.415, 'Best MSE_a': 0.835, 'best_arg': (0.009, 0.155, 23.542, 17.878, 4.982)}]
# {'Best MSE_v': 2.02, 'Best MSE_a': 0.707, 'best_arg':  (0.006, 0.119, 21.581, 9.665, 1.402)}]
# {'Best MSE_v': 0.146, 'Best MSE_a': 0.809, 'best_arg': (0.017, 0.118, 25.584, 7.122, 0.515)}]
# {'Best MSE_v': 0.158, 'Best MSE_a': 2.304, 'best_arg': (0.015, 0.676, 26.423, 7.655, 3.429)}]
# {'Best MSE_v': 0.145, 'Best MSE_a': 0.757, 'best_arg': (0.01, 0.148, 25.303, 8.563, 1.921)}]
# {'Best MSE_v': 0.145, 'Best MSE_a': 0.757, 'best_arg': (0.006, 0.143, 25.589, 5.634, 1.85)}]
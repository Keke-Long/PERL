import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from IDM import IDM

df = pd.read_csv("../Data/NGSIM/i-80_3_split_trj_matched.csv")
arg = (22.39, 0.06, 2.9, 3.91, 0.6)

# Predict a2, v2, y2
num_samples = 1
a_errors = []
v_errors = []
y_errors = []
for _ in tqdm(range(num_samples)):
    # 随机选择连续30个步长的行
    random_start = np.random.randint(0, len(df) - 30)
    selected_rows = df.iloc[random_start: random_start + 30]

    # 确保选中的行拥有相同的id
    vehicle_id = selected_rows.iloc[0]['new_id']
    if not all(selected_rows['new_id'] == vehicle_id):
        continue

    selected_rows.to_csv('selected_rows.csv', index=False)

    selected_rows['A_hat'] = np.nan
    selected_rows['V_hat'] = np.nan
    selected_rows['Y_hat'] = np.nan

    # 预测加速度、速度和位置
    selected_rows.at[selected_rows.index[0], 'A_hat'] = selected_rows.iloc[0]['A']
    selected_rows.at[selected_rows.index[0], 'V_hat'] = selected_rows.iloc[0]['V']
    selected_rows.at[selected_rows.index[0], 'Y_hat'] = selected_rows.iloc[0]['Local_Y']

    for i in range(1,len(selected_rows)):
        dt = selected_rows.iloc[i]['Global_Time'] - selected_rows.iloc[i - 1]['Global_Time']
        dv = selected_rows.iloc[i - 1]['A_hat'] * dt
        selected_rows.at[selected_rows.index[i], 'V_hat'] = selected_rows.iloc[i - 1]['V_hat'] + dv

        dy = selected_rows.iloc[i - 1]['V_hat'] * dt + 0.5 * selected_rows.iloc[i - 1]['A_hat'] * dt ** 2
        selected_rows.at[selected_rows.index[i], 'Y_hat'] = selected_rows.iloc[i - 1]['Y_hat'] + dy

        a = IDM(arg, selected_rows.iloc[i]['V_hat'],
                selected_rows.iloc[i]['V_hat'] - selected_rows.iloc[i]['pre_V'],
                selected_rows.iloc[i]['pre_Y'] - selected_rows.iloc[i]['Y_hat'])
        selected_rows.at[selected_rows.index[i], 'A_hat'] = a
        print('a', a, selected_rows.iloc[i]['Y_hat'], selected_rows.iloc[i]['Local_Y'])

    print(selected_rows)
    # 计算误差
    A_error = mean_squared_error(selected_rows['A'], selected_rows['A_hat'])
    A_errors.append(A_error)
    V_error = mean_squared_error(selected_rows['V'], selected_rows['V_hat'])
    V_errors.append(V_error)
    Y_error = mean_squared_error(selected_rows['Local_Y'], selected_rows['Y_hat'])
    Y_errors.append(Y_error)

    if _ == 0:
        plt.figure(figsize=(10, 12))
        # 子图1: 加速度
        plt.subplot(3, 1, 1)
        plt.plot(selected_rows['Global_Time'], selected_rows['A'], label='Real a')
        plt.plot(selected_rows['Global_Time'], selected_rows['A_hat'], label='Predicted a')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s^2)')
        plt.title('Prediction of Acceleration')
        plt.legend()

        # 子图2: 速度
        plt.subplot(3, 1, 2)
        plt.plot(selected_rows['Global_Time'], selected_rows['V'], label='Real speed')
        plt.plot(selected_rows['Global_Time'], selected_rows['V_hat'], label='Predicted speed')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.title('Prediction of Speed')
        plt.legend()

        # 子图3: 位置
        plt.subplot(3, 1, 3)
        plt.plot(selected_rows['Global_Time'], selected_rows['Local_Y'], label='Real position')
        plt.plot(selected_rows['Global_Time'], selected_rows['Y_hat'], label='Predicted position')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Prediction of Position')
        plt.legend()

        plt.tight_layout()
        plt.show()

# 计算平均误差
print('Average MSE when predicting acceleration:', np.mean(A_errors))
print('Average MSE when predicting speed:', np.mean(V_errors))
print('Average MSE when predicting Y position:', np.mean(Y_errors))


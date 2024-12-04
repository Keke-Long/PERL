import matplotlib.pyplot as plt
import pandas as pd
from IDM import IDM
import numpy as np


platoon_num = 40
df = pd.read_csv("../Data/ASta_platoon{}_new.csv".format(platoon_num))

# vf, A, b, s0, T
arg_mapping = {
    1: (28.7350, 0.5668, 3.8063, 4.0139, 0.7252),
    3: (28.5400, 0.5123, 2.6502, 2.4869, 1.5042),
    10: (28.1857, 0.5491, 3.9198, 3.7268, 1.0057),
    20: (29.2005, 0.4253, 4.0378, 2.8185, 1.5558),
    40: (29.9772, 0.6467, 4.5364, 2.8233, 0.8028)
}
arg = arg_mapping.get(platoon_num)

# 构造未来10步加速度的预测
num_steps = 10
predicted_accelerations = []
current_speed = df['Speed2'].iloc[-1]
current_speed_difference = df['Speed2'].iloc[-1] - df['Speed1'].iloc[-1]

for _ in range(num_steps):
    acceleration = IDM(arg, current_speed, current_speed_difference, df['IVS1'].iloc[-1])
    predicted_accelerations.append(acceleration)

    current_speed += acceleration * 0.1
    current_speed_difference = current_speed - df['Speed1'].iloc[-1]  # 使用当前步骤的预测速度更新速度差

# 添加预测结果到DataFrame中
df['A_hat'] = df['A2']  # 用实际加速度初始化预测加速度列
df['A_hat'].iloc[-num_steps:] = predicted_accelerations

# 计算MSE
mse = np.mean((df['A_hat'] - df['A2']) ** 2)
print('MSE when predicting acceleration:', mse)

# 绘制结果图
test_length = int(0.3 * len(df))
d = df[-test_length:]
plt.figure(figsize=(10, 4))
x = range(len(d))
plt.plot(x, d['A2'], label='Real a')
plt.plot(x, d['A_hat'], label='Predicted a')
plt.xlabel('Index')
plt.ylabel('Acceleration (m/s^2)')
plt.ylim(-2, 2)
plt.title('MSE: {:.4f}'.format(mse))
plt.legend()
plt.savefig('../Results/Platoon{}_IDM_result.png'.format(platoon_num))
plt.show()



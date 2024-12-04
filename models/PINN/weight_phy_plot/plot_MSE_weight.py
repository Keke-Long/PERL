import pandas as pd
import matplotlib.pyplot as plt

train_size = 12000
# 读取数据
df = pd.read_csv(f'train_size{train_size}.csv')  # 替换为您的 CSV 文件路径

# 计算每个 w_phy 值的平均误差
grouped = df.groupby('w_phy').mean()


plt.rcParams['font.size'] = 14
tick_font_size = 14
fig, ax1 = plt.subplots(figsize=(5, 3.2))

# 绘制第一个y轴的数据
ax1.plot(grouped.index, grouped['a_mse_first'], label='a in one-step prediction', color='green', marker='^', linestyle='solid')
ax1.plot(grouped.index, grouped['a_mse'], label='a in multi-step prediction', color='blue', marker='o', linestyle='solid')
# 设置第一个y轴的标签
#ax1.set_xlabel('w_phy')
ax1.set_ylabel('MSE of a $(m^2/s^4)$', color='black')
ax1.tick_params(axis='y', labelcolor='black')


# 创建第二个y轴
ax2 = ax1.twinx()
# 绘制第二个y轴的数据
ax2.plot(grouped.index, grouped['v_mse_first'], label='v in one-step prediction', color='purple', marker='x', linestyle='dashed')
ax2.plot(grouped.index, grouped['v_mse'], label='v in multi-step prediction', color='red', marker='s', linestyle='dashed')
# 设置第二个y轴的标签
ax2.set_ylabel('MSE of v $(m^2/s^2)$', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# 设置坐标轴字号
ax1.tick_params(axis='both', labelsize=tick_font_size)
ax2.tick_params(axis='both', labelsize=tick_font_size)
ax1.set_ylim(0, 1.2)
ax2.set_ylim(0, 1.2)
plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.1)

# 添加图例
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.48, 0.95), ncol=2, frameon=False)

plt.savefig(f'train_size{train_size}.png', dpi=300)

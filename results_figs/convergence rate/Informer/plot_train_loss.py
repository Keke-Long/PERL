import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import matplotlib.path as mpath
import matplotlib.patches as mpatches


# 设置全局字体和大小
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18
tick_font_size = 16

DataName = "NGSIM_US101"

def pad_data(data, max_length):
    """用数据的最后一个值填充数据，直到其长度与max_length一致"""
    return np.pad(data, (0, max_length - len(data)), 'edge')



#只画一个案例的一条线，没有别的
training_number = 1000

# 加载convergence rate数据
Data_driven_data = np.loadtxt(f"{training_number}_NN.csv", delimiter=",")
PINN_data = np.loadtxt(f"{training_number}_PINN.csv", delimiter=",")
PERL_data = np.loadtxt(f"{training_number}_PERL.csv", delimiter=",")

# 在填充数据之前，保存每个数据集的长度
data_driven_convergence_point = [len(Data_driven_data)-1]
pinn_convergence_point = [len(PINN_data)-1]
perl_convergence_point = [len(PERL_data)-1]

# 使用pad_data函数确保所有数据的长度与max_length一致
max_length = max(len(Data_driven_data), len(PINN_data), len(PERL_data))
Data_driven_data = pad_data(Data_driven_data, max_length)
PINN_data = pad_data(PINN_data, max_length)
PERL_data = pad_data(PERL_data, max_length)

fig, ax = plt.subplots(figsize=(4, 3.2))

# 橘色 NN
ax.plot(Data_driven_data[:]*2, label="NN", color="#FFA500", linestyle='-', linewidth=1)

# 紫色 PINN
ax.plot(PINN_data[:]*60, label="PINN", color="#9933FF", linestyle=(1,(5,5)), linewidth=1.3)

# 蓝色 PERL
ax.plot(PERL_data[:], label="PERL", color="#0073e6", linestyle=(1,(5,1)), linewidth=1.6)


# # 绘制三角形
# colors = ["#ff7700", "#7A00CC", "#0059b3"]
# delta = 1.5
# plt.scatter(data_driven_convergence_point, Data_driven_data[data_driven_convergence_point][0]*64 * delta,
#             marker='v', color=colors[0], s=80, label="NN Convergence")
# plt.scatter(pinn_convergence_point, PINN_data[pinn_convergence_point][0]*64 * delta,
#             marker='v', color=colors[1], s=80, label="PINN Convergence")
# plt.scatter(perl_convergence_point, PERL_data[perl_convergence_point][0]*49 * delta,
#             marker='v', color=colors[2], s=80, label="PERL Convergence")

# 其他设置
plt.xlabel("Epoch")
plt.xlim(0, max_length)

ax.set_ylabel("MSE Loss $(m^2/s^4)$")
ax.set_yscale("log")  # 设置y轴为对数尺度
plt.ylim(0.01, 1000)

ax.tick_params(axis='x', labelsize=tick_font_size)
ax.tick_params(axis='y', labelsize=tick_font_size)

plt.subplots_adjust(left=0.26, right=0.92, bottom=0.22, top=0.95)

#plt.legend(loc='upper right', frameon=False, fontsize=12, ncol=2)

plt.savefig(f'{training_number}_Convergence Rate.png', dpi=350)
#plt.show()

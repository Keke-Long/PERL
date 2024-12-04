import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.size'] = 18
tick_font_size = 16


# 数据载入
physical_model = "IDM"
#physical_model = "FVD"
df = pd.read_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_{physical_model}_results.csv")

first_id = df['chain_id'].unique()[1]
sub_df = df[df['chain_id'] == first_id]

# 设置seaborn的样式和调色板
sns.set_style("whitegrid", {"text.color": "black"})
palette = ["#1f77b4","#d62728"]   # 使用coolwarm调色板，有2种颜色



def plot_distribution_one_sample(d,c,name):
    plt.figure(figsize=(5, 3))  # 设置图形的大小
    sns.histplot(d, kde=True, color=c, label='$a$', bins=20, alpha=0.6, stat="probability")  # 使用半透明度以便于看到重叠部分

    # 获取当前的轴实例并设置外框颜色为黑色
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    # 设置字体颜色为黑色
    plt.gca().tick_params(colors='black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')

    plt.grid(False)  # 这行代码取消背景格子
    #plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.46, 1))
    plt.xlabel(f'Value of $a$ $(m/s^2)$', color='black')
    plt.ylabel('Frequency', color='black')
    ax.tick_params(axis='x', labelsize=tick_font_size)
    ax.tick_params(axis='y', labelsize=tick_font_size)
    ax.set_xlim([-1,1.5])
    ax.set_ylim([0, 0.16])

    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.95)
    plt.savefig(f"distribution of {name}.png", dpi=350)
    # plt.show()

plot_distribution_one_sample(sub_df['a'],palette[1],'a')
#plot_distribution_one_sample(sub_df[f'a_residual_{physical_model}'],palette[0],'a_residual')


# 根据chain_id计算a和a_residual的方差
variances = df.groupby('chain_id').var()[['a', f'a_residual_{physical_model}']]
# 计算a和a_residual的方差的均值
mean_variance_a = variances['a'].mean()
mean_variance_residual = variances[f'a_residual_{physical_model}'].mean()
print(f"Mean variance of 'a': {mean_variance_a}")
print(f"Mean variance of 'a_residual_{physical_model}': {mean_variance_residual}")

# 创建方差的小提琴图
plt.figure(figsize=(5, 3))
sns.violinplot(data=variances, orient='h', palette=["#d62728", "#1f77b4"], linewidth=1, edgecolor="black")

# 设置xy轴的颜色为黑色
plt.gca().spines['right'].set_color('black')
plt.gca().spines['top'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')

# 设置字体颜色为黑色
plt.gca().tick_params(colors='black')

plt.xlabel('Variance $(m/s^2)$', color='black')
plt.yticks(ticks=[0, 1], labels=['$a$', '$r^a$'])  # 这里我们设置y轴的标签
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.21, top=0.95)

# 在图上添加注释
# plt.text(0.4, 0.2, f"Mean variance of $a$ = {mean_variance_a:.3f}", color='black', fontsize=15)
# plt.text(0.4, 1.2, f"Mean variance of $r^a$ = {mean_variance_residual:.3f}", color='black', fontsize=15)
# plt.gca().set_xlim([-0.3, 2])
plt.savefig("distribution of variance a and residual.png", dpi=350)
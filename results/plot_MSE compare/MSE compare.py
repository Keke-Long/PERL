import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


models = ['Physics', 'Data', 'PINN', 'PERL']
training_data_sizes = [300, 500, 1000, 2000, 5000, 10000, 12000]
metrics = ['a_mse', 'v_mse', 'a_mse_first', 'v_mse_first']
scenarios = [0, 1, 2, 3]


def process_data(df):
    stats = {}
    for scenario in scenarios:
        stats[scenario] = {}
        for metric in metrics:
            stats[scenario][metric] = {}
            for model in models:
                stats[scenario][metric][model] = {'averages': {}, 'lower_percentiles': {}, 'upper_percentiles': {}}
                for size in training_data_sizes:
                    if (scenario == 1 and model == 'Physics') or ((scenario == 2 or scenario == 3) and model == 'Data'):
                        # 复制scenario 0的结果
                        stats[scenario][metric][model]['averages'][size] = stats[0][metric][model]['averages'].get(size, 0)
                        stats[scenario][metric][model]['lower_percentiles'][size] = stats[0][metric][model][
                            'lower_percentiles'].get(size, 0)
                        stats[scenario][metric][model]['upper_percentiles'][size] = stats[0][metric][model][
                            'upper_percentiles'].get(size, 0)
                    else:
                        filtered_data = df[(df['model'] == model) & (df['training_size'] == size) & (df['scenario'] == scenario)]
                        if not filtered_data.empty:
                            stats[scenario][metric][model]['averages'][size] = np.mean(filtered_data[metric])
                            stats[scenario][metric][model]['lower_percentiles'][size] = np.percentile(filtered_data[metric], 25)
                            stats[scenario][metric][model]['upper_percentiles'][size] = np.percentile(filtered_data[metric], 75)
                        else:
                            stats[scenario][metric][model]['averages'][size] = 0
                            stats[scenario][metric][model]['lower_percentiles'][size] = 0
                            stats[scenario][metric][model]['upper_percentiles'][size] = 0
    return stats


def plot_metric_scenario(ax, stats, scenario, metric):
    plt.rcParams['font.size'] = 18
    tick_font_size = 16
    for model, color, linestyle, marker in zip(models,
                                               ["#505050", "#ff7700", "#7A00CC", "#0059b3"],
                                               [':', ':', '--', '-'],
                                               ['d', 's', '^', 'o']):
        ax.plot(training_data_sizes, [stats[scenario][metric][model]['averages'][size] for size in training_data_sizes],
                label=model, linestyle=linestyle, color=color,
                marker=marker, markersize=5, markerfacecolor='none', linewidth=2)

        ax.fill_between(training_data_sizes,
                        [stats[scenario][metric][model]['lower_percentiles'][size] for size in training_data_sizes],
                        [stats[scenario][metric][model]['upper_percentiles'][size] for size in training_data_sizes],
                        color=color, alpha=0.3)

    ax.set_xscale("log")
    ax.tick_params(axis='x', labelsize=tick_font_size)
    ax.tick_params(axis='y', labelsize=tick_font_size)

    # ax.text(0.047, -0.03, "300", transform=ax.transAxes, verticalalignment='top', horizontalalignment='center',
    #         color='black', fontsize=tick_font_size)

    fig.text(0.58, 0.038, 'Training data size', ha='center', va='center')
    if metric in ['a_mse', 'a_mse_first']:
        fig.text(0.04, 0.55, 'MSE of $a$ $(m^2/s^4)$', ha='center', va='center', rotation='vertical')
    else:
        fig.text(0.04, 0.55, 'MSE of $v$ $(m^2/s^2)$', ha='center', va='center', rotation='vertical')

    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.20, top=0.95)



def generate_summary_table(stats, scenario, metric):
    rows = []
    for size in training_data_sizes:
        row = [size]
        for model in models:
            row.append(stats[scenario][metric][model]['averages'][size])
            row.append(stats[scenario][metric][model]['lower_percentiles'][size])
            row.append(stats[scenario][metric][model]['upper_percentiles'][size])
        rows.append(row)

    columns = ['training_size']
    for model in models:
        columns.extend([f'{model}_average', f'{model}_lower_percentile', f'{model}_upper_percentile'])

    df = pd.DataFrame(rows, columns=columns)
    return df


# 读取CSV文件
data_df = pd.read_csv("MSE_results_Data.csv")
perl_df = pd.read_csv("MSE_results_PERL.csv")
physics_df = pd.read_csv("MSE_results_Physics.csv")
pinn_df = pd.read_csv("MSE_results_PINN.csv")


# 重置索引并为每个DataFrame添加模型标签
data_df.reset_index(drop=True, inplace=True)
data_df['model'] = 'Data'
perl_df.reset_index(drop=True, inplace=True)
perl_df['model'] = 'PERL'
physics_df.reset_index(drop=True, inplace=True)
physics_df['model'] = 'Physics'
pinn_df.reset_index(drop=True, inplace=True)
pinn_df['model'] = 'PINN'

# 合并所有数据
all_data = pd.concat([data_df, perl_df, physics_df, pinn_df])
# 处理数据
stats = process_data(all_data)

# 生成 r0 的 a_mse 结果并保存
summary_table_r1_a_mse = generate_summary_table(stats, 1, 'a_mse')
summary_table_r1_a_mse.to_csv('summary_r1_a_mse.csv', index=False)

# 生成 r0 的 v_mse 结果并保存
summary_table_r1_v_mse = generate_summary_table(stats, 1, 'v_mse')
summary_table_r1_v_mse.to_csv('summary_r1_v_mse.csv', index=False)





# 读取CSV文件
data_df = pd.read_csv("MSE_results_Data.csv")
perl_df = pd.read_csv("MSE_results_PERL.csv")
#perl_df = perl_df.loc[perl_df['w'] == 4.4]
physics_df = pd.read_csv("MSE_results_Physics.csv")
pinn_df = pd.read_csv("MSE_results_PINN.csv")

# 重置索引并为每个DataFrame添加模型标签
data_df.reset_index(drop=True, inplace=True)
data_df['model'] = 'Data'
perl_df.reset_index(drop=True, inplace=True)
perl_df['model'] = 'PERL'
physics_df.reset_index(drop=True, inplace=True)
physics_df['model'] = 'Physics'
pinn_df.reset_index(drop=True, inplace=True)
pinn_df['model'] = 'PINN'


# 合并所有数据
all_data = pd.concat([data_df, perl_df, physics_df, pinn_df])
# 处理数据
stats = process_data(all_data)

img_size = (4, 3.3)
img_dpi = 350


#r0
fig, ax = plt.subplots(figsize=img_size)
ax.set_ylim([0, 0.4])
plot_metric_scenario(ax, stats, 0, 'a_mse')
plt.savefig("r0_a_mse.png", dpi=img_dpi)

fig, ax = plt.subplots(figsize=img_size)
#ax.set_ylim([0, 0.4])
plot_metric_scenario(ax, stats, 0, 'v_mse')
plt.savefig("r0_v_mse.png", dpi=img_dpi)

# fig, ax = plt.subplots(figsize=img_size)
# #ax.set_ylim([0, 0.4])
# plot_metric_scenario(ax, stats, 0, 'a_mse_first')
# plt.savefig("r0_a_mse_first.png", dpi=img_dpi)
#
# fig, ax = plt.subplots(figsize=img_size)
# #ax.set_ylim([0, 0.4])
# plot_metric_scenario(ax, stats, 0, 'v_mse_first')
# plt.savefig("r0_v_mse_first.png", dpi=img_dpi)
#
#
# # # r1
# # fig, ax = plt.subplots(figsize=img_size)
# # ax.set_ylim([0, 0.4])
# # plot_metric_scenario(ax, stats, 1, 'a_mse')
# # plt.savefig("r1_a_mse.png", dpi=img_dpi)
# #
# # fig, ax = plt.subplots(figsize=img_size)
# # #ax.set_ylim([0, 0.4])
# # plot_metric_scenario(ax, stats, 1, 'v_mse')
# # plt.savefig("r1_v_mse.png", dpi=img_dpi)
# #
# # fig, ax = plt.subplots(figsize=img_size)
# # #ax.set_ylim([0, 0.4])
# # plot_metric_scenario(ax, stats, 1, 'a_mse_first')
# # plt.savefig("r1_a_mse_first.png", dpi=img_dpi)
# #
# # fig, ax = plt.subplots(figsize=img_size)
# # #ax.set_ylim([0, 0.4])
# # plot_metric_scenario(ax, stats, 1, 'v_mse_first')
# # plt.savefig("r1_v_mse_first.png", dpi=img_dpi)
#
#
# # r2
# fig, ax = plt.subplots(figsize=img_size)
# #ax.set_ylim([0, 0.4])
# plot_metric_scenario(ax, stats, 2, 'a_mse_first')
# plt.savefig("r2_a_mse_first.png", dpi=img_dpi)
#
# fig, ax = plt.subplots(figsize=img_size)
# #ax.set_ylim([0, 0.4])
# plot_metric_scenario(ax, stats, 2, 'v_mse_first')
# plt.savefig("r2_v_mse_first.png", dpi=img_dpi)
#
#
# # r3
# fig, ax = plt.subplots(figsize=img_size)
# #ax.set_ylim([0, 0.4])
# plot_metric_scenario(ax, stats, 3, 'a_mse_first')
# plt.savefig("r3_a_mse_first.png", dpi=img_dpi)
#
# fig, ax = plt.subplots(figsize=img_size)
# #ax.set_ylim([0, 0.4])
# plot_metric_scenario(ax, stats, 3, 'v_mse_first')
# plt.savefig("r3_v_mse_first.png", dpi=img_dpi)

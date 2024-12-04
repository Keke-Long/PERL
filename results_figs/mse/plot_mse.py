import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

models = ['Physics', 'Data', 'PINN', 'PERL']
training_data_sizes = [300, 500, 1000, 2000, 5000, 10000, 15000]
# metrics = ['a_mse', 'v_mse', 'a_mse_first', 'v_mse_first']
# scenarios = [0, 1, 2, 3]

for model_name in ['GRU', 'LSTM', 'Informer', 'VAE']:

    # Read the saved results
    summary_table_r0_a_mse = pd.read_csv(f'summary_{model_name}_a_mse.csv')
    summary_table_r0_v_mse = pd.read_csv(f'summary_{model_name}_v_mse.csv')

    # Plot a_mse chart
    plt.figure(figsize=(4, 3.3))
    plt.rcParams['font.size'] = 15
    tick_font_size = 15
    for model, color, linestyle, marker in zip(models,
                                               ["#505050", "#ff7700", "#7A00CC", "#0059b3"],
                                               [':', ':', '--', '-'],
                                               ['d', 's', '^', 'o']):

        if model == 'PERL':  # Skip the PERL model
            continue

        plt.plot(summary_table_r0_a_mse['training_size'], summary_table_r0_a_mse[f'{model}_average'],
                 label=model, linestyle=linestyle, color=color,
                 marker=marker, markersize=5, markerfacecolor='none', linewidth=2)

        plt.fill_between(summary_table_r0_a_mse['training_size'],
                         summary_table_r0_a_mse[f'{model}_lower_percentile'],
                         summary_table_r0_a_mse[f'{model}_upper_percentile'],
                         color=color, alpha=0.3)

    plt.xscale("log")
    plt.xlabel('Training data size', fontsize=tick_font_size)
    plt.ylabel('MSE of $a$ $(m^2/s^4)$', fontsize=tick_font_size)
    plt.ylim([0, 0.4])
    # plt.legend()
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.20, top=0.95)
    plt.savefig(f'./mse_plots_PINN/{model_name}_a_mse.png', dpi=300)

    # Plot v_mse chart
    plt.figure(figsize=(4, 3.3))
    plt.rcParams['font.size'] = 15
    for model, color, linestyle, marker in zip(models,
                                               ["#505050", "#ff7700", "#7A00CC", "#0059b3"],
                                               [':', ':', '--', '-'],
                                               ['d', 's', '^', 'o']):
        if model == 'PERL':  # Skip the PERL model
            continue

        plt.plot(summary_table_r0_v_mse['training_size'], summary_table_r0_v_mse[f'{model}_average'],
                 label=model, linestyle=linestyle, color=color,
                 marker=marker, markersize=5, markerfacecolor='none', linewidth=2)

        plt.fill_between(summary_table_r0_v_mse['training_size'],
                         summary_table_r0_v_mse[f'{model}_lower_percentile'],
                         summary_table_r0_v_mse[f'{model}_upper_percentile'],
                         color=color, alpha=0.3)

    plt.xscale("log")
    plt.xlabel('Training data size', fontsize=tick_font_size)
    plt.ylabel('MSE of $v$ $(m^2/s^2)$', fontsize=tick_font_size)
    plt.ylim([0, 1.2])
    # Set the major interval of the y-axis
    ax = plt.gca()  # Get the current axis
    ax.yaxis.set_major_locator(MultipleLocator(0.4))
    # plt.legend()
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.20, top=0.95)
    plt.savefig(f'./mse_plots_PINN/{model_name}_v_mse.png', dpi=300)


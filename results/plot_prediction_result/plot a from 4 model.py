import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import glob
from datetime import datetime


def load_data(backward=50, forward=50, data_fraction=0.5):
    df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_Newell_results_4.6.csv")
    df['delta_y'] = df['y-1'] - df['y']

    # Initialize the lists to hold the features and targets
    X = []
    Y = []
    A_Phy = []
    V_Phy = []
    V_real = []

    # Initialize the list to hold the chain_ids for each sample
    sample_chain_ids = []

    chain_ids = df['chain_id'].unique()
    for chain_id in chain_ids:
        # Get the subset of the DataFrame for this chain ID
        chain_df = df[df['chain_id'] == chain_id]

        delta_Y_normalized = chain_df['y'].values.reshape(-1, 1)
        V_1_normalized = chain_df['v-1'].values.reshape(-1, 1)
        V_normalized   = chain_df['v'].values.reshape(-1, 1)
        A_normalized   = chain_df['a'].values.reshape(-1, 1)

        # Create the feature vectors and targets for each sample in this chain
        for i in range(0, len(chain_df) - backward - forward + 1, backward + forward):
            X_sample = np.concatenate((delta_Y_normalized[i:i + backward, 0],
                                       V_1_normalized[i:i + backward, 0],
                                       V_normalized[i:i + backward, 0],
                                       A_normalized[i:i + backward, 0]), axis=0)
            Y_sample = A_normalized[i + backward:i + backward + forward, 0]

            A_Phy.append(chain_df['a_Newell'].values[i + backward:i + backward + forward])
            V_Phy.append(0)
            V_real.append(0)

            X.append(X_sample)
            Y.append(Y_sample)
            sample_chain_ids.append(chain_id)

    # Convert the lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    A_Phy = np.array(A_Phy)
    V_real = np.array(V_real)
    print(f"Original number of samples: {len(X)}")

    # divided into a training + validation set and a test set
    X_temp, X_test, \
    y_temp, y_test,\
    _, A_Phy, \
    _, V_Phy, \
    _, V_real, \
    temp_chain_ids, test_chain_ids = train_test_split(X, Y, A_Phy, V_Phy, V_real, sample_chain_ids,
                                                                                      test_size=0.2, random_state=42)
    # divided into training set and validation set
    X_train, X_val, y_train, y_val, train_chain_ids, val_chain_ids = train_test_split(X_temp, y_temp, temp_chain_ids,
                                                                                      test_size=0.25, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test, test_chain_ids, A_Phy, V_Phy, V_real


DataName = "NGSIM_US101"
forward = 50
backward = 50

_,_, test_x, _,_, test_y_real, test_chain_ids, A_Phy, V_Phy, V_real = load_data()
A_real = test_y_real.tolist()


def plot_for_chains(chain_number, A_real, current_time):
    model_paths = {
        'NN': '/home/ubuntu/Documents/PERL/models/Data_driven/results_NGSIM_US101',
        'PINN': '/home/ubuntu/Documents/PERL/models/PINN/results_NGSIM_US101',
        'PERL': '/home/ubuntu/Documents/PERL/models/PERL/results_NGSIM_US101',
    }

    model_colors = {
        'NN': '#FFA500',
        'PINN': '#9933FF',
        'PERL': '#0073e6',
    }
    model_markers = {
        'NN': 's',  # 正方形
        'PINN': '^',  # 三角形
        'PERL': 'o',  # 圆形
    }

    plt.rcParams['font.size'] = 12
    tick_font_size = 12

    # plot acceleration results of all models
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(len(A_Phy[chain_number])) * 0.1+0.1, A_real[chain_number],
            label="Real-world", linestyle="--", color='black', linewidth=2)
    ax.plot(np.arange(len(A_Phy[chain_number])) * 0.1+0.1, A_Phy[chain_number],
            label="Physics", linestyle="-", color='gray', marker='D', markersize=2)
    for model, path in model_paths.items():
        a_model_df = pd.read_csv(os.path.join(path, 'A_300.csv'))
        a_values = a_model_df.loc[chain_number]
        ax.plot(np.arange(len(a_values))*0.1+0.1, a_values,
                label=f"{model}", color=model_colors[model], marker=model_markers[model], markersize=2)
    ax.set_xlabel('Predict Time (s)', fontsize=13)
    ax.set_ylabel("Acceleration $(m/s^2)$", fontsize=13)
    plt.ylim(-2, 2)
    plt.xlim(0, 5)
    # 修改横坐标刻度
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))  # 主刻度每秒一个
    plt.legend(ncol=3)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.20)
    plt.savefig(f"./results_{current_time}/chain{chain_number}_a.png", dpi=300)
    plt.close(fig)


# Assuming A_real is a predefined array containing real A values for all chains.
# Plot for the first 10 chains
now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H")
os.makedirs(f"./results_{current_time}")
for chain_number in range(600):
    plot_for_chains(chain_number, A_real, current_time)

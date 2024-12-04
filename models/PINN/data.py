'''
没找到好的保存数据的办法，所以每次只能重新读
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


backward = 50
forward = 20
t_chain = backward + forward

def load_data(num_samples, scenario):
    print('num_samples',num_samples)
    if scenario == 0:
        physical_model = "Newell"
    elif scenario == 1:
        physical_model = "Newell"
    elif scenario == 2:
        physical_model = "IDM"
    elif scenario == 3:
        physical_model = "FVD"

    df = pd.read_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_{physical_model}_results_5.csv")

    df['delta_y'] = df['y-1'] - df['y']
    df['delta_y2'] = df['Y-2'] - df['y-1']
    df['delta_y3'] = df['Y-3'] - df['Y-2']
    df['delta_y4'] = df['Y-4'] - df['Y-3']

    # Initialize the scaler
    scaler_delta_y = MinMaxScaler()
    scaler_v = MinMaxScaler()
    scaler_a = MinMaxScaler()

    # Fit the scaler on the entire dataset
    scaler_a.fit(np.array([-4, 4]).reshape(-1, 1))
    scaler_v.fit(np.array([0, 25]).reshape(-1, 1))
    scaler_delta_y.fit(np.array([0, 150]).reshape(-1, 1))

    # Initialize the lists to hold the features and targets
    X = []
    Y = []

    # Initialize the list to hold the chain_ids for each sample
    sample_chain_ids = []

    chain_ids = df['chain_id'].unique()
    for chain_id in chain_ids:
        # Get the subset of the DataFrame for this chain ID
        chain_df = df[df['chain_id'] == chain_id]

        # Normalize the features
        delta_Y_normalized = scaler_delta_y.transform(chain_df['delta_y'].values.reshape(-1, 1))
        delta_Y2_normalized = scaler_delta_y.transform(chain_df['delta_y2'].values.reshape(-1, 1))
        delta_Y3_normalized = scaler_delta_y.transform(chain_df['delta_y3'].values.reshape(-1, 1))
        delta_Y4_normalized = scaler_delta_y.transform(chain_df['delta_y4'].values.reshape(-1, 1))
        V_1_normalized = scaler_v.transform(chain_df['v-1'].values.reshape(-1, 1))
        A_1_normalized = scaler_a.transform(chain_df['a-1'].values.reshape(-1, 1))
        V_2_normalized = scaler_v.transform(chain_df['v-2'].values.reshape(-1, 1))
        A_2_normalized = scaler_a.transform(chain_df['a-2'].values.reshape(-1, 1))
        V_3_normalized = scaler_v.transform(chain_df['v-3'].values.reshape(-1, 1))
        A_3_normalized = scaler_a.transform(chain_df['a-3'].values.reshape(-1, 1))
        V_4_normalized = scaler_v.transform(chain_df['v-4'].values.reshape(-1, 1))
        A_4_normalized = scaler_a.transform(chain_df['a-4'].values.reshape(-1, 1))
        V_normalized = scaler_v.transform(chain_df['v'].values.reshape(-1, 1))
        A_normalized = scaler_a.transform(chain_df['a'].values.reshape(-1, 1))

        # Create the feature vectors and targets for each sample in this chain
        for i in range(0, len(chain_df) - backward - forward + 1, backward + forward):
            LSTM_input = np.concatenate((
                                        delta_Y_normalized[i:i + backward, 0], #0
                                        delta_Y2_normalized[i:i + backward, 0], #1
                                        delta_Y3_normalized[i:i + backward, 0], #2
                                        delta_Y4_normalized[i:i + backward, 0], #3
                                        V_normalized[i:i + backward, 0], #4
                                        V_1_normalized[i:i + backward, 0], #5
                                        V_2_normalized[i:i + backward, 0], #6
                                        V_3_normalized[i:i + backward, 0], #7
                                        V_4_normalized[i:i + backward, 0], #8
                                        A_normalized[i:i + backward, 0], #9
                                        A_1_normalized[i:i + backward, 0], #10
                                        A_2_normalized[i:i + backward, 0], #11
                                        A_3_normalized[i:i + backward, 0], #12
                                        A_4_normalized[i:i + backward, 0]), axis=0)  # 14个特征值
            vi_0 = df['v'][0]
            delta_v_0 = df['v'][0] - df['v-1'][0]
            delta_d_0 = df['y-1'][0] - df['y'][0]
            IDM_input = [vi_0, delta_v_0, delta_d_0]

            X_sample = [IDM_input, LSTM_input.tolist()]
            Y_sample = A_normalized[i + backward:i + backward + forward, 0]

            X.append(X_sample)
            Y.append(Y_sample)
            sample_chain_ids.append(chain_id)

    # Convert the lists to numpy arrays
    # X = np.array(X)
    # Y = np.array(Y)
    print(f"Original number of samples: {len(X)}")
    #print('sample_chain_ids', sample_chain_ids)


    # 随机选取一定数量的数据
    # If the total number of samples is greater than desired number, then choose a subset

    if len(X) > num_samples:
        indices = np.random.choice(len(X), num_samples, replace=False)
        X = [X[i] for i in indices]
        Y = [Y[i] for i in indices]
        sample_chain_ids = [sample_chain_ids[i] for i in indices]

    # divided into a training + validation set and a test set
    X_temp, X_test, y_temp, y_test, temp_chain_ids, test_chain_ids = train_test_split(X, Y, sample_chain_ids,
                                                                                      test_size=0.2, random_state=42)
    # divided into training set and validation set
    X_train, X_val, y_train, y_val, train_chain_ids, val_chain_ids = train_test_split(X_temp, y_temp, temp_chain_ids,
                                                                                      test_size=0.25, random_state=42)
    a_min = -3.414
    a_max = 3.414
    return X_train, X_val, X_test, y_train, y_val, y_test, a_min, a_max, test_chain_ids
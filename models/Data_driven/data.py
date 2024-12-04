import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


#physical_model = "IDM"
#physical_model = "FVD"
physical_model = "Newell"

def save_samples(backward=50, forward=20):
    # df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_IDM_results.csv")
    df = pd.read_csv(f"/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_{physical_model}_results_4.45.csv")

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
        delta_Y_normalized  = scaler_delta_y.transform(chain_df['delta_y'].values.reshape(-1, 1))
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
            X_sample = np.concatenate((
                                       delta_Y_normalized[i:i + backward, 0],
                                       delta_Y2_normalized[i:i + backward, 0],
                                       delta_Y3_normalized[i:i + backward, 0],
                                       # delta_Y4_normalized[i:i + backward, 0],
                                       V_normalized[i:i + backward, 0],
                                       V_1_normalized[i:i + backward, 0],
                                       V_2_normalized[i:i + backward, 0],
                                       V_3_normalized[i:i + backward, 0],
                                       # V_4_normalized[i:i + backward, 0],
                                       A_normalized[i:i + backward, 0],
                                       A_1_normalized[i:i + backward, 0],
                                       A_2_normalized[i:i + backward, 0],
                                       A_3_normalized[i:i + backward, 0]), axis=0)
                                       # A_4_normalized[i:i + backward, 0],
            Y_sample = A_normalized[i + backward:i + backward + forward, 0]

            X.append(X_sample)
            Y.append(Y_sample)
            sample_chain_ids.append(chain_id)

    # Convert the lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    print(f"Original number of samples: {len(X)}")
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    np.save('sample_chain_ids.npy', sample_chain_ids)

#save_samples(backward=50, forward=20)



def load_data(num_samples, seed):
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    sample_chain_ids = np.load('sample_chain_ids.npy')

    # 随机选取一定数量的数据
    # np.random.seed(seed)
    # If the total number of samples is greater than desired number, then choose a subset
    if len(X) > num_samples:
        indices = np.random.choice(len(X), num_samples, replace=False)
        X = X[indices]
        Y = Y[indices]
        sample_chain_ids = [sample_chain_ids[i] for i in indices]

    # divided into a training + validation set and a test set
    X_temp, X_test, y_temp, y_test, temp_chain_ids, test_chain_ids = train_test_split(X, Y, sample_chain_ids,
                                                                                      test_size=0.2, random_state=42)
    # divided into training set and validation set
    X_train, X_val, y_train, y_val, train_chain_ids, val_chain_ids = train_test_split(X_temp, y_temp, temp_chain_ids,
                                                                                      test_size=0.25, random_state=42)

    a_min = -4
    a_max = 4
    return X_train, X_val, X_test, y_train, y_val, y_test, a_min, a_max, test_chain_ids

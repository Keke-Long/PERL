import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
import data as dt
import pandas as pd
from predict import predict_function
from models.common_utils import *
import argparse


def main(args):
    DataName = "NGSIM_US101"
    backward = 50
    forward = 50
    num_samples = args.num_samples
    seed = 22
    feature_num = 11
    physical_model = "Newell"


    # Prepare data
    train_x, val_x, _, train_y, val_y, _, _, _, _ = dt.load_data(num_samples, seed)
    train_x = train_x.reshape(train_x.shape[0], backward, feature_num)
    val_x = val_x.reshape(val_x.shape[0], backward, feature_num)
    train_y = train_y.reshape(train_y.shape[0], forward, 1)
    val_y = val_y.reshape(val_y.shape[0], forward, 1)

    # 假设 train_x 和 train_y 是您的数据
    print("NaN in train_x:", np.isnan(train_x).any())
    print("NaN in train_y:", np.isnan(train_y).any())
    print("Inf in train_x:", np.isinf(train_x).any())
    print("Inf in train_y:", np.isinf(train_y).any())


    # Load model
    model = build_lstm_complex_model((backward, feature_num), forward, 512, 256)
    #model = build_GRU_model((backward, feature_num), forward, 128, 64)

    # 使用学习率衰减策略
    #lr_schedule = ExponentialDecay(initial_learning_rate=0.0015, decay_steps=200, decay_rate=0.9)
    lr_schedule = 0.0008
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')

    # 添加早停策略
    early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=0.00001, verbose=2)

    history = model.fit(train_x, train_y,
                        validation_data=(val_x, val_y),
                        epochs=1000, batch_size=256, verbose=2, callbacks=[early_stopping])

    # combined_loss_history = np.column_stack((history.history['loss'], history.history['val_loss']))
    # os.makedirs(f'./results_{DataName}', exist_ok=True)
    # np.savetxt(f"./results_{DataName}/convergence_rate_Newell_{num_samples}.csv", combined_loss_history, delimiter=",", header="train_loss,val_loss")

    # Save model
    model.save(f"./model/{DataName}.h5")
    #plot_model(model, to_file='./results/model_plot IDM LSTM.png', show_shapes=True, show_layer_names=True)

    a_mse, a_mse_first, v_mse, v_mse_first = predict_function(num_samples, seed, feature_num)  # 预测


    # 将结果保存为字典   scenario,w,layer1,layer2,training_size,a_mse,v_mse,a_mse_first,v_mse_first
    csv_file_path = f"../../results/plot_MSE compare/MSE_results_PERL.csv"
    experiment_results = {
        'scenario': 0,
        'w': 4.6,  #这里没有自动化，w每次修改需要改data和predict
        'layer1': 96,
        'layer2': 64,
        'training_size': round(num_samples * 0.6),
        'a_mse': round(a_mse, 4),
        'v_mse': round(v_mse, 4),
        'a_mse_first': round(a_mse_first, 4),
        'v_mse_first': round(v_mse_first, 4),
    }
    # 创建一个DataFrame，只包含当前实验的结果
    results_df = pd.DataFrame([experiment_results])
    results_df.to_csv(csv_file_path, mode='a', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and predict PERL')
    parser.add_argument('--physics_model', type=str, default='Newell', help='Physics model')
    parser.add_argument('--backward', type=int, default=50, help='Number of backward steps')
    parser.add_argument('--forward', type=int, default=50, help='Number of forward steps')
    parser.add_argument('--num_samples', type=int, default=20000, help='Number of samples')
    args = parser.parse_args()
    main(args)
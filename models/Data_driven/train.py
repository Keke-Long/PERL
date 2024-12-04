from tensorflow.keras.callbacks import EarlyStopping
import data as dt
import os
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam, schedules
from predict import predict_function
from models.common_utils import *


if __name__ == '__main__':
    # 准备数据
    DataName = "NGSIM_US101"
    backward = 50
    forward = 20
    num_samples = 24000
    seed = 22
    feature_num = 11

    # Prepare data
    train_x, val_x, _, train_y, val_y, _, _, _, _ = dt.load_data(num_samples, seed)
    train_x = train_x.reshape(train_x.shape[0], backward, feature_num)
    val_x = val_x.reshape(val_x.shape[0], backward, feature_num)
    train_y = train_y.reshape(train_y.shape[0], forward, 1)
    val_y = val_y.reshape(val_y.shape[0], forward, 1)

    # Load model
    model = build_lstm_complex_model((backward, feature_num), forward, 256, 256)
    #model = build_GRU_model((backward, feature_num), forward, 128, 128)

    # 使用学习率衰减策略
    #lr_schedule = ExponentialDecay(initial_learning_rate=0.0005, decay_steps=500, decay_rate=0.95)
    lr_schedule = 0.001
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')

    # 添加早停策略
    early_stopping = EarlyStopping(monitor='loss', patience=20, min_delta=0.00001, verbose=0)

    # 保存收敛过程
    history = model.fit(train_x, train_y,
                        validation_data=(val_x, val_y),
                        epochs=1000, batch_size=256, verbose=0, callbacks=[early_stopping])

    # combined_loss_history = np.column_stack((history.history['loss'], history.history['val_loss']))
    # os.makedirs(f'./results_{DataName}', exist_ok=True)
    # np.savetxt(f"./results_{DataName}/convergence_rate_{num_samples}.csv", combined_loss_history, delimiter=",", header="train_loss,val_loss")

    # Save model
    model.save(f"./model/{DataName}.h5")
    # plot_model(model, to_file='model_structure LSTM.png', show_shapes=True, show_layer_names=True)

    predict_function(num_samples, seed, feature_num)  # 预测
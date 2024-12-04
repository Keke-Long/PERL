from datetime import datetime
import os
import subprocess
import data as dt
from train import train_model
from predict import predict_model


with open(f"./results_NGSIM_US101/predict_MSE_results.txt", 'a') as f:
    f.write('\n')
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    f.write(f'{current_time}\n')

# 循环运行
for w_phy in [0.1]:
    scenario = 0
    num_samples = 24000
    for _ in range(2):
        train_x, val_x, test_x, train_y, val_y, test_y_real, A_min, A_max, test_chain_ids = dt.load_data(num_samples, scenario)
        train_model(train_x, train_y, val_x, val_y, w_phy)
        predict_model(test_x, test_y_real, A_min, A_max, test_chain_ids, scenario, num_samples, w_phy)



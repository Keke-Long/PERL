import subprocess
from datetime import datetime


with open(f"./results_NGSIM_US101/predict_MSE_results.txt", 'a') as f:
    f.write('\n')
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    f.write(f'{current_time}\n')
    f.write(f'num_samples = 24000, LSTM 128 128\n')

# 循环运行
for _ in range(1):
    subprocess.run(["python", "train.py"])

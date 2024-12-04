import subprocess
from datetime import datetime

# with open(f"./results_NGSIM_US101/predict_MSE_results.txt", 'a') as f:
#     f.write('\n')
#     now = datetime.now()
#     current_time = now.strftime("%Y-%m-%d %H:%M:%S")
#     f.write(f'{current_time}\n')
#     f.write(f'num_samples = 300 GRU 128+64 的lstm 12feature\n')
#
# # 循环运行
# for _ in range(20):
#     subprocess.run(["python", "train.py", "--num_samples", str(int(40000))])


# 300, 500, 1000, 2000, 5000, 10000, 12000
num_samples_list = [round(num / 0.6) for num in [300]]

for num_samples in num_samples_list:
    for _ in range(1):
        subprocess.run(["python", "train.py",
                        "--num_samples", str(500)])
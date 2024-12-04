import pandas as pd
import numpy as np




df = pd.read_csv('/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_Newell_results_4.6.csv')
# 计算整体MSE
df_filtered = df.dropna(subset=['a', 'a_Newell'])
mse = np.mean((df_filtered['a'] - df_filtered['a_Newell'])**2)
print(f"Average MSE: {mse}")
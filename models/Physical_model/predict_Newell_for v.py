from sklearn.metrics import mean_squared_error
import pandas as pd

combined_df = pd.read_csv("/home/ubuntu/Documents/PERL/data/NGSIM_haotian/NGSIM_US101_Newell_results_5.csv")
combined_df = combined_df.reset_index(drop=True)


# 用Newell算的加速度a，重新计算速度v
def compute_v_newell(chain_group):
    # Compute the 51st row's v_Newell
    chain_group.loc[chain_group.index[51], 'v_Newell'] = chain_group.loc[chain_group.index[50], 'v'] + \
                                                         chain_group.loc[chain_group.index[51], 'a_Newell'] * 0.1
    # For rows from 52 onwards, compute v_Newell
    for i in range(52, len(chain_group)):
        chain_group.loc[chain_group.index[i], 'v_Newell'] = chain_group.loc[chain_group.index[i - 1], 'v_Newell'] + \
                                                            chain_group.loc[chain_group.index[i], 'a_Newell'] * 0.1
    return chain_group

combined_df = combined_df.groupby('chain_id').apply(compute_v_newell).reset_index(drop=True)

# Filter the rows 51-100 for each chain
subset = combined_df.groupby('chain_id').apply(lambda x: x.iloc[51:100]).reset_index(drop=True)
mse_a = mean_squared_error(subset['a'], subset['a_Newell'])
mse_v = mean_squared_error(subset['v'], subset['v_Newell'])
print(f'MSE for a0 vs a_Newell (rows 51-100): {mse_a}')
print(f'MSE for v0 vs v_Newell (rows 51-100): {mse_v}')

# Filter the row 51 for each chain
# subset_51 = combined_df.groupby('chain_id').apply(lambda x: x.iloc[51]).reset_index(drop=True)
# mse_a_51 = mean_squared_error(subset_51['a'], subset_51['a_Newell'])
# mse_v_51 = mean_squared_error(subset_51['v'], subset_51['v_Newell'])
# print(f'MSE for a0 vs a_Newell (row 51): {mse_a_51}')
# print(f'MSE for v0 vs v_Newell (row 51): {mse_v_51}')

print(f'{mse_a}, {mse_v}, {mse_a_51}, {mse_v_51}')

import pandas as pd
import random
from IDM import IDM
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def get_data_type(config_path="/home/ubuntu/Documents/PERL/models/config.txt"):
    with open(config_path, 'r') as config_file:
        line = config_file.readline()
        parts = line.strip().split(":")
        if len(parts) == 2 and parts[0].strip() == "data":
            return parts[1].strip()
    return None

def visualize_error_histogram(df):
    e = abs((df['y-1'] - df['y']) - df['Space_Headway'])
    # 绘制误差直方图
    plt.hist(e, bins=20, edgecolor='black')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Errors')
    plt.show()

def visualize_v_Class_histogram(df):
    plt.hist(df['v_Class'], bins=20, edgecolor='black')
    plt.xlabel('v_Class')
    plt.ylabel('Frequency')
    plt.title('Histogram of v_Class')
    plt.show()

def load_data():
    data_type = get_data_type()
    if data_type == 'NGSIM':
        print(f"所有模型使用的数据类型是 {data_type}")

        # Load data
        #data_store_path = "/home/ubuntu/Documents/NGSIM-Haotian Shi/NGSIM_Cleaned_Dataset-main/I-80_download/"
        data_store_path = "/home/ubuntu/Documents/NGSIM-Haotian Shi/NGSIM_Cleaned_Dataset-main/US-101_download/"
        df_list = []
        for filename in os.listdir(data_store_path):
            if filename.endswith(".csv"):
                d = pd.read_csv(os.path.join(data_store_path, filename))
                df_list.append(d)
        df = pd.concat(df_list)

        df = df[df['v_Class'] == 2]

        # 单位转换：ft -> m, ft/s -> m/s, ft/s^2 -> m/s^2
        df['Local_X'] *= 0.3048
        df['Local_Y'] *= 0.3048
        df['v_Vel'] *= 0.3048
        #print("max_v",max(df['v_Vel']))
        df['v_Acc'] *= 0.3048
        df['Space_Headway'] *= 0.3048
        df['Global_Time'] *= 0.001

        # 统一名字
        df.rename(columns={
            'Vehicle_ID': 'id',
            'Global_Time': 't',
            'Local_Y': 'y',
            'v_Vel': 'v',
            'v_Acc': 'a',
        }, inplace=True)
        df = df[['id','t','y','v','a', 'v_Length','v_Class','Lane_ID','Preceding','Following','Space_Headway','Time_Headway']]
        df['Space_Headway'] = df['Space_Headway'].round(3)

        # 节省空间
        df['y'] = df['y'].astype(np.float32).round(3)
        df['v'] = df['v'].astype(np.float32).round(3)
        df['a'] = df['a'].astype(np.float32).round(3)
        df['Space_Headway'] = df['Space_Headway'].astype(np.float32).round(3)
        df['Time_Headway'] = df['Time_Headway'].astype(np.float32).round(3)
        df['v_Length'] = df['v_Length'].astype(np.float32)
        df['v_Class'] = df['v_Class'].astype(np.int8)
        df['Lane_ID'] = df['Lane_ID'].astype(np.int8)
        # print(df.dtypes)

        # 添加前车(-1)信息，前车是在v a Preceding后加上‘-1’表示
        df = df.merge(df[['t', 'Lane_ID', 'id', 'v', 'a', 'y', 'Preceding']],
                             left_on=['t', 'Lane_ID', 'Preceding'],
                             right_on=['t', 'Lane_ID', 'id'],
                             #how='left',
                             suffixes=('', '-1'))
        df.drop('id-1', axis=1, inplace=True)

        # 添加前车(-2)信息
        # df = df.merge(df[['t', 'Lane_ID', 'id', 'v', 'a', 'y', 'Preceding']],
        #                      left_on=['t', 'Lane_ID', 'Preceding-1'],
        #                      right_on=['t', 'Lane_ID', 'id'],
        #                      #how='left',
        #                      suffixes=('', '-2'))
        # df.drop('id-2', axis=1, inplace=True)

        # 检查Space_Headway和(df['y-1'] - df['y']) 的误差
        # visualize_error_histogram(df)

        # visualize_v_Class_histogram(df)

    return df

if __name__ == "__main__":
    load_data()
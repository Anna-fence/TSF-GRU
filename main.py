import time
import os
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from data_preprocess import Utils, MyDataset
from model import GRU
from torch import nn
import torch
import random
import numpy as np
from config import *
# import wandb
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


if __name__ == "__main__":
    need_train = False
    # WBorRB = 1  # 0表示WB，1表示RB
    setup_seed(10)  # 方便复现
    RB_index = 0  # 取第零个资源块的数据
    n_feature = 1  # 特征的长度（也就是输入的一行数据有多少个元素）
    train_window = 100
    data_type = dataset.split('_')[-1]
    model_name = f'tw{test_window}_{data_type}_epT{epoch}_hiddenT{hidden_size}_lr{lr}_GRUModel'
    pic_name = f"{data_type}_tw{test_window}_lr{lr}_{epoch}_{hidden_size}"

    filepath = './DATA/'
    # test_window = 3

    # 实例化工具包
    utils = Utils()

    print("准备数据...")
    csv_data_cell0 = pd.read_csv(filepath + dataset, delimiter=',', index_col=0)
    df_cell0 = pd.DataFrame(csv_data_cell0)
    # 判断干扰值是否有nan
    print(df_cell0['WB'].isnull().any(axis=0))
    # 对RIP值进行round操作
    df_cell0.iloc[:, -1] = round(df_cell0.iloc[:, -1])
    WB_RIP_data = df_cell0['WB'].values[-200000:]
    # print(WB_RIP_data[-1])
    WB_RIP_data = WB_RIP_data.reshape((len(WB_RIP_data), 1))
    train_WB, test_WB = utils.split_data(WB_RIP_data, train_set_ratio=0.7)

    n_input = 1
    print("数据准备完成")

    # 训练集归一化
    # normalized_train, scalar = utils.normalize_train_data(train_WB)
    # 测试集归一化
    # normalized_test = utils.normalize_test_data(test_WB, scalar)
    # 训练集序列化
    train_X, train_Y = utils.create_inout_sequences_nf(train_WB, train_window=train_window,
                                                       test_window=test_window)
    # 封装
    dataset = MyDataset(train_X, train_Y)

    # 模型建立
    model = GRU(input_dim=n_input, hidden_dim=hidden_size, output_dim=n_feature)
    if model.need_train:
        print("训练模型...")
        loss_function = nn.MSELoss()
        optimizer_trend = torch.optim.Adam(model.parameters(), lr=lr)
        # 模型参数设置
        model.set_params(need_train, False, train_window, test_window, epoch, optimizer_trend, loss_function,
                         model_name, k_fold)
        # 模型训练
        model.train_model_with_batches(model, dataset, 8)
        print("训练完成")

    else:
        print("加载模型...")
        model.load_state_dict(torch.load(f'./MODEL/GRU/{model_name}.pt'), strict=True)
        model = model.cuda()
        print("加载完成")

    # 模型测试
    pred = model.eval_GRU_nf(model, test_WB)
    # 对预测值取整
    pred = [np.round(pre) for pre in pred]
    pred = np.array(pred).reshape((len(pred), 1))
    # 去归一化
    # pred = scalar.inverse_transform(pred)

    # 测试
    # utils.show_eval_pic(test_WB[-len(actual_predictions):, :], actual_predictions, '宽带级STL+GRU预测情况', pic_name)
    print("evaluate WB GRU:")
    utils.show_eval_index(test_WB[-len(pred):, :], pred)
    utils.write_result_to_file("./RESULT/pred-value-tw1-GRU.csv", pred)

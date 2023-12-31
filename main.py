import math
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
    # set up seed, params and device
    setup_seed(10)
    train_window = 100
    n_input = 1
    n_output = 1
    data_type = dataset.split('_')[-1]
    model_name = f'tw{100}_epT{epoch}_hiddenT{hidden_size}_lr{lr}_GRUModel_exp2-1'
    pic_name = f"{data_type}_tw{test_window}_lr{lr}_{epoch}_{hidden_size}_exp2-1"
    filepath = './DATA/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    utils = Utils()

    print("preparing data...")
    csv_data = pd.read_csv(filepath + dataset, delimiter=',', index_col=0)
    df = pd.DataFrame(csv_data)
    # convert data from dbm to 0/1 (depends on threshold)
    threshold = -100
    for i in range(len(df['WB'])):
        df['WB'][i] = 0 if df['WB'][i] < threshold else 1
    df.to_csv(filepath + 'ul_interference_data_classification_100.csv')
    WB_RIP_data = df['WB'].values[-200000:]
    WB_RIP_data = WB_RIP_data.reshape((len(WB_RIP_data), 1))
    train_WB, test_WB = utils.split_data(WB_RIP_data, train_set_ratio=0.7)
    print("finish!")

    train_X, train_Y = utils.create_inout_sequences_nf(train_WB, train_window=train_window,
                                                       test_window=test_window)
    dataset = MyDataset(train_X, train_Y)

    model = GRU(input_dim=n_input, hidden_dim=hidden_size, output_dim=n_output)
    model.need_train = True
    if model.need_train:
        print("training model...")
        loss_function = nn.CrossEntropyLoss()
        optimizer_trend = torch.optim.Adam(model.parameters(), lr=lr)
        model.set_params(False, train_window, test_window, epoch, optimizer_trend, loss_function,
                         model_name, k_fold, device)
        model.train_model_with_batches(model, dataset, batch_size=8)
        print("finish!")

    else:
        print("loading model...")
        model.load_state_dict(torch.load(f'./MODEL/GRU/{model_name}.pt'), strict=True)
        model = model.to(device)
        print("finish!")

    pred = model.eval_GRU_nf(model, test_WB)
    pred = [1 if i > 0.5 else 0 for i in pred]
    pred = np.array(pred).reshape((len(pred), 1))

    print("evaluate WB GRU:")
    utils.show_eval_index(test_WB[-len(pred):, :], pred)
    utils.show_eval_pic(test_WB[-len(pred):, :], pred, 'GRU-Model-exp2-1', pic_name)
    utils.write_result_to_file("./RESULT/pred-value-tw1-GRU-exp2-1.csv", pred)

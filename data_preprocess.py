import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
import time
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class Utils:
    @staticmethod
    def unpack_data(mat_data, column_idx, data_type=float):
        """
        :param mat_data: 原始数据集dict
        :param column_idx: 需要的是第几列数据
        :param data_type: 需要的数据类型，默认float
        :return:  返回将第二个维度展开后的数据
        """
        # 去掉表头
        column_data = mat_data[1:-1, column_idx]
        unpacked_data = np.zeros((len(column_data), column_data[0].shape[1]), dtype=data_type)
        for idx, item in enumerate(column_data):
            item = np.reshape(item, (1, -1))
            unpacked_data[idx] = item
        return unpacked_data

    @staticmethod
    def normalize_train_data(data):
        data = np.reshape(data, (-1, 1))
        scalar = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scalar.fit_transform(data)
        normalized_data = torch.from_numpy(normalized_data).to(torch.float32)
        return normalized_data, scalar

    @staticmethod
    def normalize_test_data(data, scalar):
        data = np.reshape(data, (-1, 1))
        test_data = scalar.transform(data)
        return torch.from_numpy(test_data).to(torch.float32)

    @staticmethod
    def split_data(data, train_set_ratio=0.7):
        train_size = int(len(data) * train_set_ratio)
        train_set = data[:train_size, :]
        test_set = data[train_size:, :]
        train_set = torch.from_numpy(train_set)
        test_set = torch.from_numpy(test_set)
        train_set = train_set.float()
        test_set = test_set.float()
        return train_set, test_set

    @staticmethod
    def create_inout_sequences_nf(input_data, train_window=20, test_window=1):
        """
        :param test_window: 测试窗口长度
        :param input_data: 输入数据
        :param train_window: 训练的时间窗口数，默认定为一帧（20个slot）
        :return: 列表，每一行都是一个元组，包含1个窗口内的RIP值作为数据，下个窗口内的RIP作为标签
        """
        train_X = []
        train_Y = []
        L = len(input_data)
        for i in range(L - train_window - test_window - 100):
            train_seq = input_data[i: i + train_window, :]
            train_X.append(train_seq)
            # 间隔99个数
            train_label = input_data[i + train_window + 100: i + train_window + 100 + test_window, :]
            # train_label = input_data[i + train_window: i + train_window + test_window, :]
            train_Y.append(train_label)
        return train_X, train_Y

    @staticmethod
    def create_inout_sequences(input_data, features, train_window=20, test_window=1):
        """
        :param features: 将特征(窗口内的邻区利用率)和rip数据一起加入训练序列
        :param test_window: 测试窗口长度
        :param input_data: 输入数据
        :param train_window: 训练的时间窗口数，默认定为一帧（20个slot）
        :return: 列表，每一行都是一个元组，包含1个窗口内的RIP值作为数据，下个窗口内的RIP作为标签
        """
        inout_seq = []
        L = len(input_data)
        features1, features2, features3, features4, features5, features6, features7, features8, features9 = \
            features.values()
        for i in range(L - train_window):
            train_seq = input_data[i: i + train_window, :]
            feature1 = features1[i: i + train_window, :]
            feature2 = features2[i: i + train_window, :]
            feature3 = features3[i: i + train_window, :]
            feature4 = features4[i: i + train_window, :]
            feature5 = features5[i: i + train_window, :]
            feature6 = features6[i: i + train_window, :]
            feature7 = features7[i: i + train_window, :]
            feature8 = features8[i: i + train_window, :]
            feature9 = features9[i: i + train_window, :]
            train_label = input_data[i + train_window: i + train_window + test_window, :]
            inout_seq.append((train_seq, train_label, feature1, feature2, feature3, feature4, feature5, feature6, feature7,
                              feature8, feature9))
        return inout_seq

    @staticmethod
    def STL_decomposition(data, period):
        beg_depose_time = time.time()
        decomposition = STL(data, period=period).fit()
        end_depose_time = time.time()
        print(f"decompose time: {end_depose_time - beg_depose_time}")

        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        return trend, seasonal, residual

    @staticmethod
    def show_eval_pic(true_data, predicted_data, pic_name, model_name):
        plt.rcParams['figure.figsize'] = (32.0, 16.0)  # 设置figure_size尺寸
        plt.rcParams['savefig.dpi'] = 400
        plt.rcParams['figure.dpi'] = 400
        figure_len = 1000
        predicted_data = predicted_data[:figure_len]
        true_data = true_data[:figure_len]
        x = [i for i in range(len(predicted_data))]  # 点的横坐标
        pre_data = [pre for pre in predicted_data]
        true_data = true_data[:]
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号
        # pre_data = pre_data.detach().numpy()
        plt.plot(x, pre_data, 's-', linestyle='--', color='blue', label="pre_data")  # s-:方形
        plt.plot(x, true_data, 'o-', color='orange', label="true_data")  # o-:圆形
        plt.xlabel("index")
        plt.ylabel("RIP (dbm)")
        plt.legend(loc="best")  # 图例
        plt.title(u"{}".format(pic_name))
        plt.savefig(f"./RESULT/STLandGRU/{model_name}.png")
        # plt.savefig(f"/kaggle/working/{model_name}.png")
        # plt.show()

    @staticmethod
    def show_eval_index(true_data, predicted_data):
        n = len(true_data)
        p = 1
        predicted_data = np.array(predicted_data).reshape(-1, 1)
        true_data = np.array(true_data).reshape(-1, 1)
        mean_absolute_error = metrics.mean_absolute_error(true_data, predicted_data)  # MAE
        mean_squared_error = metrics.mean_squared_error(true_data, predicted_data)  # MSE
        R2 = metrics.r2_score(true_data, predicted_data)  # R square
        root_mean_squared_error = mean_squared_error ** 0.5  # RMSE
        adjusted_R2 = 1 - ((1 - R2) * (n - 1)) / (n - p - 1)  # adjusted R square, 消除了样本数量对结果的影响
        MAPE = metrics.mean_absolute_percentage_error(true_data, predicted_data)

        print("平均绝对误差(MAE)：{}".format(mean_absolute_error))
        print("均方误差(MSE)：{}".format(mean_squared_error))
        print("根均方误差(RMSE)：{}".format(root_mean_squared_error))
        print("测试集(R^2)：{}".format(R2))
        print("测试集(adjusted R^2)：{}".format(adjusted_R2))
        print("MAPE: {}".format(MAPE))

        return {
            'MAE': mean_absolute_error,
            'MSE': mean_squared_error,
            'RMSE': root_mean_squared_error,
            'R2': R2,
            'adjusted_R2': adjusted_R2,
            "MAPE": MAPE
        }

    @staticmethod
    def write_result_to_file(file_name, predicted_data):
        tmp = np.array(predicted_data).reshape((len(predicted_data),))
        # predicted_data = predicted_data.reshape((len(predicted_data), ))
        dataframe = pd.DataFrame({'predicted value': tmp})
        dataframe.to_csv(file_name)

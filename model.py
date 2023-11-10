import torch
from torch import nn
import numpy as np
import time
# import wandb
from data_preprocess import Utils
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.logistic = nn.Sigmoid()

        self.has_features = False

        self.train_window = 20
        self.test_window = 1

        self.epochs = 1
        self.optimizer = None
        self.loss_function = None
        self.model_name = None
        self.need_train = None
        self.k_fold = 5

    def forward(self, input_seq):
        gru_out, hidden = self.gru(input_seq)
        out = self.linear(hidden)
        out = self.logistic(out)
        return out

    def set_params(self, has_features, train_window, test_window, epoch, optimizer, loss_function, model_name, k_fold):
        # 属性
        # self.need_train = need_train
        self.has_features = has_features
        self.train_window = train_window
        self.test_window = test_window
        self.epochs = epoch
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model_name = model_name
        self.k_fold = k_fold

    @staticmethod
    def predict_value_nf(self, model, seq):
        predicted_value = []
        begin_time = time.time()
        value = model(seq)
        end_time = time.time()
        infer_time = end_time - begin_time
        value = value.cuda()
        predicted_value.append(value)
        predicted_value = torch.Tensor(predicted_value)
        return predicted_value.view(-1, 1), infer_time

    @staticmethod
    def predict_value(self, model, seq, features):
        test_window = self.test_window
        train_window = self.train_window
        predicted_value = []
        features1, features2, features3, features4, features5, features6, features7, features8, features9 = \
            features.values()
        for idx in range(test_window):
            feature1 = features1[idx: idx + train_window, :]
            feature2 = features2[idx: idx + train_window, :]
            feature3 = features3[idx: idx + train_window, :]
            feature4 = features4[idx: idx + train_window, :]
            feature5 = features5[idx: idx + train_window, :]
            feature6 = features6[idx: idx + train_window, :]
            feature7 = features7[idx: idx + train_window, :]
            feature8 = features8[idx: idx + train_window, :]
            feature9 = features9[idx: idx + train_window, :]
            value = model(seq[idx: idx+train_window, :], feature1, feature2, feature3, feature4, feature5, feature6,
                          feature7, feature8, feature9)
            value = value.cpu()
            predicted_value.append(value)
            seq = torch.cat((seq, value), 0)
        predicted_value = torch.Tensor(predicted_value)
        # predictedValue.requires_grad_(True)
        return predicted_value.view(-1, 1)

    def train_model(self, model, train_seq):
        print(f"######### start training {self.model_name} ############")
        time_start = time.time()
        model = model.cuda()
        loss_function = self.loss_function.cuda()
        model.train()
        for i in range(self.epochs):
            for seq, labels, feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9 in train_seq:
                self.optimizer.zero_grad()
                seq, labels, feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9 = \
                    seq.cuda(), labels.cuda(), feature1.cuda(), feature2.cuda(), feature3.cuda(), feature4.cuda(), \
                    feature5.cuda(), feature6.cuda(), feature7.cuda(), feature8.cuda(), feature9.cuda()
                # y_pre = model(seq)
                test_window = len(labels)
                train_window = len(seq)
                # y_pre = predictValue(model, test_window, seq)
                sum_loss = 0.0
                for idx in range(test_window):
                    value = model(seq[idx: idx + train_window, :], feature1, feature2, feature3, feature4, feature5,
                                  feature6, feature7, feature8, feature9)
                    seq = torch.cat((seq, value), 0)
                    each_loss = loss_function(value, labels[idx, :])
                    sum_loss = sum_loss + each_loss
                sum_loss.backward()
                # wandb.log({"sum_loss": sum_loss})
                # loss = Variable(loss.data, requires_grad=True)
                # loss.requires_grad_(True)
                self.optimizer.step()

            if i % 10 == 1:
                print(f'epoch: {i:3} loss: {sum_loss.item():10.8f}')
        print(f'epoch: {i:3} loss: {sum_loss.item():10.10f}')
        # 模型保存
        time_end = time.time()
        torch.save(model.state_dict(), f'./MODEL/STLandGRU/{self.model_name}.pt')
        print('time cost', time_end - time_start, 's')

    def train_model_nf(self, model: torch.nn.Module, train_X: torch.Tensor, train_Y: torch.Tensor):
        time_start = time.time()
        model = model.cuda()
        loss_function = self.loss_function.cuda()
        model.train()
        for i in range(self.epochs):
            sum_loss = 0.0
            for data, label in zip(train_X, train_Y):
                self.optimizer.zero_grad()
                data, label = data.cuda(), label.cuda()
                value = model(data)
                each_loss = loss_function(value, label)
                sum_loss = sum_loss + each_loss.item()
                each_loss.backward()
                # wandb.log({"sum_loss": sum_loss})
                # loss = Variable(loss.data, requires_grad=True)
                # loss.requires_grad_(True)
                self.optimizer.step()

            if i % 10 == 1:
                print(f'epoch: {i:3} loss: {sum_loss:10.8f}')
        print(f'epoch: {i:3} loss: {sum_loss:10.10f}')
        # 模型保存
        time_end = time.time()
        torch.save(model.state_dict(), f'./MODEL/GRU/{self.model_name}.pt')
        print('time cost', time_end - time_start, 's')

    def train_model_with_batches(self, model, dataset, batch_size):
        time_start = time.time()
        model = model.cuda()
        loss_function = self.loss_function.cuda()
        model.train()

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for i in range(self.epochs):
            sum_loss = 0.0
            for batch_data, batch_label in data_loader:
                self.optimizer.zero_grad()
                batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
                length = batch_data.shape[0]
                batch_data = batch_data.view(-1, length, 1)  # GRU的输入第二维度是batch_size，注意可能会多个余数
                batch_value = model(batch_data)
                batch_value = batch_value.view(length, 1, 1)
                batch_loss = loss_function(batch_value, batch_label)
                sum_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()

            if i % 10 == 1:
                print(f'epoch: {i:3} loss: {sum_loss / len(data_loader):10.8f}')
            else:
                print(f'epoch: {i:3} loss: {sum_loss / len(data_loader):10.10f}')

        # 模型保存
        time_end = time.time()
        torch.save(model.state_dict(), f'./MODEL/GRU/{self.model_name}.pt')
        print('time cost', time_end - time_start, 's')

    # 评估也在gpu上完成
    def eval_GRU_nf(self, model: torch.nn.Module, test_set: np.ndarray):
        train_window: int = self.train_window
        time_start: float = time.time()
        model.eval()
        # model = model.cpu()
        pred_value: list = []
        t_sum: float = 0.0
        with torch.no_grad():
            for test_idx in range(len(test_set) - train_window):
                test_seq: torch.FloatTensor = torch.FloatTensor(test_set[test_idx: test_idx + train_window, :]).reshape((-1, 1))
                test_seq = test_seq.cuda()
                pred, t = self.predict_value_nf(self, model, test_seq)
                t_sum = t + t_sum
                pred_value.append(pred[-1, :])
        time_end: float = time.time()
        print(f'average cost time per inference: {t_sum / len(pred_value)}s')
        print('time cost', time_end - time_start, 's')
        return pred_value

    def eval_GRU(self, model, test_set, features, feature_dim):
        time_start = time.time()
        model.eval()
        model = model.cpu()
        test_window = self.test_window
        train_window = self.train_window
        pred_value = []
        with torch.no_grad():
            for test_idx in range(len(test_set) - train_window):
                test_seq = torch.FloatTensor(test_set[test_idx: test_idx + train_window, :]).reshape((-1, feature_dim))
                pred = self.predict_value(self, model, test_seq, features)
                pred_value.append(pred[-1, :])
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        return pred_value

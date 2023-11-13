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

        self.device = None

    def forward(self, input_seq):
        gru_out, hidden = self.gru(input_seq)
        out = self.linear(hidden)
        out = self.logistic(out)
        return out

    def set_params(self, has_features, train_window, test_window, epoch, optimizer,
                   loss_function, model_name, k_fold, device):
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
        self.device = device

    @staticmethod
    def predict_value_nf(self, model, seq):
        predicted_value = []
        begin_time = time.time()
        value = model(seq)
        end_time = time.time()
        infer_time = end_time - begin_time
        value = value.to(self.device)
        predicted_value.append(value)
        predicted_value = torch.Tensor(predicted_value)
        return predicted_value.view(-1, 1), infer_time

    def train_model_nf(self, model: torch.nn.Module, train_X: torch.Tensor, train_Y: torch.Tensor):
        time_start = time.time()
        model = model.to(self.device)
        loss_function = self.loss_function.to(self.device)
        model.train()
        for i in range(self.epochs):
            sum_loss = 0.0
            for data, label in zip(train_X, train_Y):
                self.optimizer.zero_grad()
                data, label = data.to(self.device), label.to(self.device)
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
        model = model.to(self.device)
        loss_function = self.loss_function.to(self.device)
        model.train()

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for i in range(self.epochs):
            sum_loss = 0.0
            for batch_data, batch_label in data_loader:
                self.optimizer.zero_grad()
                batch_data, batch_label = batch_data.to(self.device), batch_label.to(self.device)
                length = batch_data.shape[0]
                batch_data = batch_data.view(-1, length, 1)  # GRU的输入第二维度是batch_size，注意可能会多个余数
                batch_value = model(batch_data)
                batch_value = batch_value.view(length, 1, 1)
                batch_loss = loss_function(batch_value, batch_label)
                sum_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()

            if i % 10 == 1:
                print(f'epoch: {i:3} loss: {sum_loss / len(data_loader):10.3f}')
            else:
                print(f'epoch: {i:3} loss: {sum_loss / len(data_loader):10.3f}')

        # 模型保存
        time_end = time.time()
        torch.save(model.state_dict(), f'./MODEL/GRU/{self.model_name}.pt')
        print('time cost: {:.3f} s'.format(time_end - time_start))

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
                test_seq = test_seq.to(self.device)
                pred, t = self.predict_value_nf(self, model, test_seq)
                t_sum = t + t_sum
                pred_value.append(pred[-1, :])
        time_end: float = time.time()
        print('average cost time per inference: {:.3f} s'.format(t_sum / len(pred_value)))
        print('time cost: {:.3f} s'.format(time_end - time_start))
        return pred_value

    def eval_GRU(self, model, test_set, features, feature_dim):
        time_start = time.time()
        model.eval()
        # model = model.cpu()
        test_window = self.test_window
        train_window = self.train_window
        pred_value = []
        with torch.no_grad():
            for test_idx in range(len(test_set) - train_window):
                test_seq = torch.FloatTensor(test_set[test_idx: test_idx + train_window, :]).reshape((-1, feature_dim))
                pred = self.predict_value(self, model, test_seq, features)
                pred_value.append(pred[-1, :])
        time_end = time.time()
        print('time cost: {:.3f} s'.format(time_end - time_start))
        return pred_value

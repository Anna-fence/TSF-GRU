import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--hidden_size', type=int, default=200)
parser.add_argument('--test_window', type=int, default=1)
parser.add_argument('--WBorRB', type=int, default=1)
parser.add_argument('--feature_index', type=int, default=0)
parser.add_argument('--k_fold', type=int, default=1)
parser.add_argument('--dataset', type=str, default='ul_interference_data_dbm.csv')
arguments = parser.parse_args().__dict__

lr = arguments['lr']
epoch = arguments['epoch']
hidden_size = arguments['hidden_size']
WBorRB = arguments['WBorRB']
test_window = arguments['test_window']
feature_index = arguments['feature_index']
k_fold = arguments['k_fold']
dataset = arguments['dataset']

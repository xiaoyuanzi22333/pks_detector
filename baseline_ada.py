import os
import numpy as np
from MyDataset.simulator_dataset import simulator_dataset
from MyDataset.split_dataset import generate_split_dataset
import utils.utils as utils
from torch.utils.data import DataLoader
import torch
from sklearn.svm import SVC
import argparse
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser(description="load parsers")
parser.add_argument('--map', type=int)
parser.add_argument('--partition', type=int, default=100)
parser.add_argument('--rand_seed', type=int)
parser.add_argument('--pth', type=str, help='log/model path')
parser.add_argument('--scl', type=int, help='use scheduler')
parser.add_argument('--time_split', type=int)
parser.add_argument('--time_interval', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--scl_step', type=int)
parser.add_argument('--num_chd', type=int)
args = parser.parse_args()

print("loading params")
exp_name = args.pth
fs = 30
time_split = args.time_split
time_interval = args.time_interval
cuda_device = 0
data_sub_path = 'Data_map' + str(args.map) + '_' + str(time_split) + 's_' + str(time_interval) + 's'
data_path = './Data_' + str(time_split) + 's' + str(time_interval) + 's/' + data_sub_path
batch_size = args.batch_size
num_epoch = args.epoch
record_dir = './logs_base' + '/logs_' + str(time_split) + 's_' + exp_name
model_path = './model_ch/model_saved_' + str(time_split) + 's_' + exp_name
use_scheduler = False if args.scl == 0 else True
scheduler_step = args.scl_step
num_chd = args.num_chd
rand_seed = args.rand_seed
partition = args.partition

print("time_split    : " + str(time_split))
print("time_interval : " + str(time_interval))
print("cuda_device   : " + str(cuda_device))
print("data_path     : " + str(data_path))
print("batch_size    : " + str(batch_size))
print("num_epoch     : " + str(num_epoch))
print("record_dir    : " + str(record_dir))
print("model_path    : " + str(model_path))
print("use_scheduler : " + str(use_scheduler))
print("scheduler_step: " + str(scheduler_step))
print("num_chd       : " + str(num_chd))
print("use_rand_seed : " + str(rand_seed))

def preprocess_data(train_loader):
    """
    将 DataLoader 的数据转换为 SVM 需要的格式：[n_samples, n_features]
    """
    X_list = []
    y_list = []
    for batch_data in train_loader:
        brake = batch_data[0].float()    # [batch_size, ...]
        steer = batch_data[1].float()
        throttle = batch_data[2].float()
        label = batch_data[4].float()    # [batch_size]

        # 将 brake, steer, throttle 展平并拼接为一个特征向量
        batch_size = brake.shape[0]
        brake_flat = brake.reshape(batch_size, -1)    # [batch_size, features1]
        steer_flat = steer.reshape(batch_size, -1)    # [batch_size, features2]
        throttle_flat = throttle.reshape(batch_size, -1)  # [batch_size, features3]
        features = torch.cat([brake_flat, steer_flat, throttle_flat], dim=1)  # [batch_size, total_features]
        
        X_list.append(features.numpy())
        y_list.append(label.numpy())
    
    X = np.vstack(X_list)  # [n_samples, total_features]
    y = np.hstack(y_list)  # [n_samples]
    return X, y

def train():
    print("============start training============")
    dataset = simulator_dataset(data_path)
    train_dataset, test_dataset = generate_split_dataset(dataset, data_sub_path, partition, False)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 预处理数据为 SVM 格式
    print("Preprocessing training data...")
    X_train, y_train = preprocess_data(train_loader)
    print("Preprocessing testing data...")
    X_test, y_test = preprocess_data(test_loader)

    # 初始化 AdaBoost 模型
    base_estimator = DecisionTreeClassifier(max_depth=1)  # 弱分类器：决策树桩
    ada_model = AdaBoostClassifier(estimator=base_estimator,
                                n_estimators=50,  # 弱分类器数量
                                learning_rate=1.0,  # 学习率
                                random_state=42)
    
    # 训练 SVM
    ada_model.fit(X_train, y_train)
    
    # 计算训练集和测试集准确率
    train_accuracy = ada_model.score(X_train, y_train)
    test_accuracy = ada_model.score(X_test, y_test)
    
    # 打印正确率
    print("train_acc: " + str(train_accuracy))
    print("test_acc: " + str(test_accuracy))


if __name__ == "__main__":
    train()
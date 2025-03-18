import torch
import torch.nn as nn
import math
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import os

from MyDataset.simulator_dataset import simulator_dataset
from MyDataset.split_dataset import generate_split_dataset

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
record_dir = './logs_cnnt' + '/logs_' + str(time_split) + 's_' + exp_name
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


# Define the CNN-Transformer Model
class CNNTransformerModel(nn.Module):
    def __init__(self, input_dim=15, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(CNNTransformerModel, self).__init__()

        # Input Layer (assuming input_dim is the feature dimension)
        self.input_dim = input_dim

        # Convolutional Layer
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm1d(num_features=embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.transformer_block1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, dropout=dropout, batch_first=True) #(L, E)
        self.batchnorm_block1 = nn.BatchNorm1d(num_features=embed_dim) # (E, L)
        
        self.transformer_block2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, dropout=dropout, batch_first=True)
        self.batchnorm_block2 = nn.BatchNorm1d(num_features=embed_dim)

        # Dense Layer with Sigmoid
        self.dense = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(embed_dim, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()

        # Convolutional Layer
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.conv1d(x)     # (batch_size, embed_dim, seq_len)
        x = self.batch1(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, embed_dim)

        # Transformer Encoder
        output1 = self.transformer_block1(x, x, x) # (batch_size, seq_len, embed_dim)
        output1 = output1[0].permute(0,2,1)  # (batch_size, embed_dim, seq_len)
        output1 = self.batchnorm_block1(output1 + x.transpose(1, 2)) # (batch_size, embed_dim, seq_len)
        output1 = output1.permute(0,2,1) # (batch_size, seq_len, embed_dim)
        
        output2 = self.transformer_block2(output1, output1, output1) # (batch_size, seq_len, embed_dim)
        output2 = output2[0].permute(0,2,1) # (batch_size, embed_dim, seq_len)
        output2 = self.batchnorm_block2(output2 + output1.transpose(1, 2)) # (batch_size, embed_dim, seq_len)
        output2 = output2.permute(0,2,1) # (batch_size, seq_len, embed_dim)

        output = torch.mean(output2, dim=1)
        output = self.dense(output)
        
        return output

# Positional Encoding (as used in Transformer models)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
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

def test(model, test_loader):
    model.to(cuda_device)
    model.eval()
    correct = 0  # 累计预测正确的样本数
    total = 0    # 累计所有有效样本数

    with torch.no_grad():
        for batch_data in test_loader:
            # if batch_data[0].shape[0] != batch_size:
            #     continue
            brake = batch_data[0].to(cuda_device).float()
            steer = batch_data[1].to(cuda_device).float()
            throttle = batch_data[2].to(cuda_device).float()
            label = batch_data[4].to(cuda_device).float()
            
            brake = brake.unsqueeze(-1)
            steer = steer.unsqueeze(-1)
            throttle = throttle.unsqueeze(-1)
            inputs = torch.cat([brake, throttle, steer], dim=-1)

            # 模型前向传播
            pred_output = model(inputs)

            # 获取预测的类别
            pred_class = pred_output.argmax(dim=1)  # 获取预测类别（取概率最大值的索引）
            label = label.long()          # 目标标签
            
            correct += (pred_class == label).sum().item()
            total += batch_data[0].shape[0]
            
        print(correct)
        accuracy = correct / total
        return accuracy

# Example usage
def main():
    # Hyperparameters
    seq_len = 120
    input_dim = 3
    embed_dim = 64
    
    print("============start training============")
    dataset = simulator_dataset(data_path)
    train_dataset, test_dataset = generate_split_dataset(dataset, data_sub_path, partition, False)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    loss = nn.CrossEntropyLoss()

    # Initialize model
    model_cnnt = CNNTransformerModel(input_dim=input_dim, embed_dim=embed_dim).cuda()
    for name, param in model_cnnt.named_parameters():
        print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
    optimizer = optim.Adam(model_cnnt.parameters(),lr = 1e-4, betas = (0.9, 0.999))
    
    recoder = SummaryWriter(log_dir=record_dir)
    for epoch in range(num_epoch):
        model_cnnt.train()
        print("training on epoch: ", epoch)
        total_loss = 0.0

        for i, batch_data in enumerate(tqdm(train_loader)):
            if batch_data[0].shape[0] == 1:
                continue
            
            brake = batch_data[0].to(cuda_device).float()
            steer = batch_data[1].to(cuda_device).float()
            throttle = batch_data[2].to(cuda_device).float()
            label = batch_data[4].to(cuda_device).float()
            # print("brake shape" + str(brake.shape))
            brake = brake.unsqueeze(-1)
            steer = steer.unsqueeze(-1)
            throttle = throttle.unsqueeze(-1)
            inputs = torch.cat([brake, throttle, steer], dim=-1)

            pred_output = model_cnnt(inputs)
            label = label.long()
            norm_loss = loss(pred_output, label)
            total_loss += norm_loss

            optimizer.zero_grad()
            norm_loss.backward()
            optimizer.step()
            # print("finish one batch")

        recoder.add_scalar('Loss/norm_loss', norm_loss.item(), epoch)
        recoder.add_scalar('Loss/loss', total_loss/len(train_loader), epoch)
        # recoder.add_scalar('Loss/train', loss.item(), epoch)
        print("testing")
        accuracy_train = test(model_cnnt,train_loader)
        accuracy_test = test(model_cnnt,test_loader)
        recoder.add_scalar('accuracy/trainset', accuracy_train, epoch)
        recoder.add_scalar('accuracy/testset', accuracy_test, epoch)

        
            
        if (epoch+1)%5 == 0:
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(model_cnnt.state_dict(), model_path + '/spatial_epoch_' + str(epoch+1) + '.pth')
    
    

def valid():
    input = torch.randn((32,120,3))
    model_cnnt = CNNTransformerModel(input_dim=3)
    
    output = model_cnnt(input)
    
    print(output.shape)


if __name__ == "__main__":
    main()
    # valid()
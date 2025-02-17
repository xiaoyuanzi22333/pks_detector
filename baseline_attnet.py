import os
import numpy as np
from MyDataset.simulator_dataset import simulator_dataset
from MyDataset.split_dataset import generate_split_dataset
import utils.utils as utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model.pks_attn_net.SpatialNet import SpatialNet
from model.pks_attn_net.TemporalNet import TemporalNet
from model.pks_attn_net.decoder import AtNet_decoder
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

fs = 30
time_split = 5
cuda_device = 0
data_path = './Data_' +str(time_split) + 's'
batch_size = 64
num_epoch = 50
record_dir = './logs_' +str(time_split) + 's_03'
model_path = './model_saved_' +str(time_split) + 's_03'



def train():
    print("============start training============")
    dataset = simulator_dataset(data_path)
    train_dataset, test_dataset = generate_split_dataset(dataset, True)
    # exit()
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model_spatial = SpatialNet(time_split*fs,400,time_split*fs).cuda()
    model_temporal = TemporalNet(30,[25,30],time_split*fs,5).cuda()
    model_decoder = AtNet_decoder(time_split*fs,150,2).cuda()

    print("Device: " +str(next(model_spatial.parameters()).device))
    optimizer_spatial = optim.Adam(model_spatial.parameters(),lr = 1e-5, betas = (0.5, 0.999))
    optimizer_temporal = optim.Adam(model_temporal.parameters(),lr = 1e-5, betas = (0.5, 0.999))
    optimizer_decoder = optim.Adam(model_decoder.parameters(),lr = 1e-5, betas = (0.5, 0.999))
    loss = nn.CrossEntropyLoss()

    recoder = SummaryWriter(log_dir=record_dir)

    for epoch in range(num_epoch):
        model_spatial.train()
        model_temporal.train()
        model_decoder.train()
        print("training on epoch: ", epoch)

        for i, batch_data in enumerate(tqdm(train_loader)):
            # if batch_data[0].shape[0] != batch_size:
            #     continue
            
            brake = batch_data[0].to(cuda_device).float()
            steer = batch_data[1].to(cuda_device).float()
            throttle = batch_data[2].to(cuda_device).float()
            label = batch_data[4].to(cuda_device).float()
            # print("brake shape" + str(brake.shape))

            spatial_output = model_spatial(brake, steer, throttle)
            temp_output = model_temporal(brake, steer, throttle)

            fused_output = spatial_output + temp_output
            pred_output = model_decoder(fused_output)

            # print("pred_output: " + str(pred_output.shape))            
            
            # print("pred_output: " + str(pred_output.shape)) 
            # print(pred_output)
            label = label.long()
            # print(label.shape)
            norm_loss = loss(pred_output, label)

            optimizer_spatial.zero_grad()
            optimizer_temporal.zero_grad()
            optimizer_decoder.zero_grad()
            norm_loss.backward()
            optimizer_spatial.step()
            optimizer_temporal.step()
            optimizer_decoder.step()

            # print("finish one batch")

        recoder.add_scalar('norm_Loss/train', norm_loss.item(), epoch)
        # recoder.add_scalar('Loss/train', loss.item(), epoch)
        print("testing")
        accuracy_train = test(model_spatial, model_temporal, model_decoder, train_dataset)
        accuracy_test = test(model_spatial, model_temporal, model_decoder, test_dataset)
        recoder.add_scalar('accuracy/trainset', accuracy_train, epoch)
        recoder.add_scalar('accuracy/testset', accuracy_test, epoch)
        
            
        if (epoch+1)%5 == 0:
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(model_spatial.state_dict(), model_path + '/spatial_epoch_' + str(epoch+1) + '.pth')
            torch.save(model_temporal.state_dict(), model_path + '/temporal_epoch_' + str(epoch+1) + '.pth')
            torch.save(model_decoder.state_dict(), model_path + '/decoder_epoch_' + str(epoch+1) + '.pth')




def test(model_spatial, model_temporal, model_decoder, test_dataset):
    model_spatial.to(cuda_device)
    model_spatial.eval()
    model_temporal.to(cuda_device)
    model_temporal.eval()
    model_decoder.to(cuda_device)
    model_decoder.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
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

            # 模型前向传播
            spatial_output = model_spatial(brake, steer, throttle)
            temp_output = model_temporal(brake, steer, throttle)
            fused_output = spatial_output + temp_output
            pred_output = model_decoder(fused_output)

            # 获取预测的类别
            pred_class = pred_output.argmax(dim=1)  # 获取预测类别（取概率最大值的索引）
            label = label.long()          # 目标标签
            
            correct += (pred_class == label).sum().item()
            # print(correct)
            total += batch_size

        accuracy = correct / total
        return accuracy




if __name__ == "__main__":
    train()
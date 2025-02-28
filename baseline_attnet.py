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
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description="load parsers")
parser.add_argument('--pth', type=str, default= "0000_00", help='log/model path')
parser.add_argument('--scl', type=int, default= 0, help='use scheduler')
parser.add_argument('--time_split', type=int, default= 3)
parser.add_argument('--time_interval', type=int,default= 1)
parser.add_argument('--batch_size', type=int,default= 32)
parser.add_argument('--epoch', type=int, default = 90)
parser.add_argument('--scl_step', type=int, default=60)
args = parser.parse_args()

print("loading params")
exp_name = args.pth
fs = 30
time_split = args.time_split
time_interval = args.time_interval
cuda_device = 0
data_path = './Data_' +str(time_split) + 's_' + str(time_interval) + 's'
batch_size = args.batch_size
num_epoch = args.epoch
record_dir = './logs/logs_' +str(time_split) + 's_' + exp_name
model_path = './model_saves/model_saved_' +str(time_split) + 's_' + exp_name
use_scheduler = False if args.scl==0 else True
scheduler_step = args.scl_step

print("time_split    : "  + str(time_split))
print("time_interval : "  + str(time_interval))
print("cuda_device   : "  + str(cuda_device))
print("data_path     : "  + str(data_path))
print("batch_size    : "  + str(batch_size))
print("num_epoch     : "  + str(num_epoch))
print("record_dir    : "  + str(record_dir))
print("model_path    : "  + str(model_path))
print("use_scheduler : "  + str(use_scheduler))
print("scheduler_step: "  + str(scheduler_step))


def train():
    print("============start training============")
    dataset = simulator_dataset(data_path)
    train_dataset, test_dataset = generate_split_dataset(dataset,time_split, time_interval, False)
    # exit()
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model_spatial = SpatialNet(time_split*fs,400,time_split*fs).cuda()
    model_temporal = TemporalNet(30,[25,30],time_split*fs,5).cuda()
    model_decoder = AtNet_decoder(time_split*fs,150,2).cuda()

    print("Device: " +str(next(model_spatial.parameters()).device))
    optimizer_spatial = optim.Adam(model_spatial.parameters(),lr = 1e-4, betas = (0.9, 0.999))
    optimizer_temporal = optim.Adam(model_temporal.parameters(),lr = 1e-4, betas = (0.9, 0.999))
    optimizer_decoder = optim.Adam(model_decoder.parameters(),lr = 1e-4, betas = (0.9, 0.999))
    
    if use_scheduler:
        scheduler_spatial = StepLR(optimizer_spatial, scheduler_step, 0.1)
        scheduler_temporal = StepLR(optimizer_temporal, scheduler_step, 0.1)
        scheduler_decoder = StepLR(optimizer_decoder, scheduler_step, 0.1)
    
    loss = nn.CrossEntropyLoss()
    
    recoder = SummaryWriter(log_dir=record_dir)
    recoder.add_text("Hyperparameters/Batchsize", str(batch_size), global_step=0)
    recoder.add_text("Hyperparameters/EpochNum", str(num_epoch), global_step=0)
    if use_scheduler:
        recoder.add_text("Hyperparameters/scheduleStep", str(scheduler_step), global_step=0)
    
    for epoch in range(num_epoch):
        model_spatial.train()
        model_temporal.train()
        model_decoder.train()
        print("training on epoch: ", epoch)
        total_loss = 0.0

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
            total_loss += norm_loss

            optimizer_spatial.zero_grad()
            optimizer_temporal.zero_grad()
            optimizer_decoder.zero_grad()
            norm_loss.backward()
            optimizer_spatial.step()
            optimizer_temporal.step()
            optimizer_decoder.step()
            # print("finish one batch")

        recoder.add_scalar('Loss/norm_loss', norm_loss.item(), epoch)
        recoder.add_scalar('Loss/loss', total_loss/len(train_loader), epoch)
        # recoder.add_scalar('Loss/train', loss.item(), epoch)
        print("testing")
        accuracy_train = test(model_spatial, model_temporal, model_decoder, train_dataset)
        accuracy_test = test(model_spatial, model_temporal, model_decoder, test_dataset)
        recoder.add_scalar('accuracy/trainset', accuracy_train, epoch)
        recoder.add_scalar('accuracy/testset', accuracy_test, epoch)
        recoder.add_scalar('learning_rate', optimizer_spatial.param_groups[0]['lr'], epoch)
        if use_scheduler:
            scheduler_spatial.step()
            scheduler_temporal.step()
            scheduler_decoder.step()
        
            
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
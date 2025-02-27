import os
import numpy as np
from MyDataset.simulator_dataset import simulator_dataset
from MyDataset.split_dataset import generate_split_dataset
import utils.utils as utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model.pksNet_old import pksNet
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

cuda_device = 0
data_path = './Data'
batch_size = 4
num_epoch = 50
record_dir = './logs_01'
model_path = './model_saved_01'



def train():
    print("============start training============")
    dataset = simulator_dataset(data_path)
    train_dataset, test_dataset = generate_split_dataset(dataset)
    # exit()
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = pksNet().cuda()

    print("Device: " +str(next(model.parameters()).device))
    optimizer = optim.Adam(model.parameters(),lr = 1e-4, betas = (0.5, 0.999))
    loss = nn.CrossEntropyLoss()

    recoder = SummaryWriter(log_dir=record_dir)

    for epoch in range(num_epoch):
        model.train()
        print("training on epoch: ", epoch)

        for i, batch_data in enumerate(tqdm(train_loader)):
            brake = batch_data[0].unsqueeze(1).to(cuda_device).float()
            steer = batch_data[1].unsqueeze(1).to(cuda_device).float()
            throttle = batch_data[2].unsqueeze(1).to(cuda_device).float()
            mask = batch_data[3].unsqueeze(1).to(cuda_device).float()
            label = batch_data[4].unsqueeze(1).to(cuda_device).float()

            pred_output = model(brake, steer, throttle)
            # print("pred_output: " + str(pred_output.shape)) 
            # print(pred_output)
            label = label.squeeze().long()
            # print(label.shape)

            
            mask_loss = loss(pred_output, label)

            final_loss = mask_loss * mask
            norm_loss = final_loss.sum() / mask.sum() # 归一化处理

            norm_loss.backward()
            optimizer.step()

        recoder.add_scalar('norm_Loss/train', norm_loss.item(), epoch)
        # recoder.add_scalar('Loss/train', loss.item(), epoch)
        print("testing")
        accuracy_train = test(model, train_dataset)
        accuracy_test = test(model, test_dataset)
        recoder.add_scalar('accuracy/trainset', accuracy_train, epoch)
        recoder.add_scalar('accuracy/testset', accuracy_test, epoch)
        
            
        if (epoch+1)%5 == 0:
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(model.state_dict(), model_path + '/pksNet_epoch_' + str(epoch+1) + '.pth')




def test(model, test_dataset):
    model.to(cuda_device)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    correct = 0  # 累计预测正确的样本数
    total = 0    # 累计所有有效样本数

    with torch.no_grad():
        for batch_data in test_loader:
            brake = batch_data[0].unsqueeze(1).to(cuda_device).float()
            steer = batch_data[1].unsqueeze(1).to(cuda_device).float()
            throttle = batch_data[2].unsqueeze(1).to(cuda_device).float()
            mask = batch_data[3].unsqueeze(1).to(cuda_device).float()
            label = batch_data[4].unsqueeze(1).to(cuda_device).float()

            # 模型前向传播
            pred_output = model(brake, steer, throttle)

            # 获取预测的类别
            pred_class = pred_output.argmax(dim=1)  # 获取预测类别（取概率最大值的索引）
            label = label.squeeze().long()          # 目标标签
            
            correct += (pred_class == label).sum().item()
            # print(correct)
            total += batch_size

        accuracy = correct / total
        return accuracy





if __name__ == "__main__":
    train()
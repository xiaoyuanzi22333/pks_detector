import os
import numpy as np
from MyDataset.simulator_dataset import simulator_dataset
import utils.utils as utils
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model import pksNet
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

cuda_device = 0
data_path = './Data'
batch_size = 4
num_epoch = 50
record_dir = './logs'
model_path = './model_saved'



def train():
    print("============start training============")
    train_dataset = simulator_dataset(data_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True )

    model = pksNet()

    print("Device: " + (pksNet.parameters()).device)
    optimizer = optim.Adam(model.parameters(),lr = 1e-3, betas = (0.5, 0.999))
    loss = nn.CrossEntropyLoss()

    recoder = SummaryWriter(log_dir=record_dir)



    for epoch in range(num_epoch):
        print("training on epoch: ", epoch)

        for i, batch_data in enumerate(tqdm(train_loader)):
            brake = batch_data[0].to(cuda_device)
            steer = batch_data[1].to(cuda_device)
            throttle = batch_data[2].to(cuda_device)
            mask = batch_data[3].to(cuda_device)
            label = batch_data[4].to(cuda_device)

            pred_output = model(brake, steer, throttle)
            mask_loss = loss(pred_output, label)

            loss = mask_loss * mask
            norm_loss = loss.sum() / mask.sum() # 归一化处理

            norm_loss.backward()
            optimizer.step()

        recoder.add_scalar('norm_Loss/train', norm_loss.item(), epoch)
        recoder.add_scalar('Loss/train', loss.item(), epoch)
            
        if epoch%5 == 0:
            if not os.path.exists(model_path):
                os.mkdir(model_path)
                torch.save(model.state_dict(), model_path + '/model_' + str(epoch) + '.pth')



if __name__ == "__main__":
    train()
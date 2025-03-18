import os
import numpy as np
from MyDataset.simulator_dataset import simulator_dataset
from MyDataset.split_dataset import generate_split_dataset
import utils.utils as utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model.pks_attn_net.SpatialNet_new import SpatialNet_new
from model.pks_attn_net.TemporalNet_new import TemporalNet_new
from model.pks_attn_net.TemporalNet_new_2 import TemporalNet_new_2
from model.pks_attn_net.decoder import AtNet_decoder
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import random



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
data_sub_path = 'Data_map' + str(args.map) + '_' +str(time_split) + 's_' + str(time_interval) + 's'
data_path = './Data_'+str(time_split)+'s'+str(time_interval)+'s/'+ data_sub_path
batch_size = args.batch_size
num_epoch = args.epoch
record_dir = './logs_base' + '/logs_' +str(time_split) + 's_' + exp_name
model_path = './model_ch/model_saved_' +str(time_split) + 's_' + exp_name
use_scheduler = False if args.scl==0 else True
scheduler_step = args.scl_step
num_chd = args.num_chd
rand_seed = args.rand_seed
partition = args.partition

# if rand_seed != 0:
#     # 动态生成一个随机种子
#     seed = random.randint(0, 1000)
# else :
#     # 使用固定的种子
#     seed = 71

# # 设置 Python 的随机种子
# random.seed(seed)
# # 设置 NumPy 的随机种子
# np.random.seed(seed)
# # 设置 PyTorch 的随机种子
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置种子
# # 确保 CUDA 的计算是确定性的（可能会牺牲一些性能）
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


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
print("num_chd       : "  + str(num_chd))
print("use_rand_seed : "  + str(rand_seed))
# print("seed_number   : "  + str(seed))



def train():
    print("============start training============")
    dataset = simulator_dataset(data_path)
    train_dataset, test_dataset = generate_split_dataset(dataset,data_sub_path,partition, False)
    # exit()
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model_spatial = SpatialNet_new(1,32,64, num_chd=num_chd).cuda().cuda()
    # model_temporal = TemporalNet_new(1, [32,32], 64).cuda() #old
    model_temporal = TemporalNet_new_2(1, [32,32], 64, data_len=fs*time_split, num_chd=num_chd).cuda() #new
    model_decoder = AtNet_decoder(64,16,2).cuda()

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
            if batch_data[0].shape[0] == 1:
                continue
            
            brake = batch_data[0].to(cuda_device).float()
            steer = batch_data[1].to(cuda_device).float()
            throttle = batch_data[2].to(cuda_device).float()
            label = batch_data[4].to(cuda_device).float()
            # print("brake shape" + str(brake.shape))
            inputs = [brake, throttle, steer]

            spatial_output = model_spatial(inputs)
            temp_output = model_temporal(inputs)
            
            fused_output = temp_output + spatial_output
            pred_output = model_decoder(fused_output)

            label = label.long()
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
            inputs = [brake, throttle, steer]


            # 模型前向传播
            spatial_output = model_spatial(inputs)
            temp_output = model_temporal(inputs)
            fused_output = temp_output + spatial_output
            pred_output = model_decoder(fused_output)

            # 获取预测的类别
            pred_class = pred_output.argmax(dim=1)  # 获取预测类别（取概率最大值的索引）
            label = label.long()          # 目标标签
            
            correct += (pred_class == label).sum().item()
            # print(correct)
            total += batch_data[0].shape[0]

        accuracy = correct / total
        return accuracy




if __name__ == "__main__":
    train()
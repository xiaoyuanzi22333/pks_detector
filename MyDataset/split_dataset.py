import os
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Subset
import json



def generate_split_dataset(dataset,data_path,partition=100,new=False):
    length = len(dataset)
    json_file = "./idx_folder/idx_" +data_path+".json"
    if not os.path.exists(json_file) or new:
        # 创建全新的索引
        train_size = int(0.85 * length)
        test_size = length - train_size
        
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_indices = train_dataset.indices
        test_indices = test_dataset.indices
        
        # 根据 partition 参数调整训练集大小
        adjusted_train_size = int(len(train_indices) * (partition / 100))
        adjusted_train_indices = train_indices[:adjusted_train_size]  # 取前 partition% 的索引

        print(f"trainset: {len(train_indices)}")
        print(f"testset : {len(test_indices)}")
        print(f"adjusted trainset: {len(adjusted_train_indices)} (partition={partition}%)")
        
        with open(json_file, "w") as f:
            json.dump({"train_indices": train_indices, 
                "test_indices": test_indices}, f)
        
        # 使用调整后的索引创建训练集
        adjusted_train_dataset = Subset(dataset, adjusted_train_indices)
        return adjusted_train_dataset, test_dataset
    
    else:
        # 加载保存的索引
        try:
            with open(json_file, "r") as f:
                indices = json.load(f)
        except:
            return generate_split_dataset(dataset,data_path,partition,new)
        
        else:
            train_indices = indices["train_indices"]
            test_indices = indices["test_indices"]

            # 使用 Subset 创建分割后的数据集
            # 根据 partition 参数调整训练集大小
            adjusted_train_size = int(len(train_indices) * (partition / 100))
            adjusted_train_indices = train_indices[:adjusted_train_size]  # 取前 partition% 的索引
            
            adjusted_train_dataset = Subset(dataset, adjusted_train_indices)
            test_dataset = Subset(dataset, test_indices)
            
            print(f"original trainset: {len(train_indices)}")
            print(f"adjusted trainset: {len(adjusted_train_indices)} (partition={partition}%)")
            print(f"testset: {len(test_indices)}")
            # print(train_indices)
            # print(test_indices)
            
            return adjusted_train_dataset, test_dataset
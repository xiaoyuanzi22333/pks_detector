import os
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Subset
import json



def generate_split_dataset(dataset, new=False):
    length = len(dataset)
    if not os.path.exists("dataset_split_indices.json") or new:
        # 创建全新的索引
        train_size = int(0.85 * length)
        test_size = length - train_size
        
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_indices = train_dataset.indices
        test_indices = test_dataset.indices

        print(f"trainset: {len(train_indices)}")
        print(f"testset : {len(test_indices)}")
        
        with open("dataset_split_indices.json", "w") as f:
            json.dump({"train_indices": train_indices, 
                "test_indices": test_indices}, f)
        
        return train_dataset, test_dataset
    
    else:
        # 加载保存的索引
        try:
            with open("dataset_split_indices.json", "r") as f:
                indices = json.load(f)
        except:
            return generate_split_dataset(dataset, True)
        
        else:
            train_indices = indices["train_indices"]
            test_indices = indices["test_indices"]

            # 使用 Subset 创建分割后的数据集
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)
            
            print(f"trainset: {len(train_indices)}")
            print(f"testset : {len(test_indices)}")
            
            # print(train_indices)
            # print(test_indices)
            
            return train_dataset, test_dataset
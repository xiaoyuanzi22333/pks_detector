import os
import matplotlib.pyplot as plt
import numpy as np
from MyDataset.simulator_dataset import simulator_getter


data_path = './Data'


if __name__ == "__main__":
    print("baseline")

    training_dataset = simulator_getter(data_path)
    max = 0
    for brake, steer, throttle in training_dataset:
        print(len(brake))
        if len(brake) > max:
            max = len(brake)
    
    print(max)

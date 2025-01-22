import os
import matplotlib.pyplot as plt
import numpy as np
from MyDataset.simulator_dataset import simulator_getter


data_path = './Data'


if __name__ == "__main__":
    print("baseline")

    training_dataset = simulator_getter(data_path)

    # print(len(training_dataset))
    for brake, steer, throttle in training_dataset: #  dtype = np.mdarray
        # print(len(brake))
        # print(len(steer))
        # print(len(throttle))
        x = np.arange(len(steer))
        plt.figure(1)
        plt.plot(x, steer, label='steer', color='blue', linestyle='--')
        plt.figure(2)
        plt.plot(x, brake, label='brake', color='green', linestyle='-.')
        plt.figure(3)
        plt.plot(x, throttle, label='throttle', color='red', linestyle=':')


    plt.figure(1)
    plt.savefig('./data_visualize/steer_data.png')
    plt.figure(2)
    plt.savefig('./data_visualize/brake_data.png')
    plt.figure(3)
    plt.savefig('./data_visualize/throttle_data.png')

    # plt.show()
import time
import pdb
import torch
import matplotlib.pyplot as plt
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from model import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    # create a dataset
    dataset = dataloader(opt)
    
    
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    sample_idx = torch.randint(int(len(dataset)/4), size = (9,))
    count = 0
    for i, data in enumerate(dataset):
        if i in sample_idx:
            print(i)
            count += 1
            figure.add_subplot(rows, cols, count)
            plt.title(data['label'][0])
            plt.axis("off")
            img = data['img'][0].permute(1, 2, 0)
            plt.imshow(img.squeeze())
        if count == cols * rows: break
    plt.show()
        

    
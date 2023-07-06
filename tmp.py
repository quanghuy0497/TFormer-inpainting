import time
import pdb
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from model import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    # create a dataset
    dataset = dataloader(opt)
    pdb.set_trace()

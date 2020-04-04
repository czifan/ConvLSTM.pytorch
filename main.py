# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from configs.build_config import build_config
from utils.dataset import MovingMNISTDataset
from networks.ConvLSTM import ConvLSTM
import torch
from torch.utils.data import DataLoader
from utils.utils import save_checkpoint
from utils.utils import build_logging
from utils.functions import train
from utils.functions import valid
from utils.functions import test
from networks.CrossEntropyLoss import CrossEntropyLoss
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='5x5_5x5_128_5x5_64_5x5_64')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    name = args.config
    config = build_config(name)
    logger = build_logging(config)
    model = ConvLSTM(config).to(config.device)
    #criterion = CrossEntropyLoss().to(config.device)
    criterion = torch.nn.MSELoss().to(config.device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    train_dataset = MovingMNISTDataset(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                            num_workers=config.num_workers, shuffle=True, pin_memory=True)
    valid_dataset = MovingMNISTDataset(config, split='valid')
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
                            num_workers=config.num_workers, shuffle=False, pin_memory=True)
    test_dataset = MovingMNISTDataset(config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size,
                            num_workers=config.num_workers, shuffle=False, pin_memory=True)
    train_records, valid_records, test_records = [], [], []
    for epoch in range(config.epochs):
        epoch_records = train(config, logger, epoch, model, train_loader, criterion, optimizer)
        train_records.append(np.mean(epoch_records['loss']))
        valid_records = valid(config, logger, epoch, model, valid_loader, criterion)
        valid_records.append(np.mean(epoch_records['loss']))
        test_records = test(config, logger, epoch, model, test_loader, criterion)
        test_records.append(np.mean(epoch_records['loss']))
        plt.plot(range(epoch + 1), train_records, label='train')
        plt.plot(range(epoch + 1), valid_records, label='valid')
        plt.plot(range(epoch + 1), test_records, label='test')
        plt.legend()
        plt.savefig(os.path.join(config.output_dir, '{}.png'.format(name)))
        plt.close()

if __name__ == '__main__':
    main()
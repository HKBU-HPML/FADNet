from __future__ import print_function
import os
import argparse
import datetime
import random
import torch
import logging
import shutil
import numpy as np


layer = 'dispnetc.conv1.0.weight'
saved_diretory = 'weights'
def parse_model(path, dnn='fadnet'):
    epochs=20
    nround=4
    for i in range(nround):
        for j in range(0, epochs):
            pretrain = '%s/fadnet_%d_%d.pth' % (path, i, j)
            print('model: ', pretrain)
            model_data = torch.load(pretrain)
            weight = model_data['state_dict'][layer]
            weight = weight.view(-1).cpu().numpy()
            np.save('%s/%s_%s_%d_%d.npy' % \
                    (saved_diretory, dnn, layer, i, j), weight)


def plot_weight(path, dnn='fadnet'):
    import matplotlib
    import matplotlib.pyplot as plt
    epochs=20
    nround=4
    for i in range(nround):
        for j in range(0, epochs):
            weight = np.load('%s/%s_%s_%d_%d.npy' % \
                    (saved_diretory, dnn, layer, i, j)).abs().view(-1).cpu().numpy()
            d = weight.numel()
            sorted_weight = np.sort(weight)[::-1]
            ax.plot(np.arange(1, d+1), sorted_weight, label='round-%d-epoch-%d'%(i,j))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parse_model('models/fadnet-sceneflow/')

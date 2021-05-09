from __future__ import print_function
import os
import argparse
import datetime
import random
import torch
import logging
import shutil

import torch.nn as nn
import torch.backends.cudnn as cudnn
import horovod.torch as hvd

from utils.common import *
from dltrainer import DisparityTrainer
from net_builder import SUPPORT_NETS
from losses.multiscaleloss import multiscaleloss
import wandb

cudnn.benchmark = True

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if state['epoch'] % 100 == 0:
        torch.save(state, os.path.join(opt.outf,filename))
    if is_best:
        torch.save(state, os.path.join(opt.outf,'model_best.pth'))
        #shutil.copyfile(os.path.join(opt.outf,filename), os.path.join(opt.outf,'model_best.pth'))

def main(opt):

    rank = hvd.rank()
    ngpu= hvd.size()
    torch.cuda.set_device(rank%opt.nwpernode)

    # load the training loss scheme
    loss_json = load_loss_scheme(opt.loss)
    train_round = loss_json["round"]
    loss_scale = loss_json["loss_scale"]
    loss_weights = loss_json["loss_weights"]
    epoches = loss_json["epoches"]
    logger.info(loss_weights)

    #high_res_EPE = multiscaleloss(scales=1, downscale=1, weights=(1), loss='L1', sparse=False)
    # initialize a trainer
    trainer = DisparityTrainer(opt.net, opt.lr, [0], opt.dataset, opt.trainlist, opt.vallist, opt.datapath, opt.batch_size, opt.maxdisp, opt.model, ngpu=ngpu, rank=rank, hvd=True)

    logger.info('Broadcast parameters....')
    hvd.broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')


    optimizer = hvd.DistributedOptimizer(trainer.optimizer, named_parameters=trainer.net.named_parameters())
    trainer.optimizer = optimizer

    # validate the pretrained model on test data
    best_EPE = -1
    if trainer.is_pretrain:
        with torch.no_grad():
            best_EPE = trainer.validate()

    start_epoch = opt.startEpoch

    is_sparse = False
    if 'kitti' in opt.dataset:
        is_sparse = True
    for r in range(opt.startRound, train_round):

        criterion = multiscaleloss(loss_scale, 1, loss_weights[r], loss='L1', sparse=is_sparse)
        trainer.set_criterion(criterion)
        end_epoch = epoches[r]
        #end_epoch = min(epoches[r], opt.endEpoch)

        logger.info('round %d: %s' % (r, str(loss_weights[r])))
        logger.info('num of epoches: %d' % end_epoch)
        logger.info('\t'.join(['epoch', 'time_stamp', 'train_loss', 'train_EPE', 'EPE', 'lr']))
        for i in range(start_epoch, end_epoch):
            avg_loss, avg_EPE = trainer.train_one_epoch(i)
            with torch.no_grad():
                val_EPE = trainer.validate()
            if rank == 0:
                is_best = best_EPE < 0 or val_EPE < best_EPE
                if is_best:
                    best_EPE = val_EPE

                save_checkpoint({
                    'round': r + 1,
                    'epoch': i + 1,
                    'arch': 'dispnet',
                    'state_dict': trainer.get_model(),
                    'best_EPE': best_EPE,    
                }, is_best, '%s_%d_%d.pth' % (opt.net, r, i))
        
                logger.info('Validation [round:%d,epoch:%d]: '%(r,i)+'\t'.join([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(avg_loss), str(float(avg_EPE)), str(float(val_EPE)), str(trainer.current_lr)]))

                wandb.log({'train-epe': float(avg_EPE), 'val-epe': float(val_EPE), 'epoch': i+1, 'round': r+1, 'curr_lr': trainer.current_lr})
        start_epoch = 0
        trainer.train_iter = 0


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicates the name of net', default='simplenet', choices=SUPPORT_NETS)
    parser.add_argument('--ngpus', type=int, default=1, help='# of GPUs per node')
    parser.add_argument('--nwpernode', type=int, default=4, help='Number of workers per node')
    parser.add_argument('--loss', type=str, help='indicates the loss scheme', default='simplenet_flying')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd, alpha parameter for adam. default=0.9')
    parser.add_argument('--beta', type=float, default=0.999, help='beta parameter for adam. default=0.999')
    parser.add_argument('--cuda', action='store_true', help='enables, cuda')
    parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--model', type=str, help='model for finetuning', default='')
    parser.add_argument('--startRound', type=int, help='the round number to start training, useful of lr scheduler', default='0')
    parser.add_argument('--startEpoch', type=int, help='the epoch number to start training, useful of lr scheduler', default='0')
    parser.add_argument('--logFile', type=str, help='logging file', default='./train.log')
    parser.add_argument('--showFreq', type=int, help='display frequency', default='100')
    parser.add_argument('--flowDiv', type=float, help='the number by which the flow is divided.', default='1.0')
    parser.add_argument('--maxdisp', type=int, help='disparity search range.', default='-1')
    parser.add_argument('--dataset', type=str, help='provide the dataset name', default='sceneflow')
    parser.add_argument('--datapath', type=str, help='provide the root path of the data', default='data/')
    parser.add_argument('--trainlist', type=str, help='provide the train file (with file list)', default='FlyingThings3D_release_TRAIN.list')
    parser.add_argument('--vallist', type=str, help='provide the val file (with file list)', default='FlyingThings3D_release_TEST.list')
    parser.add_argument('--augment', type=int, help='if augment data in training', default=0)
    
    opt = parser.parse_args()

    hvd.init()
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    logfile = opt.logFile.replace('.log', '-%d.log' % hvd.rank())
    if hvd.rank() == 0:
        hdlr = logging.FileHandler(logfile)
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 
    logger.info('Configurations: %s', opt)
    
    if hvd.rank() == 0:
        wandb.init(project='fadnet', entity='hpml-hkbu', name=logfile, config=opt)
    
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    logger.info("Random Seed: %s", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    
    if torch.cuda.is_available() and not opt.cuda:
        logger.warning("WARNING: You should run with --cuda since you have a CUDA device.")
    main(opt)


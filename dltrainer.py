from __future__ import print_function
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from net_builder import build_net
#from dataset import DispDataset
from dataloader.SceneFlowLoader import DispDataset
from utils.AverageMeter import AverageMeter
from utils.common import logger
from losses.multiscaleloss import EPE
from utils.preprocess import scale_disp

class DisparityTrainer(object):
    def __init__(self, net_name, lr, devices, trainlist, vallist, datapath, batch_size, pretrain=None):
        super(DisparityTrainer, self).__init__()
        self.net_name = net_name
        self.lr = lr
        self.current_lr = lr
        self.devices = devices
        self.devices = [int(item) for item in devices.split(',')]
        ngpu = len(devices)
        self.ngpu = ngpu
        self.trainlist = trainlist
        self.vallist = vallist
        self.datapath = datapath
        self.batch_size = batch_size
        self.pretrain = pretrain 

        #self.criterion = criterion
        self.criterion = None
        self.epe = EPE
        self.initialize()

    def _prepare_dataset(self):

        train_dataset = DispDataset(txt_file = self.trainlist, root_dir = self.datapath, phase='train')
        test_dataset = DispDataset(txt_file = self.vallist, root_dir = self.datapath, phase='test')
        
        self.train_loader = DataLoader(train_dataset, batch_size = self.batch_size, \
                                shuffle = True, num_workers = 16, \
                                pin_memory = True)
        
        self.test_loader = DataLoader(test_dataset, batch_size = 4, \
                                shuffle = False, num_workers = 4, \
                                pin_memory = True)
        self.num_batches_per_epoch = len(self.train_loader)

    def _build_net(self):
        #self.net = build_net(self.net_name)(batchNorm=True, using_resblock=True)
        #self.net = build_net(self.net_name)(len(self.devices), batchNorm=True)
        self.net = build_net(self.net_name)(len(self.devices), False, True)
        self.is_pretrain = False

        if self.ngpu > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.devices).cuda()
        else:
            self.net.cuda()

        if self.pretrain == '':
            logger.info('Initial a new model...')
        else:
            if os.path.isfile(self.pretrain):
                model_data = torch.load(self.pretrain)
                logger.info('Load pretrain model: %s', self.pretrain)
                if 'state_dict' in model_data.keys():
                    self.net.load_state_dict(model_data['state_dict'])
                else:
                    self.net.load_state_dict(model_data)
                self.is_pretrain = True
            else:
                logger.warning('Can not find the specific model %s, initial a new model...', self.pretrain)


    def _build_optimizer(self):
        beta = 0.999
        momentum = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr,
                                        betas=(momentum, beta))

    def initialize(self):
        self._build_net()
        self._prepare_dataset()
        self._build_optimizer()

    def adjust_learning_rate(self, epoch):
        cur_lr = self.lr / (2**(epoch // 10))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
        self.current_lr = cur_lr
        return cur_lr

    def set_criterion(self, criterion):
        self.criterion = criterion
 
    def train_one_epoch(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        flow2_EPEs = AverageMeter()
        # switch to train mode
        self.net.train()
        end = time.time()
        self.adjust_learning_rate(epoch)
        for i_batch, sample_batched in enumerate(self.train_loader):
         
            left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(), requires_grad=False)
            right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(), requires_grad=False)
            input = torch.cat((left_input, right_input), 1)

            target = sample_batched['gt_disp']
            target = target.cuda()

            input_var = torch.autograd.Variable(input, requires_grad=False)
            target_var = torch.autograd.Variable(target, requires_grad=False)
            data_time.update(time.time() - end)

            if self.net_name == "dispnetcres":
                output_net1, output_net2 = self.net(input_var)
                loss_net1 = self.criterion(output_net1, target_var)
                loss_net2 = self.criterion(output_net2, target_var)
                loss = loss_net1 + loss_net2
                output_net2_final = output_net2[0]
                flow2_EPE = self.epe(output_net2_final, target_var)
            else:
                output = self.net(input_var)
                loss = self.criterion(output, target_var)
                if type(loss) is list:
                    loss = np.sum(loss)
                if type(output) is list:
                    flow2_EPE = self.epe(output[0], target_var)
                else:
                    flow2_EPE = self.epe(output, target_var)

            # record loss and EPE
            losses.update(loss.data.item(), target.size(0))
            flow2_EPEs.update(flow2_EPE.data.item(), target.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i_batch % 10 == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})'.format(
                  epoch, i_batch, self.num_batches_per_epoch, batch_time=batch_time, 
                  data_time=data_time, loss=losses, flow2_EPE=flow2_EPEs))

        return losses.avg, flow2_EPEs.avg

    def validate(self):
        batch_time = AverageMeter()
        flow2_EPEs = AverageMeter()
        losses = AverageMeter()
        # switch to evaluate mode
        self.net.eval()
        end = time.time()
        for i, sample_batched in enumerate(self.test_loader):
            left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(), requires_grad=False)
            right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(), requires_grad=False)
            input = torch.cat((left_input, right_input), 1)
    
            target = sample_batched['gt_disp']
            target = target.cuda()
            input_var = torch.autograd.Variable(input, requires_grad=False)
            target_var = torch.autograd.Variable(target, requires_grad=False)

            if self.net_name == 'dispnetcres':
                output_net1, output_net2 = self.net(input_var)

                output_net1 = output_net1.squeeze(1)
                output_net1 = scale_disp(output_net1.data.cpu().numpy(), (output_net1.size()[0], 540, 960))
                output_net1 = torch.from_numpy(output_net1).unsqueeze(1).cuda()
                output_net2 = output_net2.squeeze(1)
                output_net2 = scale_disp(output_net2.data.cpu().numpy(), (output_net2.size()[0], 540, 960))
                output_net2 = torch.from_numpy(output_net2).unsqueeze(1).cuda()

                loss_net1 = self.epe(output_net1, target_var)
                loss_net2 = self.epe(output_net2, target_var)
                loss = loss_net1 + loss_net2
                flow2_EPE = self.epe(output_net2, target_var)
            else:
                output = self.net(input_var)
                loss = self.criterion(output, target_var)

                if type(loss) is list:
                    loss = loss[0]
                if type(output) is list:
                    flow2_EPE = self.epe(output[0], target_var)
                else:
                    flow2_EPE = self.epe(output, target_var)

            # record loss and EPE
            losses.update(loss.data.item(), target.size(0))
            flow2_EPEs.update(flow2_EPE.data.item(), target.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % 10 == 0:
                logger.info('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                      .format(i, len(self.test_loader), batch_time.val, flow2_EPEs.val))
        logger.info(' * EPE {:.3f}'.format(flow2_EPEs.avg))
        return flow2_EPEs.avg

    def get_model(self):
        return self.net.state_dict()



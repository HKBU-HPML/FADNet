from __future__ import print_function
import argparse
import os, shutil, sys
import numpy as np
import time, datetime
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

from dataset import *
#from dispnet_v2 import *
from dispnet import *
from multiscaleloss import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import csv


parser = argparse.ArgumentParser()
parser.add_argument('--domain_transfer', type=int, help='if open the function of domain transer', default=0)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd, alpha parameter for adam. default=0.9')
parser.add_argument('--beta', type=float, default=0.999, help='beta parameter for adam. default=0.999')
parser.add_argument('--cuda', action='store_true', help='enables, cuda')
parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
# parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--model', type=str, help='model for finetuning', default='')
parser.add_argument('--startEpoch', type=int, help='the epoch number to start training, useful of lr scheduler', default='0')
parser.add_argument('--endEpoch', type=int, help='the epoch number to end training', default='50')
parser.add_argument('--logFile', type=str, help='logging file', default='./train.log')
parser.add_argument('--showFreq', type=int, help='display frequency', default='100')
parser.add_argument('--flowDiv', type=float, help='the number by which the flow is divided.', default='1.0')
parser.add_argument('--datapath', type=str, help='provide the root path of the data', default='data')
parser.add_argument('--trainlist', type=str, help='provide the train file (with file list)', default='FlyingThings3D_release_TRAIN.list')
parser.add_argument('--tdlist', type=str, help='provide the target domain file (with file list)', default='real_sgm_release.list')
parser.add_argument('--vallist', type=str, help='provide the val file (with file list)', default='FlyingThings3D_release_TEST.list')
parser.add_argument('--augment', type=int, help='if augment data in training', default=0)

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You should run with --cuda since you have a CUDA device.")

# input transform, normalize with 255
input_transform = transforms.Compose([
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        # transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
        ])

# disparity transform, divided by flowDiv, 1.0 with batchNorm seems to yield good results
target_transform = transforms.Compose([
        transforms.Normalize(mean=[0],std=[opt.flowDiv])
        ])
augment= True if opt.augment==1 else False
print('augment: ', augment)
train_dataset = DispDataset(txt_file = opt.trainlist, root_dir = opt.datapath, transform=[input_transform, target_transform], augment=augment)
test_dataset = DispDataset(txt_file = opt.vallist, root_dir = opt.datapath, transform=[input_transform, target_transform])

# for i in range(3):
#     sample = train_dataset[i]
#     print(i, sample['img_left'].size(), type(sample['img_left']), \
#              sample['img_right'].size(), type(sample['img_right']), \
#              sample['gt_disp'].size(), type(sample['gt_disp'])  \
#          )


train_loader = DataLoader(train_dataset, batch_size = opt.batchSize, \
                        shuffle = True, num_workers = opt.workers, \
                        pin_memory = True)

target_loader = None
opt.domain_transfer = False if opt.domain_transfer == 0 else True
if opt.domain_transfer:
    td_dataset = DispDataset(txt_file = opt.tdlist, root_dir = opt.datapath, transform=[input_transform, target_transform])
    td_loader  = DataLoader(td_dataset, batch_size = opt.batchSize, \
                        shuffle = True, num_workers = opt.workers, \
                        pin_memory = True)


test_loader = DataLoader(test_dataset, batch_size = opt.batchSize, \
                        shuffle = False, num_workers = opt.workers, \
                        pin_memory = True)

# use multiple-GPUs training
devices = [int(item) for item in opt.devices.split(',')]
ngpu = len(devices)
#net = DispNetCSRes(ngpu, False, True)
# net = DispNetC(ngpu, True)

# Shaohuai
if opt.domain_transfer:
    net = DispNetCSResWithDomainTransfer(ngpu, False, True)
else:
    net = DispNetCSRes(ngpu, False, True, input_channel=4)
print(net)

#start_epoch = 0
#model_data = torch.load('./dispC_epoch_29.pth')
#print(model_data.keys())
#if 'state_dict' in model_data.keys():
#    net.load_state_dict(model_data['state_dict'])
#else:
#    net.load_state_dict(model_data)
start_epoch = opt.startEpoch
end_epoch = opt.endEpoch
# pre-loading the model according to args
if opt.model == '': 
    print('Initial a new model...')
else:
    if os.path.isfile(opt.model):
        model_data = torch.load(opt.model)
        print(model_data.keys())
        if 'state_dict' in model_data.keys():
            net.load_state_dict(model_data['state_dict'])
        else:
            net.load_state_dict(model_data)
    else:
        print('Can not find the specific model, initial a new model...')


net = torch.nn.DataParallel(net, device_ids=devices).cuda()

loss_weights = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
#loss_weights = (0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005)


# qiang
#loss_weights = (0.6, 0.32, 0.08, 0.04, 0.02, 0.01, 0.005)
#loss_weights = (0.8, 0.16, 0.04, 0.02, 0.01, 0.005, 0.0025)
#loss_weights = (1, 0, 0, 0, 0, 0, 0)

# shaohuai for girl data
#loss_weights = (0.8, 0.1, 0.04, 0.04, 0.02, 0.01, 0.005)
#loss_weights = (0.9, 0.05, 0.02, 0.02, 0.01, 0.005, 0.0025)
#loss_weights = (0.99, 0.005, 0.002, 0.002, 0.001, 0.001, 0.0005)

# shaohuai for kitti data
#loss_weights = (0.6, 0.32, 0.08, 0.04, 0.02, 0.01, 0.005)

criterion = multiscaleloss(7, 1, loss_weights, loss='L1', sparse=False)
high_res_EPE = multiscaleloss(scales=1, downscale=1, weights=(1), loss='L1', sparse=False)

print('=> setting {} solver'.format('adam'))
init_lr = opt.lr
param_groups = [{'params': net.module.bias_parameters(), 'weight_decay': 0},
                    {'params': net.module.weight_parameters(), 'weight_decay': 4e-4}]

# optimizer = torch.optim.SGD(param_groups, init_lr,
#                                     momentum=0.9)
optimizer = torch.optim.Adam(param_groups, init_lr,
                                    betas=(opt.momentum, opt.beta))

# write opt and network
with open(os.path.join('logs', opt.logFile), 'a') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow([str(opt)])
    writer.writerow([str(net)])

# write table header
with open(os.path.join('logs', opt.logFile), 'a') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(['epoch', 'time_stamp', 'train_loss', 'train_EPE', 'EPE', 'lr'])

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    cur_lr = init_lr / (2**(epoch // 60))
    #cur_lr = init_lr / (2**(epoch // 20))
    #cur_lr = init_lr / (2**(epoch // 5))
    # if epoch != 0 and epoch % 10 == 0:
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i_batch, sample_batched in enumerate(train_loader):
        
        input = torch.cat((sample_batched['img_left'], sample_batched['img_right']), 1)

        target = sample_batched['gt_disp']
        # print(input.size())

        data_time.update(time.time() - end)
        target = target.cuda()
        input = input.cuda()
        # print(i_batch, input.size(), target.size())
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output and loss
        # output = model(input_var)
        # loss = criterion(output, target_var)
        # flow2_EPE = high_res_EPE(output[0], target_var) * opt.flowDiva

        output_net1, output_net2 = model(input_var)
        loss_net1 = criterion(output_net1, target_var)
        loss_net2 = criterion(output_net2, target_var)
        loss = loss_net1 + loss_net2
        flow2_EPE = high_res_EPE(output_net2[0], target_var) * opt.flowDiv
        # output = model(input_var)
        # loss = criterion(output, target_var)
        # flow2_EPE = high_res_EPE(output[0], target_var) * opt.flowDiv

        #output_net1, output_net2 = model(input_var)
        #loss_net1 = criterion(output_net1, target_var)
        #loss_net2 = criterion(output_net2, target_var)
        #loss = loss_net1 + loss_net2
        #flow2_EPE = high_res_EPE(output_net1[0], target_var) * opt.flowDiv
        
        # record loss and EPE
        losses.update(loss.data.item(), target.size(0))
        flow2_EPEs.update(flow2_EPE.data.item(), target.size(0))

        # if np.isnan(flow2_EPE.data[0]):
        #     sys.exit(-1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # print(loss.grad)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i_batch % opt.showFreq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
              'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})'.format(
              epoch, i_batch, len(train_loader), batch_time=batch_time, 
              data_time=data_time, loss=losses, flow2_EPE=flow2_EPEs))
 
	# debug  	
	#if i_batch >= 3:
	#    break

    return losses.avg, flow2_EPEs.avg
    # return losses.avg

def train_with_domain_transfer(source_loader, target_loader, model, optimizer, epoch):
    domain_criterion = nn.NLLLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()
    domain_errs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    len_dataloader = min(len(source_loader), len(target_loader))
    len_targetloader = len(target_loader)
    nbatch_of_target = len_targetloader / opt.batchSize

    for i_batch, sample_batched in enumerate(source_loader):
        
        #print('orig_sample_batched: ', sample_batched)
        #model.zero_grad()
        p = float(i_batch + epoch * len_dataloader) / opt.endEpoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        input = torch.cat((sample_batched['img_left'], sample_batched['img_right']), 1)
        #print('sample_batched: ', input)
        target = sample_batched['gt_disp']
        data_time.update(time.time() - end)
        target = target.cuda()
        input = input.cuda()
        domain_label = torch.zeros(opt.batchSize)
        domain_label = domain_label.long().cuda()

        # training model using source data
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        domain_label_var = torch.autograd.Variable(domain_label)

        output_net1, output_net2, domain_output  = model(input_var, alpha=alpha)
        loss_net1 = criterion(output_net1, target_var)
        loss_net2 = criterion(output_net2, target_var)
        loss = loss_net1 + loss_net2
        flow2_EPE = high_res_EPE(output_net2[0], target_var) * opt.flowDiv
        losses.update(loss.data.item(), target.size(0))
        flow2_EPEs.update(flow2_EPE.data.item(), target.size(0))

        source_domain_loss = domain_criterion(domain_output, domain_label_var)

        # training model using target data
        if i_batch % nbatch_of_target == 0:
            target_loader_iter = iter(target_loader)
        td_batched = target_loader_iter.next()
        #print('td_batched: ', td_batched)
        td_input = torch.cat((td_batched['img_left'], td_batched['img_right']), 1)
        #print('tdshape: ', td_input)
        td_input = td_input.cuda()
        domain_label = torch.ones(opt.batchSize)
        domain_label = domain_label.long().cuda()
        td_input_var = torch.autograd.Variable(td_input)
        domain_label_var = torch.autograd.Variable(domain_label)
        _, _, target_domain_output  = model(td_input_var, alpha=alpha)
        target_domain_loss = domain_criterion(target_domain_output, domain_label_var)
        err = source_domain_loss + target_domain_loss + loss 
        domain_errs.update(target_domain_loss.data.item())
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i_batch % opt.showFreq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
              'Domain loss {domain_errs.val:.3f} ({domain_errs.avg:.3f})\t'
              'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})'.format(
              epoch, i_batch, len(train_loader), batch_time=batch_time, 
              data_time=data_time, loss=losses, domain_errs=domain_errs, flow2_EPE=flow2_EPEs))
 
	# debug  	
	#if i_batch >= 3:
	#    break

    return losses.avg, flow2_EPEs.avg



def validate(val_loader, model, criterion, high_res_EPE):
    global args

    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, sample_batched in enumerate(val_loader):
        input = torch.cat((sample_batched['img_left'], sample_batched['img_right']), 1)
        target = sample_batched['gt_disp']
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

	# compute output
        # output = model(input_var)
        # loss = criterion(output[0], target_var)
        # flow2_EPE = high_res_EPE(output[0], target_var) * opt.flowDiv

        output_net1, output_net2 = model(input_var)
        loss_net1 = criterion(output_net1, target_var)
        loss_net2 = criterion(output_net2, target_var)
        loss = loss_net1 + loss_net2
        flow2_EPE = high_res_EPE(output_net2, target_var) * opt.flowDiv
        #output_net1, output_net2 = model(input_var)
        #loss_net1 = criterion(output_net1, target_var)
        #loss_net2 = criterion(output_net2, target_var)
        #loss = loss_net1 + loss_net2
        #flow2_EPE = high_res_EPE(output_net2, target_var) * opt.flowDiv


        # record loss and EPE
        losses.update(loss.data.item(), target.size(0))
        flow2_EPEs.update(flow2_EPE.data.item(), target.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i < len(output_writers):  # log first output of first batches
        #     if epoch == 0:
        #         output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0].cpu().numpy(), max_value=10), 0)
        #         output_writers[i].add_image('Inputs', input[0][0].numpy().transpose(1, 2, 0) + np.array([0.411,0.432,0.45]), 0)
        #         output_writers[i].add_image('Inputs', input[1][0].numpy().transpose(1, 2, 0) + np.array([0.411,0.432,0.45]), 1)
        #     output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output.data[0].cpu().numpy(), max_value=10), epoch)

        if i % opt.showFreq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                  .format(i, len(val_loader), batch_time.val, flow2_EPEs.val))

	# debug
	#if i >= 3:
	#    break

    print(' * EPE {:.3f}'.format(flow2_EPEs.avg))

    return flow2_EPEs.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, os.path.join(opt.outf,filename))
    if is_best:
        shutil.copyfile(os.path.join(opt.outf,filename), os.path.join(opt.outf,'model_best.pth'))

best_EPE = -1

if opt.model != '' and not opt.domain_transfer:
    if opt.vallist.split("/")[-1].split("_")[0] != 'KITTI':
        EPE = validate(test_loader, net, criterion, high_res_EPE)
        if best_EPE < 0:
            best_EPE = EPE



for epoch in range(start_epoch, end_epoch):
    cur_lr = adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    if opt.domain_transfer:
        train_loss, train_EPE = train_with_domain_transfer(train_loader, td_loader, net, optimizer, epoch)
    else:
        train_loss, train_EPE = train(train_loader, net, optimizer, epoch)
    # evaluate on validation set
    if opt.vallist.split("/")[-1].split("_")[0] != 'KITTI':
        if not opt.domain_transfer:
            EPE = validate(test_loader, net, criterion, high_res_EPE)
        else:
            EPE = train_EPE
    else:
        EPE = train_EPE

    if best_EPE < 0:
        best_EPE = EPE

    # remember best prec@1 and save checkpoint
    is_best = EPE < best_EPE
    best_EPE = min(EPE, best_EPE)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': 'dispnet',
        'state_dict': net.module.state_dict(),
        'best_EPE': best_EPE,    
    }, is_best, 'dispS_epoch_%d.pth' % epoch)

    with open(os.path.join('logs', opt.logFile), 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([epoch, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), train_loss, train_EPE, EPE, cur_lr])

    # torch.save(net.module.state_dict(), '%s/dispC_epoch_%d.pth' % (opt.outf, epoch))


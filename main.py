from __future__ import print_function
import argparse
import os, shutil, sys
import numpy as np
import time, datetime
import random
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

from dataset import *
from dispnet import *
from multiscaleloss import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import csv


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
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

# data transformation
# scale = RandomRescale((1024, 1024))
# crop = RandomCrop((384, 768))
# tt = ToTensor()
# composed = transforms.Compose([scale, crop, tt])


input_transform = transforms.Compose([
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        # transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
        ])

target_transform = transforms.Compose([
        transforms.Normalize(mean=[0],std=[opt.flowDiv])
        ])

train_dataset = DispDataset(txt_file = 'FlyingThings3D_release_TRAIN.list', root_dir = 'data', transform=[input_transform, target_transform])
test_dataset = DispDataset(txt_file = 'FlyingThings3D_release_TEST.list', root_dir = 'data', transform=[input_transform, target_transform])

# for i in range(3):
#     sample = train_dataset[i]
#     print(i, sample['img_left'].size(), type(sample['img_left']), \
#              sample['img_right'].size(), type(sample['img_right']), \
#              sample['gt_disp'].size(), type(sample['gt_disp'])  \
#          )


train_loader = DataLoader(train_dataset, batch_size = opt.batchSize, \
                        shuffle = True, num_workers = 8, \
                        pin_memory = True)

test_loader = DataLoader(test_dataset, batch_size = 16, \
                        shuffle = False, num_workers = 8, \
                        pin_memory = True)

# use multiple-GPUs training
devices = [int(item) for item in opt.devices.split(',')]
ngpu = len(devices)
net = DispNet(ngpu, False)
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

loss_weights = (0.005, 0.02, 0.02, 0.02, 0.02, 0.32, 0.32)
criterion = multiscaleloss(7, 1, loss_weights, loss='L1', sparse=False)
high_res_EPE = multiscaleloss(scales=1, downscale=1, weights=(1), loss='L1', sparse=False)

print('=> setting {} solver'.format('adam'))
init_lr = 1e-5
param_groups = [{'params': net.module.bias_parameters(), 'weight_decay': 0},
                    {'params': net.module.weight_parameters(), 'weight_decay': 4e-4}]

# optimizer = torch.optim.SGD(param_groups, init_lr,
#                                     momentum=0.9)
optimizer = torch.optim.Adam(param_groups, init_lr,
                                    betas=(0.9, 0.999))


with open(os.path.join('logs', opt.logFile), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(['train_loss', 'train_EPE', 'EPE'])

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
    if epoch != 0 and epoch % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i_batch, sample_batched in enumerate(train_loader):
        
        # print(i_batch, sample_batched['img_left'].size(), \
        #                sample_batched['img_right'].size(), \
        #                sample_batched['gt_disp'].size())
        input = torch.cat((sample_batched['img_left'], sample_batched['img_right']), 1)
        target = sample_batched['gt_disp']

        data_time.update(time.time() - end)
        target = target.cuda()
        input = input.cuda()
        # print(i_batch, input.size(), target.size())
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        
        # debug
        # print(output[0])
        # print(target_var)
        
        # np_out = output[0].data.cpu().numpy()
        # # print(np_out.shape)
        # test_value = np_out[[0], [0], [0], [0]]
        # # print(test_value[0])
        # if np.isnan(test_value[0]):
        #     sys.exit(-1)
        # else:
        #     pfm_arr = np_out[0].transpose(1, 2, 0)
        #     save_pfm('test.pfm', pfm_arr)
        #     target_out = target_var.data.cpu().numpy()
        #     pfm_arr = target_out[0].transpose(1, 2, 0)
        #     save_pfm('gt.pfm', pfm_arr)

	# compute loss
        loss = criterion(output, target_var)
        flow2_EPE = high_res_EPE(output[0], target_var) * opt.flowDiv
        # record loss and EPE
        losses.update(loss.data[0], target.size(0))
        flow2_EPEs.update(flow2_EPE.data[0], target.size(0))

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
	# if i_batch >= 1:
	#     break

    return losses.avg, flow2_EPEs.avg
    # return losses.avg


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
        output = model(input_var)
        loss = criterion(output, target_var)
        ## flow2_EPE = realEPE(output[0], target_var)
        flow2_EPE = high_res_EPE(output, target_var)
        # record loss and EPE
        losses.update(loss.data[0], target.size(0))
        flow2_EPEs.update(flow2_EPE.data[0], target.size(0))
        
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
	# if i >= 1:
	#     break

    print(' * EPE {:.3f}'.format(flow2_EPEs.avg))

    return flow2_EPEs.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, os.path.join(opt.outf,filename))
    if is_best:
        shutil.copyfile(os.path.join(opt.outf,filename), os.path.join(opt.outf,'model_best.pth'))

best_EPE = -1

for epoch in range(start_epoch, end_epoch):
    adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train_loss, train_EPE = train(train_loader, net, optimizer, epoch)
    # evaluate on validation set
    EPE = validate(test_loader, net, criterion, high_res_EPE)
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
        writer.writerow([train_loss, train_EPE, EPE])

    # torch.save(net.module.state_dict(), '%s/dispC_epoch_%d.pth' % (opt.outf, epoch))


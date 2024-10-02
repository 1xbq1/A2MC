import argparse
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np

import scd.builder
#from torch.utils.tensorboard import SummaryWriter
from dataset import get_pretraining_set



parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[100, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--checkpoint-path', default='./checkpoints/pretrain/', type=str)
parser.add_argument('--skeleton-representation', type=str,
                    help='input skeleton-representation  for self supervised training (joint or motion or bone)')
parser.add_argument('--pre-dataset', default='ntu60', type=str,
                    help='which dataset to use for self supervised training (ntu60 or ntu120)')
parser.add_argument('--protocol', default='cross_subject', type=str,
                    help='training protocol cross_view/cross_subject/cross_setup')

# specific configs:
parser.add_argument('--encoder-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--encoder-k', default=16384, type=int,
                    help='queue size; number of negative keys (default: 16384)')
parser.add_argument('--encoder-m', default=0.999, type=float,
                    help='momentum of updating key encoder (default: 0.999)')
parser.add_argument('--encoder-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
                    
parser.add_argument('--gpu', default=0)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # pretraining dataset and protocol
    from options import options_pretraining as options 
    if args.pre_dataset == 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.pre_dataset == 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()
    elif args.pre_dataset == 'pku_part1' and args.protocol == 'cross_subject':
        opts = options.opts_pku_part1_cross_subject()
    elif args.pre_dataset == 'pku_part2' and args.protocol == 'cross_subject':
        opts = options.opts_pku_part2_cross_subject()

    opts.train_feeder_args['input_representation'] = args.skeleton_representation

    # create model
    print("=> creating model")

    model = scd.builder.SCD_Net(opts.encoder_args, args.encoder_dim, args.encoder_k, args.encoder_m, args.encoder_t)
    print("options",opts.train_feeder_args)
    print(model)

    # single gpu training
    model = model.cuda()
    # multiple gpu training
    #model = nn.DataParallel(model)
    #model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # Memory_Bank = scd.builder.Adversary_Negatives(args.encoder_k, args.encoder_dim).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Memory_Bank.load_state_dict(checkpoint['memory'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    train_dataset = get_pretraining_set(opts)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    #writer = SummaryWriter(args.checkpoint_path)

    #init memory bank
    #model.eval()
    if not args.resume:
        init_memory(train_loader, model, criterion,
                    optimizer, 0, args)
        print("Init memory bank finished!!")

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss, acc1 = train(train_loader, model, criterion, optimizer, epoch, args)
        #writer.add_scalar('train_loss', loss.avg, global_step=epoch)
        #writer.add_scalar('acc', acc1.avg, global_step=epoch)

        if epoch % 50 == 0:
                  save_checkpoint({
                      'epoch': epoch + 1,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                  }, is_best=False, filename=args.checkpoint_path+'/checkpoint.pth.tar')

def init_memory(train_loader, model, criterion, optimizer,epoch, args):
    
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    
    for i, (q_input, k_input) in enumerate(train_loader):

        q_input = q_input.float().cuda(non_blocking=True)
        # qa_input = qa_input.float().cuda(non_blocking=True)
        k_input = k_input.float().cuda(non_blocking=True)
        
        #attack
        # pred = model(None, qa_input, None)
        # predlabel = torch.argmax(pred, axis=1)
        # qa_input = SMART_attack(qa_input, predlabel, model)

        # compute output
        output1, output2, output3, output4, target1, target2, target3, target4 = model(q_input, k_input, init_me=True, epoch=i)
        
        batch_size = output2.size(0)
        # interactive level loss
        loss = criterion(output1, target1) + criterion(output2, target2) + criterion(output3, target3) \
                + criterion(output4, target4)
        losses.update(loss.item(), batch_size)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i)
        if (i+1) * batch_size >= model.K:
            break

    for param_q, param_k in zip(model.encoder_q.parameters(),
                                model.encoder_k.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        #param_k.requires_grad = False

def train(train_loader, model, criterion, optimizer, epoch, args):
    #dummy_tensor = torch.randn(50000, 50000, device='cuda')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    #losses1 = AverageMeter('Loss1', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1,],
        prefix="Epoch: [{}] Lr_rate [{}]".format(epoch, optimizer.param_groups[0]['lr']))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (qa_input, k_input) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #q_input = q_input.float().cuda(non_blocking=True)
        qa_input = qa_input.float().cuda(non_blocking=True)
        k_input = k_input.float().cuda(non_blocking=True)
        
        #attack
        pred = model(qa_input, None)
        predlabel = torch.argmax(pred, axis=1)
        '''import math
        cosine_increase = 0.5 * (1 - math.cos(math.pi * epoch / 450.0))
        att_lr = 1e-5 + (1e-4 - 1e-5) * cosine_increase'''
        qa_input = SMART_attack(qa_input, predlabel, model,lr=0.0001)

        #qa_input = SMART_attack(qa_input, predlabel, model)

        # compute output
        #output1, output2, output3, output4, outputa1, outputa2, outputa3, outputa4, target1, target2, target3, target4, targeta1, targeta2, targeta3, targeta4 = model(q_input, qa_input, k_input)
        output1, output2, output3, output4, target1, target2, target3, target4, loss_mix = model(qa_input, k_input)

        batch_size = output2.size(0)

        # interactive level loss
        loss = criterion(output1, target1) + criterion(output2, target2) + criterion(output3, target3) \
                + criterion(output4, target4) + loss_mix
        '''lossa1 = -torch.mean(torch.sum(torch.log(outputa1) * targeta1, dim=1))
        lossa2 = -torch.mean(torch.sum(torch.log(outputa2) * targeta2, dim=1))
        lossa3 = -torch.mean(torch.sum(torch.log(outputa3) * targeta3, dim=1))
        lossa4 = -torch.mean(torch.sum(torch.log(outputa4) * targeta4, dim=1))
        loss = loss1234 + (lossa1 + lossa2 + lossa3 + lossa4)/4.0'''

        losses.update(loss.item(), batch_size)
        #losses1.update(loss1234.item(), batch_size)
 
        # measure accuracy of model m1 and m2 individually
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, _ = accuracy(output2, target2, topk=(1, 5))
        top1.update(acc1[0], batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
       
    return losses, top1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        with open('log.txt', 'a') as file:
            file.writelines('\t'.join(entries))
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class MyAdam:
    def __init__(self, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False, initial_decay=0):

        # Arguments
        # lr: float >= 0. Learning rate.
        # beta_1: float, 0 < beta < 1. Generally close to 1.
        # beta_2: float, 0 < beta < 1. Generally close to 1.
        # epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        # decay: float >= 0. Learning rate decay over each update.
        # amsgrad: boolean. Whether to apply the AMSGrad variant of this
        #    algorithm from the paper "On the Convergence of Adam and
        #    Beyond".

        iteration = 0.0
        iteration = np.array(iteration)
        lr = np.array(lr)
        beta_1 = np.array(beta_1)
        beta_2 = np.array(beta_2)
        epsilon = np.array(epsilon)
        decay = np.array(decay)
        initial_decay = np.array(initial_decay)
        self.iteration = torch.from_numpy(iteration).type(torch.FloatTensor)
        self.learningRate = torch.from_numpy(lr).type(torch.FloatTensor)
        self.beta_1 = torch.from_numpy(beta_1).type(torch.FloatTensor)
        self.beta_2 = torch.from_numpy(beta_2).type(torch.FloatTensor)
        self.epsilon = torch.from_numpy(epsilon).type(torch.FloatTensor)
        self.decay = torch.from_numpy(decay).type(torch.FloatTensor)
        self.amsgrad = amsgrad
        self.initial_decay = torch.from_numpy(initial_decay).type(torch.FloatTensor)

    def get_updates(self, grads, params):

        N, C, T, V, M = params.size()
        params = params.permute(0, 4, 2, 3, 1).contiguous()
        params = params.view(N * M, T, V, C)
        
        grads = grads.permute(0, 4, 2, 3, 1).contiguous()
        grads = grads.view(N * M , T, V, C)

        rets = torch.zeros(params.shape).cuda()

        lr = self.learningRate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * self.iteration))

        t = self.iteration + 1
        lr_t = lr * (torch.sqrt(1. - torch.pow(self.beta_2, t)) /
                     (1. - torch.pow(self.beta_1, t)))
        lr_t = lr_t.cuda()

        epsilon = self.epsilon.cuda()

        ms = torch.zeros(params.shape).cuda()
        vs = torch.zeros(params.shape).cuda()

        if self.amsgrad:
            vhats = torch.zeros(params.shape).cuda()
        else:
            vhats = torch.zeros(params.shape).cuda()

        for i in range(0, rets.shape[0]):
            p = params[i]
            g = grads[i]
            m = ms[i]
            v = vs[i]
            vhat = vhats[i]

            # print('m',m.shape)
            # print('g',g.shape)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * torch.mul(g, g)
            if self.amsgrad:
                vhat_t = torch.max(vhat, v_t)
                p_t = p - lr_t * m_t / (torch.sqrt(vhat_t) + epsilon)
                vhat = vhat_t

            else:
                p_t = p - lr_t * m_t / (torch.sqrt(v_t) + epsilon)

            rets[i] = p_t
        
        rets = rets.view(N, M, T, V, C)
        rets = rets.permute(0, 4, 2, 3, 1).contiguous()

        return rets

    def zero_grad(self, model):
        """Sets gradients of all model parameters to zero."""
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

class PercepertionLoss(nn.Module):
    def __init__(self, perpLossType, classWeight, reconWeight, boneLenWeight):
        super().__init__()
        neighbor_link = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                        (22, 23), (23, 8), (24, 25), (25, 12)]

        jointWeights = [[[0.02, 0.02, 0.02, 0.02, 0.02,
                          0.02, 0.02, 0.02, 0.02, 0.02,
                          0.04, 0.04, 0.04, 0.04, 0.04,
                          0.02, 0.02, 0.02, 0.02, 0.02,
                          0.02, 0.02, 0.02, 0.02, 0.02]]]

        neighbor_link=torch.tensor(neighbor_link,dtype=int)-1
        jointWeights = torch.tensor(jointWeights, dtype=torch.float32).cuda()
        self.register_buffer('neighbor_link', neighbor_link)
        self.register_buffer('jointWeights', jointWeights)

        self.deltaT = 1 / 30
        self.perpLossType = perpLossType
        self.classWeight = classWeight
        self.reconWeight = reconWeight
        self.boneLenWeight = boneLenWeight

    def forward(self, refData, adData):

        N, C, T, V, M = refData.size()
        refData = refData.permute(0, 4, 2, 3, 1).contiguous()
        refData = refData.view(N * M, T, V, C)

        adData = adData.permute(0, 4, 2, 3, 1).contiguous()
        adData = adData.view(N * M, T, V, C)

        diff = adData - refData
        squaredLoss = torch.sum(torch.mul(diff, diff), dim=-1)
        #weightedSquaredLoss = squaredLoss * self.jointWeights
        squareCost = torch.sum(torch.sum(squaredLoss, axis=-1), axis=-1)
        #squareCost = torch.sum(torch.sum(weightedSquaredLoss, axis=-1), axis=-1)
        oloss = torch.mean(squareCost, axis=-1)

        if self.perpLossType == 'l2':

            return oloss

        elif self.perpLossType == 'acc':

            refAcc = (refData[:, 2:, :, : ] - 2 * refData[:, 1:-1, :, :] + refData[:, :-2, :, :])/self.deltaT/self.deltaT
            adAcc = (adData[:, 2:, :, :] - 2 * adData[:, 1:-1, :, :] + adData[:, :-2, :, :])/self.deltaT/self.deltaT

            diff = adAcc-refAcc
            jointAcc = torch.mean(torch.sum(torch.sum(torch.sum(torch.mul(diff, diff), axis=-1), axis=-1), axis=-1), axis=-1)

            return jointAcc * (1 - self.reconWeight) + oloss * self.reconWeight

        elif self.perpLossType == 'smoothness':

            adAcc = (adData[:, 2:, :, :] - 2 * adData[:, 1:-1, :, :] + adData[:, :-2, :, :]) / self.deltaT / self.deltaT

            jointAcc = torch.mean(torch.sum(torch.sum(torch.sum(torch.mul(adAcc, adAcc), axis=-1), axis=-1), axis=-1), axis=-1)

            return jointAcc * (1 - self.reconWeight) + oloss * self.reconWeight

        elif self.perpLossType == 'jerkness':

            adJerk = (adData[:, 3:, :, :] - 3 * adData[:, 2:-1, :, :] + 3 * adData[:, 1:-2, :, :] + adData[:, :-3, :, :])/self.deltaT/self.deltaT/self.deltaT

            jointJerk = torch.mean(torch.sum(torch.sum(torch.sum(torch.mul(adJerk, adJerk), axis=-1), axis=-1), axis=-1), axis=-1)

            return jointJerk * (1 - self.reconWeight) + oloss * self.reconWeight

        elif self.perpLossType == 'acc-jerk':

            refAcc = (refData[:, 2:, :, :] - 2 * refData[:, 1:-1, :, :] + refData[:, :-2, :, :]) / self.deltaT / self.deltaT

            adAcc = (adData[:, 2:, :, :] - 2 * adData[:, 1:-1, :, :] + adData[:, :-2, :, :]) / self.deltaT / self.deltaT

            diff = adAcc - refAcc
            jointAcc = torch.mean(torch.sum(torch.sum(torch.sum(torch.mul(diff, diff), axis=-1), axis=-1), axis=-1), axis=-1)

            adJerk = (adData[:, 3:, :, :] - 3 * adData[:, 2:-1, :, :] + 3 * adData[:, 1:-2, :, :] + adData[:, :-3, :, :]) / self.deltaT / self.deltaT / self.deltaT

            jointJerk = torch.mean(torch.sum(torch.sum(torch.sum(torch.mul(adJerk, adJerk), axis=-1), axis=-1), axis=-1), axis=-1)

            jerkWeight = 0.7

            return jointJerk * (1 - self.reconWeight) * jerkWeight + jointAcc * (1 - self.reconWeight) * (1 - jerkWeight) + oloss * self.reconWeight
        elif self.perpLossType == 'bone':

            adboneVecs = adData[:, :, self.neighbor_link[:, 0], :] - adData[:, :, self.neighbor_link[:, 1], :]
            adboneLengths = torch.sum(torch.mul(adboneVecs, adboneVecs), axis=-1)

            refboneVecs = refData[:, :, self.neighbor_link[:, 0], :] - refData[:, :, self.neighbor_link[:, 1], :]
            refboneLengths = torch.sum(torch.mul(refboneVecs, refboneVecs), axis=-1)

            diff = refboneLengths - adboneLengths
            boneLengthsLoss = torch.mean(torch.sum(torch.sum(torch.mul(diff, diff), axis=-1), axis=-1), axis=-1)

            return boneLengthsLoss * (1 - self.reconWeight) + oloss * self.reconWeight

        elif self.perpLossType == 'acc-bone':

            refAcc = (refData[:, 2:, :, :] - 2 * refData[:, 1:-1, :, :] + refData[:, :-2, :, :]) / self.deltaT / self.deltaT

            adAcc = (adData[:, 2:, :, :] - 2 * adData[:, 1:-1, :, :] + adData[:, :-2, :, :]) / self.deltaT / self.deltaT

            diff = refAcc - adAcc

            squaredLoss = torch.sum(torch.mul(diff, diff),axis=-1)

            weightedSquaredLoss = squaredLoss * self.jointWeights

            jointAcc = torch.mean(torch.sum(torch.sum(weightedSquaredLoss, axis=-1), axis=-1), axis=-1)

            adboneVecs = adData[:, :, self.neighbor_link[:,0], :] - adData[:, :, self.neighbor_link[:,1], :]
            adboneLengths = torch.sum(torch.mul(adboneVecs, adboneVecs), axis=-1)

            refboneVecs = refData[:, :, self.neighbor_link[:, 0], :] - refData[:, :, self.neighbor_link[:, 1], :]
            refboneLengths = torch.sum(torch.mul(refboneVecs, refboneVecs), axis=-1)

            diff = refboneLengths -adboneLengths
            boneLengthsLoss = torch.mean(torch.sum(torch.sum(torch.mul(diff, diff), axis=-1), axis=-1), axis=-1)

            return boneLengthsLoss * (1 - self.reconWeight) * self.boneLenWeight + jointAcc * (1 - self.reconWeight) * (1 - self.boneLenWeight) + oloss * self.reconWeight

def SMART_attack(x_batch, y_batch, model, classWeight=0.6, reconWeight=0.4, boneLenWeight=0.7, updateClip=0.01, max_iter=1, lr=0.001):
        
    for param in model.parameters():
        param.requires_grad = False
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(params) == 0
        
    Adam = MyAdam(lr)
    PerLoss = PercepertionLoss(perpLossType='acc-bone', classWeight=classWeight, reconWeight=reconWeight, boneLenWeight=boneLenWeight)
        
    x = x_batch.clone().detach().requires_grad_(True).cuda()
    N,C,T,V,M = x.shape
    #print("x",x.requires_grad)
        
    for _ in range(max_iter):
        
        pred = model(x, None)
        predlabel = torch.argmax(pred, axis=1)
        #print("pred",pred.requires_grad)
        #print("predlabel",predlabel.requires_grad)
        classLoss = -torch.nn.CrossEntropyLoss()(pred, y_batch)
        
        x.grad = None
        classLoss.backward(retain_graph=True)
        cgs = x.grad.data
            
        percepLoss = PerLoss(x_batch, x)
        x.grad = None
        percepLoss.backward(retain_graph=True)
        pgs = x.grad.data
            
        # foolRate = foolRateCal(y_batch, predlabel)
            
        cgs = cgs.permute(0, 4, 2, 3, 1).contiguous() #N,C,T,V,M -> N,M,T,V,C
        cgs = cgs.view(N*M, T, V*C)
        cgsnorms = torch.sqrt(torch.sum(torch.square(cgs), axis=-1))
        cgsnorms = cgsnorms + 1e-18
        cgs = cgs / cgsnorms[:, :, np.newaxis]
        cgs = cgs.view(N, M, T, V, C)
        cgs = cgs.permute(0, 4, 2, 3, 1).contiguous() #N,M,T,V,C -> N,C,T,V,M
            
        pgs = pgs.permute(0, 4, 2, 3, 1).contiguous() #N,C,T,V,M -> N,M,T,V,C
        pgs = pgs.view(N*M, T, V*C)
        pgsnorms = torch.sqrt(torch.sum(torch.square(pgs), axis=-1))
        pgsnorms = pgsnorms + 1e-18
        pgs = pgs / pgsnorms[:, :, np.newaxis]
        pgs = pgs.view(N, M, T, V, C)
        pgs = pgs.permute(0, 4, 2, 3, 1).contiguous() #N,M,T,V,C -> N,C,T,V,M
            
        grads = cgs*classWeight + pgs*(1-classWeight)
        x.data = Adam.get_updates(grads, x.data)
    
    for param in model.parameters():
        param.requires_grad = True
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(params) > 0
    
    return x

if __name__ == '__main__':
    main()

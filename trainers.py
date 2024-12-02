########################################################################################
########################################################################################
####################                                                ####################
####################                     CvT                        ####################
####################                                                ####################
########################################################################################
########################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import torch
from tqdm import tqdm

from timm.data import Mixup
# from torch.cuda.amp import autocast

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if isinstance(output, list):
        output = output[-1]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

def train_one_epoch_cvt(config, train_loader, model, criterion, optimizer, epoch, scaler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # tqdm for displaying progress
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, position=0, leave=False, desc='Training')

    # logging.info('=> switch to train mode')
    model.train()

    aug = config.aug
    mixup_fn = Mixup(
        mixup_alpha=aug.mixup, cutmix_alpha=aug.mixcut,
        cutmix_minmax=None, # changed
        prob=aug.mixup_prob, switch_prob=aug.mixup_switch_prob,
        mode=aug.mixup_mode, label_smoothing=config.loss.label_smoothing,
        num_classes=config.num_classes
    ) if aug.mixup_prob > 0.0 else None
    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        # print(f"y shape: {y.shape}, dtype: {y.dtype}")
        # print(f"y values: min={y.min()}, max={y.max()}")
        # print(f"num_classes: {mixup_fn.num_classes}")
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        if mixup_fn:
            x, y = mixup_fn(x, y)


        outputs = model(x)
        # print(f"Model output size: {outputs.size()}")
        loss = criterion(outputs, y)

        # compute gradient and do update step
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') \
            and optimizer.is_second_order

        loss.backward(create_graph=is_second_order)

        if config.train.clip_grad_norm > 0.0:
            # Unscales the gradients of optimizer's assigned params in-place


            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.clip_grad_norm
            )

        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))

        if mixup_fn:
            y = torch.argmax(y, dim=1)
        prec1, prec5 = accuracy(outputs, y, (1, 5))

        top1.update(prec1, x.size(0))
        top5.update(prec5, x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            msg = '=> Epoch[{0}][{1}/{2}]: ' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,
                      speed=x.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            #logging.info(msg)
        
        batch_bar.set_postfix(
            loss="{:.04f}".format(losses.avg), 
            prec1="{:.04f}".format(top1.avg), 
            prec5="{:.04f}".format(top5.avg)
        )
        batch_bar.update()

        torch.cuda.synchronize()

    return top1.avg, top5.avg, losses.avg
        


@torch.no_grad()
def test_cvt(config, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()

    # tqdm for displaying progress
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Validating')

    end = time.time()
    for i, (x, y) in enumerate(val_loader):
        # compute output
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        outputs = model(x)
        loss = criterion(outputs, y)

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))

        prec1, prec5 = accuracy(outputs, y, (1, 5))
        top1.update(prec1, x.size(0))
        top5.update(prec5, x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        batch_bar.set_postfix(
            loss="{:.04f}".format(losses.avg), 
            prec1="{:.04f}".format(top1.avg), 
            prec5="{:.04f}".format(top5.avg)
        )
        batch_bar.update()

    top1_acc, top5_acc, loss_avg = map(
        lambda x: x.avg,
        [top1, top5, losses]
    )

    """
    msg = '=> TEST using Reassessed labels:\t' \
        'Accuracy@1 {top1:.3f}%\t' \
        'Accuracy@5 {top5:.3f}%\t'.format(
            error1=top1_acc,
            error5=top5_acc
        )
    logging.info(msg)

    """
    logging.info('=> switch to train mode')
    model.train()

    return top1_acc, top5_acc, loss_avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


########################################################################################
########################################################################################
####################                                                ####################
####################                     utils                      ####################
####################                                                ####################
########################################################################################
########################################################################################
def get_trainer(model_name='cvt'):
    if model_name in ['cvt', 'dcvt', 'rcvt']:
        return train_one_epoch_cvt
    else:
        raise Exception('only cvt, dcvt, rcvt are supported')
def get_tester(model_name='cvt'):
    if model_name in ['cvt', 'dcvt', 'rcvt']:
        return test_cvt
    else:
        raise Exception('only cvt, dcvt, rcvt are supported')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
########################################################################################
########################################################################################
####################                                                ####################
####################                   SwinV2                       ####################
####################                                                ####################
########################################################################################
########################################################################################
import numpy as np

import torch
import timm.utils
from torch import inf


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def train_one_epoch_swinv2(config, data_loader, model, criterion, optimizer, epoch, lr_scheduler):
    loss_scaler=NativeScalerWithGradNormCount()
    mixup_fn = None
    mixup_active = config.aug.mixup > 0 or config.aug.cutmix > 0.
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.aug.mixup, cutmix_alpha=config.aug.cutmix, cutmix_minmax=None,
            prob=config.aug.mixup_prob, switch_prob=config.aug.mixup_switch_prob, mode=config.aug.mixup_mode,
            label_smoothing=config.model.label_smoothing, num_classes=config.num_classes)
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    start = time.time()
    end = time.time()

    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, position=0, leave=False, desc='Training')
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        original_targets = targets.clone()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        """
        with torch.amp.autocast('cuda', enabled=config.amp_enable):
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.train.accumulation_steps
        """
        outputs = model(samples)
        loss = criterion(outputs, targets)

        # print('samples shape', samples.shape)
        # print('targets shape', targets.shape)
        # print('outputs shape', outputs.shape) 
        
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') \
            and optimizer.is_second_order
        loss.backward(create_graph=is_second_order)
        if config.train.clip_grad > 0.0:
            # Unscales the gradients of optimizer's assigned params in-place


            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.clip_grad
            )

        optimizer.step()

        """
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.train.clip_grad,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.train.accumulation_steps == 0)
        """
        if (idx + 1) % config.train.accumulation_steps == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.train.accumulation_steps)
        
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
        acc1_meter.update(acc1, original_targets.size(0))
        acc5_meter.update(acc5, original_targets.size(0))  

        loss_meter.update(loss.item(), targets.size(0))
        end = time.time()


        if idx % config.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']

        batch_bar.set_postfix(
            loss="{:.04f}".format(loss_meter.avg), 
            acc1="{:.04f}".format(acc1_meter.avg),
            acc5="{:.04f}".format(acc5_meter.avg),
            lr="{:.09f}".format(lr),
            wd="{:.04f}".format(wd),
        )
        batch_bar.update()

    
    # lr_scheduler.step(epoch=epoch + 1)

    return None, None, loss_meter.avg


@torch.no_grad()
def test_swinv2(config, data_loader, model, criterion):

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()

    # tqdm for displaying progress
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, position=0, leave=False, desc='Validating')

    for idx, (images, target) in enumerate(data_loader):
        criterion = torch.nn.CrossEntropyLoss()

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        """
        with torch.amp.autocast('cuda', enabled=config.amp_enable):
            output = model(images)
        """

        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        loss = criterion(output, target)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # acc1 = reduce_tensor(acc1)
        # acc5 = reduce_tensor(acc5)
        # loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1, target.size(0))
        acc5_meter.update(acc5, target.size(0))

        batch_bar.set_postfix(
            loss="{:.04f}".format(loss_meter.avg), 
            prec1="{:.04f}".format(acc1_meter.avg), 
            prec5="{:.04f}".format(acc5_meter.avg)
        )
        batch_bar.update()
        
    top1_acc, top5_acc, loss_avg = map(
        lambda x: x.avg,
        [acc1_meter, acc5_meter, loss_meter]
    )

    logging.info('=> switch to train mode')
    model.train()
        
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

########################################################################################
########################################################################################
####################                                                ####################
####################                     CvT                        ####################
####################                                                ####################
########################################################################################
########################################################################################
import logging
import time
import torch
from tqdm import tqdm


from thop import profile
from timm.data import Mixup
import wandb

# from torch.cuda.amp import autocast

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if isinstance(output, list):
        output = output[-1]

    maxk = max(topk)
    batch_size = target.size(0)

    if target.ndim > 1:  # Check if target is one-hot encoded
        target = target.argmax(dim=1)
    
    #print(target)
    #print(output)
    #predictions = output.argmax(dim=1)
    #print(predictions)
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

    # Compute FLOPs and Params once per epoch
    # dummy_input = torch.randn((1, 3, *config.train.image_size)).cuda(non_blocking=True)
    # flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    # logging.info(f'Epoch {epoch}: FLOPs: {flops / 1e9:.2f} GFLOPs, Params: {params / 1e6:.2f} M')
    
    # # Log FLOPs and Params to WandB
    # wandb.log({"Epoch": epoch, "FLOPs (GFLOPs)": flops / 1e9, "Params (M)": params / 1e6})


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

    # # Compute FLOPs and Params once per validation phase
    # dummy_input = torch.randn((1, 3, *config.test.image_size)).cuda(non_blocking=True)
    # flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    # logging.info(f'Validation FLOPs: {flops / 1e9:.2f} GFLOPs, Params: {params / 1e6:.2f} M')
    
    # # Log FLOPs and Params to WandB
    # wandb.log({"Validation FLOPs (GFLOPs)": flops / 1e9, "Validation Params (M)": params / 1e6})


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
            top1="{:.04f}".format(top1.avg), 
            top5="{:.04f}".format(top5.avg)
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
    elif model_name == 'swinv2':
        return train_one_epoch_swinv2
    else:
        raise Exception('only cvt, dcvt, rcvt, swinv2 are supported')
def get_tester(model_name='cvt'):
    if model_name in ['cvt', 'dcvt', 'rcvt']:
        return test_cvt
    elif model_name == 'swinv2':
        return test_swinv2
    else:
        raise Exception('only cvt, dcvt, rcvt, swinv2 are supported')

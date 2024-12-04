from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
########################################################################################
########################################################################################
####################                                                ####################
####################                   ConNeXtV2                    ####################
####################                                                ####################
########################################################################################
########################################################################################

def build_scheduler_convnextv2(cfg, optimizer, begin_epoch):
    if 'method' not in cfg.train.scheduler:
        raise ValueError('Please set train.scheduler.method!')
    elif cfg.train.scheduler.method == 'timm':
        args = cfg.train.scheduler.args
        scheduler, _ = create_scheduler(args, optimizer)
        scheduler.step(begin_epoch)
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(
            cfg.train.scheduler.method))
    return scheduler

########################################################################################
########################################################################################
####################                                                ####################
####################                     swinv2                     ####################
####################                                                ####################
########################################################################################
########################################################################################
import bisect

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler_swinv2(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.train.epochs * n_iter_per_epoch)
    warmup_steps = int(config.train.warmup_epochs * n_iter_per_epoch)
    decay_steps = int(config.train.scheduler.decay_epochs * n_iter_per_epoch)
    multi_steps = [i * n_iter_per_epoch for i in config.train.scheduler.multisteps]

    lr_scheduler = None
    if config.train.scheduler.name == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(num_steps - warmup_steps) if config.train.scheduler.warmup_prefix else num_steps,
            lr_min=config.train.min_lr,
            warmup_lr_init=config.train.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=config.train.scheduler.warmup_prefix,
        )

    return lr_scheduler



########################################################################################
########################################################################################
####################                                                ####################
####################                     CvT                        ####################
####################                                                ####################
########################################################################################
########################################################################################
import torch
from timm.scheduler import create_scheduler


def build_scheduler_cvt(cfg, optimizer, begin_epoch):
    if 'method' not in cfg.train.scheduler:
        raise ValueError('Please set train.scheduler.method!')
    elif cfg.train.scheduler.method == 'timm':
        args = cfg.train.scheduler.args
        scheduler, _ = create_scheduler(args, optimizer)
        scheduler.step(begin_epoch)
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(
            cfg.train.scheduler.method))
    return scheduler

########################################################################################
########################################################################################
####################                                                ####################
####################                     utils                      ####################
####################                                                ####################
########################################################################################
########################################################################################
def build_scheduler(mode_name, cfg, optimizer, begin_epoch=None, n_iter_per_epoch=None):
    if mode_name == 'cvt':
        return build_scheduler_cvt(cfg=cfg, optimizer=optimizer, begin_epoch=begin_epoch)
    elif mode_name == 'swinv2':
        return build_scheduler_swinv2(config=cfg, optimizer=optimizer, n_iter_per_epoch=n_iter_per_epoch)
    elif mode_name == 'convnextv2':
        return build_scheduler_convnextv2(cfg=cfg, optimizer=optimizer, begin_epoch=begin_epoch)
    else:
        raise Exception('Only cvt, swinv2, convnextv2 supported.')

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
def build_scheduler(mode_name, cfg, optimizer, begin_epoch):
    return build_scheduler_cvt(cfg=cfg, optimizer=optimizer, begin_epoch=begin_epoch)
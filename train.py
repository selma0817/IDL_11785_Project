from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
logging.basicConfig(
        level=logging.INFO,  # Minimum log level to capture
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
logger = logging.getLogger(__name__)
import os
import pprint
import time
from tqdm import tqdm
import yaml
from yacs.config import CfgNode as CN

import torch
import torch.nn.parallel
import torch.optim
from torch.utils.collect_env import get_pretty_env_info
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

from networks import build_model
from optimizers import build_optimizer
from schedulers import build_scheduler
from dataloader import build_dataloader
from criterions import build_criterion



def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')
    parser.add_argument(
        '--mode', help='train or test', default='train'
    )
    parser.add_argument(
        '--model', help='options: swinv2, convnextv2, cvt', default='cvt'
    )
    parser.add_argument(
        '--config', help='options', default='swinv2_tiny_patch_window8_256.yaml'
    )
    parser.add_argument(
        "--iterations", type=int, help="Number of iterations", default=1
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # start config
    root = 'configs'
    if args.model in ['swinv2', 'convnextv2', 'cvt']:
        config_path = os.path.join(root, args.model, args.config)
        with open('config.yaml', 'r') as file:
            cfg = CN(yaml.safe_load(file))
    else:
        raise Exception('model does not exist, try any of \n swinv2, convnextv2, cvt')
    
    # build model 
    logging.info('=> building model')
    if args.mode == 'train':
        model = build_model(cfg)
    else:
        raise Exception('only train mode suppported, check the ipynbs for testing')
    model.to(torch.device('cuda'))

    # initialize optimizer
    # different for each model
    logging.info('=> building optimzer')
    optimizer = build_optimizer(cfg.model.name, model)

    # initialize scheduler
    logging.info('=> building scheduler')
    begin_epoch = cfg.train.begin_epoch
    scheduler = build_scheduler(cfg.model.name, cfg, optimizer, begin_epoch)

    # define loss
    logging.info('=> building criterion')
    criterion = build_criterion(cfg, is_train=True)
    criterion.cuda()
    criterion_eval = build_criterion(cfg, is_train=False)
    criterion_eval.cuda()

    # add data augmentations
    logging.info('=> use timm loader for training')
    train_loader = build_dataloader(cfg.model.name, cfg, True)
    valid_loader = build_dataloader(cfg.model.name, cfg, False)

   
    # begin epoch = cfg.train.begin_epoch already defined above
    # TODO: add resuming checkpoint
    """
    best_perf, begin_epoch = resume_checkpoint(
        model, optimizer, cfg, final_output_dir, True
    )
    """
    checkpoint_dir = os.path.join('checkpoints', cfg.model.name, cfg.train.save_dir)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)

    logging.info('=> start training')
    for epoch in range(begin_epoch, begin_epoch+cfg.train.epoch):
        head = 'Epoch[{}]:'.format(epoch)
        logging.info('=> {} epoch start'.format(head))

        start = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        logging.info('=> {} train start'.format(head))
        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            train_one_epoch(config, train_loader, model, criterion, optimizer,
                            epoch, final_output_dir, tb_log_dir, writer_dict,
                            scaler=scaler)
        logging.info(
            '=> {} train end, duration: {:.2f}s'
            .format(head, time.time()-start)
        )

        # evaluate on validation set
        logging.info('=> {} validate start'.format(head))
        val_start = time.time()

        if epoch >= config.TRAIN.EVAL_BEGIN_EPOCH:
            perf = test(
                config, valid_loader, model, criterion_eval,
                final_output_dir, tb_log_dir, writer_dict,
                args.distributed
            )

            best_model = (perf > best_perf)
            best_perf = perf if best_model else best_perf

        logging.info(
            '=> {} validate end, duration: {:.2f}s'
            .format(head, time.time()-val_start)
        )

        # TODO: modify scheduler
        lr_scheduler.step(epoch=epoch+1)
        if cfg.TRAIN.LR_SCHEDULER.METHOD == 'timm':
            lr = lr_scheduler.get_epoch_values(epoch+1)[0]
        else:
            lr = lr_scheduler.get_last_lr()[0]
        logging.info(f'=> lr: {lr}')

        # TODO: save model
        if best_model:
            save_model(
                model, args.distributed, checkpoint_dir, 'model_best.pth'
            )
        if config.train.save_epoch_models:
            save_model()

        logging.info(
            '=> {} epoch end, duration : {:.2f}s'
            .format(head, time.time()-start)
        )
    logging.info('=> finish training')

if __name__ == '__main__':
    main()
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
import yaml
from yacs.config import CfgNode as CN
import wandb

import torch
import torch.nn.parallel
import torch.optim
from torch.utils.collect_env import get_pretty_env_info
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(device)
from datetime import datetime
# from fvcore.nn import FlopCountAnalysis

from networks import build_model
from optimizers import build_optimizer
from schedulers import build_scheduler
from dataloader import build_dataloader
from criterions import build_criterion
from trainers import get_trainer, get_tester


def get_num_layer_for_convnext_single(var_name, depths):
    """
    Each layer is assigned distinctive layer ids
    """
    if var_name.startswith("downsample_layers"):
        stage_id = int(var_name.split('.')[1])
        layer_id = sum(depths[:stage_id]) + 1
        return layer_id
    
    elif var_name.startswith("stages"):
        stage_id = int(var_name.split('.')[1])
        block_id = int(var_name.split('.')[2])
        layer_id = sum(depths[:stage_id]) + block_id + 1
        return layer_id
    
    else:
        return sum(depths) + 1


def get_num_layer_for_convnext(var_name):
    """
    Divide [3, 3, 27, 3] layers into 12 groups; each group is three 
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    num_max_layer = 12
    if var_name.startswith("downsample_layers"):
        stage_id = int(var_name.split('.')[1])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1 or stage_id == 2:
            layer_id = stage_id + 1
        elif stage_id == 3:
            layer_id = 12
        return layer_id

    elif var_name.startswith("stages"):
        stage_id = int(var_name.split('.')[1])
        block_id = int(var_name.split('.')[2])
        if stage_id == 0 or stage_id == 1:
            layer_id = stage_id + 1
        elif stage_id == 2:
            layer_id = 3 + block_id // 3 
        elif stage_id == 3:
            layer_id = 12
        return layer_id
    else:
        return num_max_layer + 1

class LayerDecayValueAssigner(object):
    def __init__(self, values, depths=[3,3,27,3], layer_decay_type='single'):
        self.values = values
        self.depths = depths
        self.layer_decay_type = layer_decay_type

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        if self.layer_decay_type == 'single':
            return get_num_layer_for_convnext_single(var_name, self.depths)
        else:
            return get_num_layer_for_convnext(var_name)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')
    parser.add_argument(
        '--mode', help='train or test', default='train'
    )
    parser.add_argument(
        '--model', help='options: swinv2, convnextv2, cvt, dcvt, rcvt', default='swinv2'
    )
    parser.add_argument(
        '--config', help='options', default='swinv2_tiny_patch4_window8_256.yaml'
    )
    parser.add_argument(
        '--run_id', help='run id for wandb', default='IDL_SG2_Ray'
    )
    parser.add_argument(
        '--resume', help='path for checkpoint to resume', default=None
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # start config
    root = 'configs'
    if args.model in ['swinv2', 'convnextv2', 'cvt', 'dcvt', 'rcvt']:
        config_path = os.path.join(root, args.model, args.config)
        with open(config_path, 'r') as file:
            cfg = CN(yaml.safe_load(file))
    else:
        raise Exception('model does not exist, try any of \n swinv2, convnextv2, cvt, dcvt, rcvt')
    
    """
    linear_scaled_lr = cfg.train.base_lr * cfg.train.batch_size / 512.0
    linear_scaled_warmup_lr = cfg.train.warmup_lr * cfg.train.batch_size / 512.0
    linear_scaled_min_lr = cfg.train.min_lr * cfg.train.batch_size / 512.0

    cfg.defrost()
    cfg.train.base_lr = linear_scaled_lr
    cfg.train.warmup_lr = linear_scaled_warmup_lr
    cfg.train.min_lr = linear_scaled_min_lr
    cfg.freeze()
    """

    
    # build model 
    logging.info('=> building model')
    if args.mode == 'train':
        model = build_model(cfg.model.architecture)
    else:
        raise Exception('only train mode suppported, check the ipynbs for testing')
    model.to(torch.device(device))

    # if convnextv2 initialize the assigner
    if cfg.model.name == 'convnextv2':
        if cfg.train.layer_decay < 1.0 or cfg.train.layer_decay > 1.0:
            assert cfg.train.layer_decay_type in ['single', 'group']
            if cfg.train.layer_decay_type == 'group': # applies for Base and Large models
                num_layers = 12
            else:
                num_layers = sum(model.depths)
            assigner = LayerDecayValueAssigner(
                list(cfg.train.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)),
                depths=model.depths, layer_decay_type=cfg.train.layer_decay_type)
        else:
            assigner = None
        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

    # initialize optimizer
    # different for each model
    # if convnextv2, pass in assigner parameters
    logging.info('=> building optimzer')
    if cfg.model.name == 'convnextv2':
        optimizer = build_optimizer(cfg.model.name, cfg, model, get_num_layer=assigner.get_layer_id if assigner is not None else None, get_layer_scale=assigner.get_scale if assigner is not None else None)
    else:
        optimizer = build_optimizer(cfg.model.name, cfg, model)

    # define loss
    logging.info('=> building criterion')
    criterion = build_criterion(cfg.model.name, cfg, is_train=True)
    criterion.cuda()
    criterion_eval = build_criterion(cfg.model.name, cfg, is_train=False)
    criterion_eval.cuda()

    # add data augmentations
    logging.info('=> use timm loader for training')
    train_loader = build_dataloader(cfg.model.name, cfg, True)
    valid_loader = build_dataloader(cfg.model.name, cfg, False)

    # initialize scheduler
    logging.info('=> building scheduler')
    begin_epoch = cfg.train.begin_epoch
    scheduler = build_scheduler(cfg.model.name, cfg, optimizer, begin_epoch, len(train_loader))

   
    # begin epoch = cfg.train.begin_epoch already defined above
    # TODO: add resuming checkpoint
    best_perf = 0.0
    best_model = True
    """
    best_perf, begin_epoch = resume_checkpoint(
        model, optimizer, cfg, final_output_dir, True
    )
    """

    # create unique folder with datatime of job for each run so we don't overwrite prev checkpoints
    
    # curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = '/home/ray/proj/IDL_11785_Project/checkpoints/convnextv2/test1'
    # logging.info(f'=> checkpoints dir: {checkpoint_dir}')
    # os.makedirs(checkpoint_dir, exist_ok=True)

    # scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)

     # scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)
    
    # logging flops
    
    # dummy_input = torch.randn((1, 3, 224, 224)).cuda(non_blocking=True)
    # flops = FlopCountAnalysis(model, dummy_input)
    # flops_total = flops.total()
    # flops_module_operator = flops.by_module_and_operator() 
    # logger.info(f"=>flops total '{flops_total}', flops_module_operator {flops_module_operator}")


    ## handle resume in training
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_perf = checkpoint.get('best_perf', 0.0)
            logging.info(f"=> Loaded checkpoint '{args.resume}' (epoch {begin_epoch})")
        else:
            logging.error(f"=> No checkpoint found at '{args.resume}'")
            raise FileNotFoundError(f"Checkpoint '{args.resume}' does not exist.")
        

    # wandb setup
    logging.info('=> login to wandb')
    wandb.login(key='c8a7fb1f22a9fd377ab46b13a6a9a572f152b896')
    # wandb.login()
    # wandb.setup(api_key='57c916d673703185e1b47000c74bd854db77bcf8')
    run = wandb.init(
        name = cfg.model.architecture+"_"+cfg.train.save_dir, ## Wandb creates random run names if you skip this field
        #reinit = True, ### Allows reinitalizing runs when you re-run this cell
        id = "convnextv2_tiny_test1", ### Insert specific run id here if you want to resume a previous run
        resume = True, ### You need this to resume previous runs, but comment out reinit = True when using this
        project = args.run_id, ### Project should be created in your wandb account
        config = cfg ### Wandb Config for your run
    )

    end_epoch = begin_epoch+cfg.train.epochs
    for epoch in range(begin_epoch, end_epoch):
        head = 'Epoch[{}]:'.format(epoch)
        #logging.info('=> {} epoch start'.format(head))

        start = time.time()

        # train for one epoch
        #logging.info('=> {} train start'.format(head))
        trainer = get_trainer(cfg.model.name)
        top1_train, top5_train, loss_train = trainer(cfg, train_loader, model, criterion, optimizer, epoch, scheduler)
        #logging.info(
        #   '=> {} train end, duration: {:.2f}s'
        #   .format(head, time.time()-start)
        #)

        # evaluate on validation set
        # logging.info('=> {} validate start'.format(head))
        val_start = time.time()

        tester = get_tester(cfg.model.name)
        top1_val, top5_val, loss_val = tester(cfg, valid_loader, model, criterion_eval)

        # update best model
        best_model = (top1_val > best_perf)
        best_perf = top1_val if best_model else best_perf

        logging.info(
            '=> {} validate end, duration: {:.2f}s'
            .format(head, time.time()-val_start)
        )
        # scheduler.step(epoch=epoch+1)
        """
        if cfg.train.scheduler.method == 'timm':
            # lr = scheduler.get_epoch_values(epoch+1)[0]
            lr = scheduler._get_lr(epoch+1)[0]
        else:
            lr = scheduler._get_lr(epoch+1)[0]
        logging.info(f'=> lr: {lr}')
        """

        # print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(loss_train, lr))
        print("\tVal Top1 {:.04f}%\tVal Top5 {:.04f}%\t Val Loss {:.04f}".format(top1_val, top5_val, loss_val))

        wandb.log({
            'train_loss': loss_train,
            'valid_top1': top1_val,
            'valid_top5': top5_val,
            'valid_loss': loss_val,
            #'lr'        : lr
        })

        # save model
        if best_model:
            best_path = os.path.join(checkpoint_dir, 'model_best.pth')
            save_model(
                model, optimizer, scheduler, best_path, epoch, best_perf,
            )
            wandb.save(best_path)
            
        if cfg.train.save_epoch_models:
            last_path = os.path.join(checkpoint_dir, 'model_last.pth')
            save_model(model, optimizer, scheduler, last_path, epoch, best_perf,
            )
            wandb.save(last_path)

        logging.info(
            '=> {} epoch end, duration : {:.2f}s'
            .format(head, time.time()-start)
        )
    logging.info('=> finish training')

def save_model(model, optimizer, scheduler, path, epoch, best_perf):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict(),
         'epoch'                    : epoch,
         'best_perf'                : best_perf,
         },
         path
    )

def load_model(path, model, optimizer= None, scheduler= None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # begin_epoch = checkpoint.get('epoch', cfg.train.begin_epoch)
            # best_perf = checkpoint.get('best_perf', 0.0)
    return [model, optimizer, scheduler]

if __name__ == '__main__':
    
    main()
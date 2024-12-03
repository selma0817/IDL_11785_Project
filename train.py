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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from datetime import datetime
from fvcore.nn import FlopCountAnalysis


print(device)

from networks import build_model
from optimizers import build_optimizer
from schedulers import build_scheduler
from dataloader import build_dataloader
from criterions import build_criterion
from trainers import get_trainer, get_tester



def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')
    parser.add_argument(
        '--mode', help='train or test', default='train'
    )
    parser.add_argument(
        '--model', help='options: swinv2, convnextv2, cvt, dcvt, rcvt', default='cvt'
    )
    parser.add_argument(
        '--config', help='options', default='rcvt_13_224.yaml'
    )
    parser.add_argument(
        '--run_id', help='run id for wandb', default='IDLSG2_yiyan'
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
    
    # build model 
    logging.info('=> building model')
    if args.mode == 'train':
        model = build_model(cfg.model.architecture)
    else:
        raise Exception('only train mode suppported, check the ipynbs for testing')
    model.to(torch.device('cuda'))



    # initialize optimizer
    # different for each model
    logging.info('=> building optimzer')
    optimizer = build_optimizer(cfg.model.name, cfg, model)

    # initialize scheduler
    logging.info('=> building scheduler')
    begin_epoch = cfg.train.begin_epoch
    scheduler = build_scheduler(cfg.model.name, cfg, optimizer, begin_epoch)

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
    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join('/ix1/hkarim/yip33/IDL_11785_project','checkpoints', cfg.model.name, cfg.train.save_dir, curr_time)
    logging.info(f'=> checkpoints dir: {checkpoint_dir}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)

#     checkpoint_dir = os.path.join('checkpoints', cfg.model.name, cfg.train.save_dir)
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#     # scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)
    
    # logging flops
    dummy_input = torch.randn((1, 3, 224, 224)).cuda(non_blocking=True)
    flops = FlopCountAnalysis(model, dummy_input)
    flops_total = flops.total()
    flops_module_operator = flops.by_module_and_operator() 
    logger.info(f"=>flops total '{flops_total}', flops_module_operator {flops_module_operator}")


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
    wandb.login(key='57c916d673703185e1b47000c74bd854db77bcf8')
    # wandb.login()
    # wandb.setup(api_key='57c916d673703185e1b47000c74bd854db77bcf8')
    run = wandb.init(
        name = cfg.model.architecture+"_"+cfg.train.save_dir, ## Wandb creates random run names if you skip this field
        #reinit = True, ### Allows reinitalizing runs when you re-run this cell
        id = "00g5anil", ### Insert specific run id here if you want to resume a previous run
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
        top1_train, top5_train, loss_train = trainer(cfg, train_loader, model, criterion, optimizer,
                            epoch, scaler=None)
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
        scheduler.step(epoch=epoch+1)
        if cfg.train.scheduler.method == 'timm':
            # lr = scheduler.get_epoch_values(epoch+1)[0]
            lr = scheduler._get_lr(epoch+1)[0]
        else:
            lr = scheduler.get_lr()[0]
        logging.info(f'=> lr: {lr}')

        print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(loss_train, lr))
        print("\tVal Top1 {:.04f}%\tVal Top5 {:.04f}%\t Val Loss {:.04f}".format(top1_val, top5_val, loss_val))

        wandb.log({
            'train_loss': loss_train,
            'valid_top1': top1_val,
            'valid_top5': top5_val,
            'valid_loss': loss_val,
            'lr'        : lr
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
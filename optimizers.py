from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
########################################################################################
########################################################################################
####################                                                ####################
####################                     swinv2                     ####################
####################                                                ####################
########################################################################################
########################################################################################
from functools import partial
from torch import optim as optim


def build_optimizer_swinv2(config, model, simmim=False, is_pretrain=False):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    if simmim:
        if is_pretrain:
            parameters = get_pretrain_param_groups(model, skip, skip_keywords)
        else:
            depths = config.model.depths
            num_layers = sum(depths)
            get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
            scales = list(config.train.layer_decay ** i for i in reversed(range(num_layers + 2)))
            parameters = get_finetune_param_groups(model, config.train.base_lr, config.train.weight_decay, get_layer_func, scales, skip, skip_keywords)
    else:
        parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.train.optimizer.name.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.train.optimizer.momentum, nesterov=True,
                              lr=config.train.base_lr, weight_decay=config.train.weight_decay)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.train.optimizer.eps, betas=config.train.optimizer.betas,
                                lr=config.train.base_lr, weight_decay=config.train.weight_decay)
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())



########################################################################################
########################################################################################
####################                                                ####################
####################                     CvT                        ####################
####################                                                ####################
########################################################################################
########################################################################################
import torch.nn as nn
import torch.optim as optim

from timm.optim import create_optimizer

def _is_depthwise(m):
    return (
        isinstance(m, nn.Conv2d)
        and m.groups == m.in_channels
        and m.groups == m.out_channels
    )


def set_wd(cfg, model):
    without_decay_list = cfg.train.without_weight_decay_list
    without_decay_depthwise = []
    without_decay_norm = []
    for m in model.modules():
        if _is_depthwise(m) and 'dw' in without_decay_list:
            without_decay_depthwise.append(m.weight)
        elif isinstance(m, nn.BatchNorm2d) and 'bn' in without_decay_list:
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)
        elif isinstance(m, nn.GroupNorm) and 'gn' in without_decay_list:
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)
        elif isinstance(m, nn.LayerNorm) and 'ln' in without_decay_list:
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)

    with_decay = []
    without_decay = []

    skip = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()

    skip_keys = {}
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keys = model.no_weight_decay_keywords()

    for n, p in model.named_parameters():
        ever_set = False

        if p.requires_grad is False:
            continue

        skip_flag = False
        if n in skip:
            print('=> set {} wd to 0'.format(n))
            without_decay.append(p)
            skip_flag = True
        else:
            for i in skip:
                if i in n:
                    print('=> set {} wd to 0'.format(n))
                    without_decay.append(p)
                    skip_flag = True

        if skip_flag:
            continue

        for i in skip_keys:
            if i in n:
                print('=> set {} wd to 0'.format(n))

        if skip_flag:
            continue

        for pp in without_decay_depthwise:
            if p is pp:
                if cfg.verbose:
                    print('=> set depthwise({}) wd to 0'.format(n))
                without_decay.append(p)
                ever_set = True
                break

        for pp in without_decay_norm:
            if p is pp:
                if cfg.verbose:
                    print('=> set norm({}) wd to 0'.format(n))
                without_decay.append(p)
                ever_set = True
                break

        if (
            (not ever_set)
            and 'bias' in without_decay_list
            and n.endswith('.bias')
        ):
            if cfg.verbose:
                print('=> set bias({}) wd to 0'.format(n))
            without_decay.append(p)
        elif not ever_set:
            with_decay.append(p)

    # assert (len(with_decay) + len(without_decay) == len(list(model.parameters())))
    params = [
        {'params': with_decay},
        {'params': without_decay, 'weight_decay': 0.}
    ]
    return params


def build_optimizer_cvt(cfg, model):
    if cfg.train.optimizer == 'timm':
        args = cfg.train.optimizer_args

        print(f'=> usage timm optimizer args: {cfg.train.optimizer_args}')
        optimizer = create_optimizer(args, model)

        return optimizer

    optimizer = None
    params = set_wd(cfg, model)
    if cfg.train.optimizer == 'sgd':
        optimizer = optim.SGD(
            params,
            # filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay,
            nesterov=cfg.train.nesterov
        )
    elif cfg.train.optimizer == 'adam':
        optimizer = optim.Adam(
            params,
            # filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == 'adamW':
        optimizer = optim.AdamW(
            params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            params,
            # filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.weight_decay,
            alpha=cfg.train.rmsprop_alpha,
            centered=cfg.train.rmsprop_centered
        )

    return optimizer

########################################################################################
########################################################################################
####################                                                ####################
####################                     utils                      ####################
####################                                                ####################
########################################################################################
########################################################################################
def build_optimizer(model_name, cfg, model):
    if model_name in ['cvt', 'dcvt', 'rcvt']:
        return build_optimizer_cvt(cfg, model)
    elif model_name == 'swinv2':
        return build_optimizer_swinv2(config=cfg, model=model)
    else:
        raise Exception('only cvt, dcvt, rcvt are supported')
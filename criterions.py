########################################################################################
########################################################################################
####################                                                ####################
####################                   SwinV2                       ####################
####################                                                ####################
########################################################################################
########################################################################################
import torch
import timm.loss
def build_criterion_swinv2(config):
    if config.aug.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.model.label_smoothing> 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.model.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion

########################################################################################
########################################################################################
####################                                                ####################
####################                     CvT                        ####################
####################                                                ####################
########################################################################################
########################################################################################
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def linear_combination(x, y, epsilon):
        return epsilon*x + (1-epsilon)*y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' \
            else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = th.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


def build_criterion_cvt(cfg, is_train=True):
    if cfg.aug.mixup_prob > 0.0 and cfg.loss.loss == 'softmax':
        criterion = SoftTargetCrossEntropy() \
            if is_train else nn.CrossEntropyLoss()
    elif cfg.loss.label_smoothing > 0.0 and cfg.loss.loss == 'softmax':
        criterion = LabelSmoothingCrossEntropy(cfg.loss.label_smoothing)
    elif cfg.loss.loss == 'softmax':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError('Unkown loss {}'.format(cfg.loss.loss))

    return criterion


########################################################################################
########################################################################################
####################                                                ####################
####################                     utils                      ####################
####################                                                ####################
########################################################################################
########################################################################################
def build_criterion(model_name, cfg, is_train=True):
    if model_name in ['cvt', 'dcvt', 'rcvt']:
        return build_criterion_cvt(cfg, is_train)
    elif model_name == 'swinv2':
        return build_criterion_swinv2(cfg)
    else:
        raise Exception('only cvt, dcvt, rcvt, swinv2 are supported')
import torch
import os
from torchvision import datasets, transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def build_imagenet_val_dataset(input_size):
    transform = build_transform(input_size)
    root = 'data/imagenet'

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")
    print("reading from datapath", 'data/imagenet')
    root = os.path.join(root, 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = 1000
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes

def build_imagenet100_dataset(transforms=None, input_size=224, is_train=False, is_test=False):
    if transforms==None:
        transforms = build_transform(input_size)
    root = 'data/imagenet100' # change  data/imagenet100
    #root = "/ix1/hkarim/yip33/kaggle_dataset/image_net100"
    print("Transform = ")
    if isinstance(transforms, tuple):
        for trans in transforms:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transforms.transforms:
            print(t)
    print("---------------------------")
    print("reading from datapath", root)
    if is_test and not is_train: 
        root = os.path.join(root, 'test_data')
    elif is_train:
        root = os.path.join(root, 'train_data')
    else:
        root = os.path.join(root, 'val_data')
    print('building dataset', root)
    dataset = datasets.ImageFolder(root, transform=transforms)

    return dataset

def build_transform(input_size, crop_pct=None):
    resize_im = input_size > 32
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if input_size >= 384:  
            t.append(
            transforms.Resize((input_size, input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {input_size} size input images...")
        else:
            if crop_pct is None:
                crop_pct = 224 / 256
            size = int(input_size / crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
########################################################################################
########################################################################################
####################                                                ####################
####################                   ConNeXtV2                    ####################
####################                                                ####################
########################################################################################
########################################################################################
import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

def build_dataloader_convnextv2(is_train, cfg, is_test=False):
    transform = build_transform_convnextv2(is_train, cfg)
    if is_test and not is_train:
        dataset = build_imagenet100_dataset(transform, cfg.image_size[0], is_train, is_test)
        print('built test dataloader')
    elif is_train:
        dataset = build_imagenet100_dataset(transform, cfg.image_size[0], is_train)
        print('built train dataloader')
    else:
        dataset = build_imagenet100_dataset(transform, cfg.image_size[0], is_train)
        print('built val dataloader')
    if is_train:
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=cfg.train.batch_size,
            num_workers=6,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=cfg.train.batch_size,
            num_workers=6,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )

    mixup_fn = None
    mixup_active = cfg.train.aug.mixup > 0 or cfg.train.aug.cutmix > 0. or cfg.train.aug.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=cfg.train.aug.mixup, cutmix_alpha=cfg.train.aug.cutmix, cutmix_minmax=None,
            prob=cfg.train.aug.mixup_prob, switch_prob=cfg.train.aug.mixup_switch_prob, mode=cfg.train.aug.mixup_mode,
            label_smoothing=cfg.train.aug.smoothing, num_classes=cfg.num_classes)
    
    return dataloader


def build_transform_convnextv2(is_train, cfg):
    resize_im = cfg.image_size[0] > 32
    mean = IMAGENET_DEFAULT_MEAN
    std =  IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=cfg.image_size,
            is_training=True,
            color_jitter=cfg.train.aug.color_jitter,
            auto_augment=cfg.train.aug.aa,
            interpolation=cfg.train.aug.train_interpolation,
            re_prob=cfg.train.aug.reprob,
            re_mode=cfg.train.aug.remode,
            re_count=cfg.train.aug.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                cfg.image_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if cfg.image_size[0] >= 384:  
            t.append(
            transforms.Resize(cfg.image_size, 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {cfg.image_size} size input images...")
        else:
            if cfg.test.crop_pct == 'None':
                crop_pct = 224 / 256
            size = int(cfg.image_size[0] / crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(cfg.image_size[0]))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

########################################################################################
########################################################################################
####################                                                ####################
####################                   SwinV2                       ####################
####################                                                ####################
########################################################################################
########################################################################################
import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_dataloader_swinv2(config, is_train=True):
    if is_train:
        dataset = build_dataset_swinv2(is_train=is_train, config=config)
        print(f"successfully build train dataset")
    else:
        dataset = build_dataset_swinv2(is_train=is_train, config=config)
        print(f"successfully build val dataset")

    if is_train:
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.train.batch_size,
            num_workers=6,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
            drop_last=False
        )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.aug.mixup > 0 or config.aug.cutmix > 0.
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.aug.mixup, cutmix_alpha=config.aug.cutmix, cutmix_minmax=None,
            prob=config.aug.mixup_prob, switch_prob=config.aug.mixup_switch_prob, mode=config.aug.mixup_mode,
            label_smoothing=config.model.label_smoothing, num_classes=config.num_classes)
    
    return data_loader


def build_dataset_swinv2(is_train, config):
    transforms = build_transforms_swinv2(is_train, config)
    dataset = build_imagenet100_dataset(transforms=transforms, input_size=config.train.image_size[0], is_train=is_train)

    return dataset


def build_transforms_swinv2(is_train, config):
    resize_im = config.train.image_size[0] > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.train.image_size,
            is_training=True,
            color_jitter=config.aug.color_jitter if config.aug.color_jitter > 0 else None,
            auto_augment=config.aug.auto_augment if config.aug.auto_augment != 'none' else None,
            re_prob=config.aug.reprob,
            re_mode=config.aug.remode,
            re_count=config.aug.recount,
            interpolation=config.data.interpolation,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.train.image_size, padding=4)
        return transform

    t = []
    if resize_im:
        if config.test.crop:
            size = int((256 / 224) * config.train.image_size[0])
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.data.interpolation)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.train.image_size[0]))
        else:
            t.append(
                transforms.Resize(config.train.image_size,
                                  interpolation=_pil_interp(config.data.interpolation))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

########################################################################################
########################################################################################
####################                                                ####################
####################                     CvT                        ####################
####################                                                ####################
########################################################################################
########################################################################################
from timm.data import create_loader
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import ImageFilter
import random

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
def build_transforms_cvt(cfg, is_train=True):
    normalize = T.Normalize(mean=cfg.input.mean, std=cfg.input.std)
    if is_train:
        aug = cfg.aug
        scale = aug.scale
        ratio = aug.ratio
        ts = [
            T.RandomResizedCrop(
                cfg.train.image_size[0], scale=scale, ratio=ratio,
                interpolation=cfg.aug.interpolation
            ),
            T.RandomHorizontalFlip(),
        ]
        """
        cj = aug.color_jitter
        if cj[-1] > 0.0:
            ts.append(T.RandomApply([T.ColorJitter(*cj[:-1])], p=cj[-1]))

        gs = aug.GRAY_SCALE
        if gs > 0.0:
            ts.append(T.RandomGrayscale(gs))

        gb = aug.GAUSSIAN_BLUR
        if gb > 0.0:
            ts.append(T.RandomApply([GaussianBlur([.1, 2.])], p=gb))
        """

        ts.append(T.ToTensor())
        ts.append(normalize)
    else:
        if cfg.test.center_crop:
            transforms = T.Compose([
                T.Resize(
                    int(cfg.test.image_size[0] / 0.875),
                    interpolation=cfg.test.interpolation
                ),
                T.CenterCrop(cfg.test.image_size[0]),
                T.ToTensor(),
                normalize,
            ])
        else:
            transforms = T.Compose([
                T.Resize(
                    (cfg.test.image_size[1], cfg.test.image_size[0]),
                    interpolation=cfg.test.interpolation
                ),
                T.ToTensor(),
                normalize,
            ])

        transforms = T.Compose(ts)

def build_dataloader_cvt(cfg, is_train=False, is_test=False):
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = True
    else:
        batch_size = cfg.test.batch_size
        shuffle = False
    
    transforms = build_transforms_cvt(cfg)
    dataset = build_imagenet100_dataset(transforms=transforms, input_size=cfg.train.image_size[0], is_train=is_train, is_test=is_test)

    if cfg.aug.timm_aug.use_loader and is_train:
        print('scale is', cfg.aug.scale, type(cfg.aug.scale))
        timm_cfg = cfg.aug.timm_aug
        data_loader = create_loader(
            dataset,
            input_size=[3]+cfg.train.image_size, #TODO, edit later
            batch_size=cfg.train.batch_size,
            is_training=True,
            use_prefetcher=True,
            no_aug=False,
            re_prob=timm_cfg.re_prob,
            re_mode=timm_cfg.re_mode,
            re_count=timm_cfg.re_count,
            re_split=timm_cfg.re_split,
            scale=cfg.aug.scale,
            ratio=cfg.aug.ratio,
            hflip=timm_cfg.hflip,
            vflip=timm_cfg.vflip,
            color_jitter=timm_cfg.color_jitter,
            auto_augment=timm_cfg.auto_augment,
            num_aug_splits=0,
            interpolation=timm_cfg.interpolation,
            mean=cfg.input.mean,
            std=cfg.input.std,
            num_workers=6,
            distributed=False, # not using
            collate_fn=None,
            pin_memory=True,
            use_multi_epochs_loader=True,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=6,
            pin_memory=True, # spped boost
            sampler=None, # changed sampler to None
            drop_last=True if is_train else False,
        )

    return data_loader


########################################################################################
########################################################################################
####################                                                ####################
####################                     utils                      ####################
####################                                                ####################
########################################################################################
########################################################################################
def build_dataloader(model_name, cfg, is_train=True, is_test=False):
    if model_name in ['cvt', 'dcvt', 'rcvt']:
        return build_dataloader_cvt(cfg, is_train, is_test)
    elif model_name == 'swinv2':
        return build_dataloader_swinv2(config=cfg, is_train=is_train)
    elif model_name == 'convnextv2':
        return build_dataloader_convnextv2(is_train=is_train, cfg = cfg)
    else:
        raise Exception('only cvt, dcvt, rcvt, swinv2, convnextv2 are supported')
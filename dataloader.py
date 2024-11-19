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

def build_imagenet100_dataset(transforms=None):
    if transforms==None:
        transforms = build_transform(224)
    root = 'data/imagenet'

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
    print("reading from datapath", 'data/imagenet')
    root = os.path.join(root, 'val')
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

def build_dataloader_cvt(cfg, is_train=True):
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = True
    else:
        batch_size = cfg.test.batch_size
        shuffle = False
    
    transforms = build_dataloader_cvt(cfg)
    dataset = build_imagenet100_dataset(transforms=transforms)

    if cfg.aug.timm_aug.use_loader and is_train:
        timm_cfg = cfg.aug.timm_aug
        data_loader = create_loader(
            dataset,
            input_size=cfg.train.image_size[0],
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
            shuffle=shuffle
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
def build_dataloader(model_name, cfg, is_train=True):
    if model_name=='cvt':
        return build_dataloader_cvt(cfg, is_train)
    else:
        raise Exception('only cvt is supported')
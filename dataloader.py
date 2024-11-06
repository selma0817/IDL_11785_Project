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
from .utils import (
    inception_normalize,
    imagenet_normalize,
    MinMaxResize,
)
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from .randaug import RandAugment
from transformers import DetrFeatureExtractor
from functools import partial


def pixelbert_transform(size=800):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

def pixelbert_transform_randaug(size=800):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs

def imagenet_transform(size=800):
    return transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            imagenet_normalize,
        ]
    )

def imagenet_transform_randaug(size=800):
    trs = transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            imagenet_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs

def vit_transform(size=800):
    return transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )

def vit_transform_randaug(size=800):
    trs = transforms.Compose(
        [
            Resize(size, interpolation=Image.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs

def clip_transform(size):
    return Compose([
        Resize(size, interpolation=Image.BICUBIC),
        CenterCrop(size),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def clip_transform_randaug(size):
    trs = Compose([
        Resize(size, interpolation=Image.BICUBIC),
        CenterCrop(size),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    trs.transforms.insert(0, lambda image: image.convert('RGBA'))
    trs.transforms.insert(0, RandAugment(2, 9))
    trs.transforms.insert(0, lambda image: image.convert('RGB'))
    return trs

def detr_transform(size):
    feature_extractor = DetrFeatureExtractor.from_pretrained('/workspace/src/trans_models/detr')
    # the following re-assignment is to avoid OOM
    feature_extractor.max_size = int(feature_extractor.max_size/feature_extractor.size*size)
    feature_extractor.size = size
    return partial(feature_extractor.__call__, return_tensors='pt') 

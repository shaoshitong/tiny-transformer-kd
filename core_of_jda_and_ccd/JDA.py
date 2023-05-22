import os
import numpy as np
import torch
from torch.utils.data import Dataset
import math
import torch
import random
import torchvision.datasets
from torchvision.transforms import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset


class BaseDatasetWrapper(Dataset):
    def __init__(self, org_dataset):
        self.org_dataset = org_dataset

    def __getitem__(self, index):
        sample, target, features = self.org_dataset.__getitem__(index)
        return sample, target, features

    def __len__(self):
        return len(self.org_dataset)


def rotate_with_fill(img, magnitude):
    rot = img.convert('RGBA').rotate(magnitude)
    return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4), rot).convert(img.mode)


def shearX(img, magnitude, fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0), Image.BICUBIC,
                         fillcolor=fillcolor)


def shearY(img, magnitude, fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0), Image.BICUBIC,
                         fillcolor=fillcolor)


def translateX(img, magnitude, fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                         fillcolor=fillcolor)


def translateY(img, magnitude, fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                         fillcolor=fillcolor)


def rotate(img, magnitude, fillcolor):
    return rotate_with_fill(img, magnitude)


def color(img, magnitude, fillcolor):
    return ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))


def posterize(img, magnitude, fillcolor):
    return ImageOps.posterize(img, magnitude)


def solarize(img, magnitude, fillcolor):
    return ImageOps.solarize(img, magnitude)


def contrast(img, magnitude, fillcolor):
    return ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice([-1, 1]))


def sharpness(img, magnitude, fillcolor):
    return ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.choice([-1, 1]))


def brightness(img, magnitude, fillcolor):
    return ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1]))


def autocontrast(img, magnitude, fillcolor):
    return ImageOps.autocontrast(img)


def equalize(img, magnitude, fillcolor):
    return ImageOps.equalize(img)


def invert(img, magnitude, fillcolor):
    return ImageOps.invert(img)


class SubPolicy:

    def __init__(self, p1, operation1, magnitude_idx1, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
        ranges = {
            'shearX': np.linspace(0, 0.3, 10),
            'shearY': np.linspace(0, 0.3, 10),
            'translateX': np.linspace(0, 150 / 331, 10),
            'translateY': np.linspace(0, 150 / 331, 10),
            'rotate': np.linspace(0, 30, 10),
            'color': np.linspace(0.0, 0.9, 10),
            'posterize': np.round(np.linspace(8, 4, 10), 0).astype(int),
            'solarize': np.linspace(256, 0, 10),
            'contrast': np.linspace(0.0, 0.9, 10),
            'sharpness': np.linspace(0.0, 0.9, 10),
            'brightness': np.linspace(0.0, 0.9, 10),
            'autocontrast': [0] * 10,
            'equalize': [0] * 10,
            'invert': [0] * 10
        }

        func = {
            'shearX': shearX,
            'shearY': shearY,
            'translateX': translateX,
            'translateY': translateY,
            'rotate': rotate,
            'color': color,
            'posterize': posterize,
            'solarize': solarize,
            'contrast': contrast,
            'sharpness': sharpness,
            'brightness': brightness,
            'autocontrast': autocontrast,
            'equalize': equalize,
            'invert': invert
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]

    def __call__(self, img):
        label = 0
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1, self.fillcolor)
            label = 1
        return img, label


class PolicyDatasetC10(BaseDatasetWrapper):
    def __init__(self, org_dataset, p=0.5):
        super(PolicyDatasetC10, self).__init__(org_dataset)
        self.primary_tfl = org_dataset.primary_tfl
        org_dataset.primary_tfl = lambda x:x
        self.secondary_tfl = org_dataset.secondary_tfl
        org_dataset.secondary_tfl = lambda x:x
        self.final_tfl = org_dataset.final_tfl
        org_dataset.final_tfl = lambda x:x
        self.policies = [
            SubPolicy(p, 'invert', 7),
            SubPolicy(p, 'rotate', 2),
            SubPolicy(p, 'shearY', 8),
            SubPolicy(p, 'posterize', 9),
            SubPolicy(p, 'autocontrast', 8),
            SubPolicy(p, 'color', 3),
            SubPolicy(p, 'sharpness', 9),
            SubPolicy(p, 'equalize', 5),
            SubPolicy(p, 'contrast', 7),
            SubPolicy(p, 'translateY', 3),
            SubPolicy(p, 'brightness', 6),
            SubPolicy(p, 'solarize', 2),
            SubPolicy(p, 'translateX', 3),
            SubPolicy(p, 'shearX', 8),
        ]
        self.policies_len = len(self.policies)

    def __getitem__(self, index):
        sample, target, features = super(PolicyDatasetC10, self).__getitem__(index)
        policy_index = torch.zeros(self.policies_len).float()
        sample = self.primary_tfl(sample)
        new_sample = sample
        for i in range(self.policies_len):
            new_sample, label = self.policies[i](new_sample)
            policy_index[i] = label
        new_sample = self.final_tfl(new_sample).detach()
        sample = self.final_tfl(sample).detach()
        if isinstance(target, torch.Tensor) and target.ndim == 2 and target.shape[-1] != 1:
            target = target.argmax(1)
        elif not isinstance(target, torch.Tensor):
            target = torch.LongTensor([target])
        target = target.unsqueeze(0).expand(2, -1)  # 2,1
        sample = torch.stack([
            sample,
            new_sample,
        ])
        return sample, target, features



class PolicyDatasetC100(BaseDatasetWrapper):
    def __init__(self, org_dataset, p=0.3):
        super(PolicyDatasetC100, self).__init__(org_dataset)
        self.primary_tfl = org_dataset.primary_tfl
        org_dataset.primary_tfl = lambda x:x
        self.secondary_tfl = org_dataset.secondary_tfl
        org_dataset.secondary_tfl = lambda x:x
        self.final_tfl = org_dataset.final_tfl
        org_dataset.final_tfl = lambda x:x
        self.org_dataset=org_dataset
        print("the probability of CIFAR-100 is {}".format(p))
        self.policies = [
            SubPolicy(p, 'autocontrast', 2),
            SubPolicy(p, 'contrast', 3),
            SubPolicy(p, 'posterize', 0),
            SubPolicy(p, 'solarize', 4),

            SubPolicy(p, 'translateY', 8),
            SubPolicy(p, 'shearX', 5),
            SubPolicy(p, 'brightness', 3),
            SubPolicy(p, 'shearY', 0),
            SubPolicy(p, 'translateX', 1),

            SubPolicy(p, 'sharpness', 5),
            SubPolicy(p, 'invert', 4),
            SubPolicy(p, 'color', 4),
            SubPolicy(p, 'equalize', 8),
            SubPolicy(p, 'rotate', 3),

        ]
        self.policies_len = len(self.policies)

    def __getitem__(self, index):
        sample, target, features = super(PolicyDatasetC100, self).__getitem__(index)
        policy_index = torch.zeros(self.policies_len).float()
        sample = self.primary_tfl(sample)
        new_sample = sample
        for i in range(self.policies_len):
            new_sample, label = self.policies[i](new_sample)
            policy_index[i] = label
        new_sample = self.final_tfl(new_sample).detach()
        sample = self.final_tfl(sample).detach()
        if isinstance(target, torch.Tensor) and target.ndim == 2 and target.shape[-1] != 1:
            target = target.argmax(1)
        elif not isinstance(target, torch.Tensor):
            target = torch.LongTensor([target])
        target = target.unsqueeze(0).expand(2, -1)  # 2,1
        sample = torch.stack([  # 2,XXX
            sample,
            new_sample,
        ])
        return sample, target, features


class RADatasetC100(BaseDatasetWrapper):
    def __init__(self, org_dataset):
        super(RADatasetC100, self).__init__(org_dataset)
        self.primary_tfl = org_dataset.primary_tfl
        org_dataset.primary_tfl = lambda x:x
        self.secondary_tfl = org_dataset.secondary_tfl
        org_dataset.secondary_tfl = lambda x:x
        self.final_tfl = org_dataset.final_tfl
        org_dataset.final_tfl = lambda x:x
        self.org_dataset=org_dataset
        self.RandAugment = transforms.RandAugment(3,5)

    def __getitem__(self, index):
        sample, target, features = super(RADatasetC100, self).__getitem__(index)
        sample = self.primary_tfl(sample)
        new_sample = sample
        new_sample = self.RandAugment(new_sample)
        new_sample = self.final_tfl(new_sample).detach()
        sample = self.final_tfl(sample).detach()
        if isinstance(target, torch.Tensor) and target.ndim == 2 and target.shape[-1] != 1:
            target = target.argmax(1)
        elif not isinstance(target, torch.Tensor):
            target = torch.LongTensor([target])
        target = target.unsqueeze(0).expand(2, -1)  # 2,1
        sample = torch.stack([  # 2,XXX
            sample,
            new_sample,
        ])
        return sample, target, features


class AADatasetC100(BaseDatasetWrapper):
    def __init__(self, org_dataset):
        super(AADatasetC100, self).__init__(org_dataset)
        self.primary_tfl = org_dataset.primary_tfl
        org_dataset.primary_tfl = lambda x:x
        self.secondary_tfl = org_dataset.secondary_tfl
        org_dataset.secondary_tfl = lambda x:x
        self.final_tfl = org_dataset.final_tfl
        org_dataset.final_tfl = lambda x:x
        self.org_dataset=org_dataset
        self.RandAugment = transforms.RandAugment(3,5)

    def __getitem__(self, index):
        sample, target, feature = super(AADatasetC100, self).__getitem__(index)
        sample = self.primary_tfl(sample)
        new_sample = sample
        new_sample = self.RandAugment(new_sample)
        new_sample = self.final_tfl(new_sample).detach()
        sample = self.final_tfl(sample).detach()
        if isinstance(target, torch.Tensor) and target.ndim == 2 and target.shape[-1] != 1:
            target = target.argmax(1)
        elif not isinstance(target, torch.Tensor):
            target = torch.LongTensor([target])
        target = target.unsqueeze(0).expand(2, -1)  # 2,1
        sample = torch.stack([  # 2,XXX
            sample,
            new_sample,
        ])
        return sample, target, feature


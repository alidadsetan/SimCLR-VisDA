from torchvision import transforms
import random
import math
from PIL.ImageOps import colorize, grayscale

def random_255():
    return math.floor(random.random() * 255)

class RandomColorize(object):
    def __init__(self, p=.1):
        self.p = p
    
    def __call__(self, sample):
        if random.random() < self.p:
            random_color = [random_255() for _ in range(3)]
            return colorize(grayscale(sample),black=random_color,white=(255,255,255))
        return sample

def transform_builder(input_height):
    return ({
        "linear_transform": transforms.Compose([
            transforms.Resize((input_height,input_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        "contrast_train_transforms": transforms.Compose([transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=input_height,scale=(.25,.5)),
            RandomColorize(.8),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.8, 
                                        contrast=0.8, 
                                        saturation=0.5, 
                                        hue=0.2)
            ], p=0.8),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        "contrast_valid_transforms": transforms.Compose([transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=input_height,scale=(.25,.5)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.8, 
                                        contrast=0.8, 
                                        saturation=0.5, 
                                        hue=0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    })
from utils import *
from test import *
from modelCNN import *

import numpy as np
import torch

import argparse

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

mean_transform = {}
mean_transform['CIFAR10'] = [125.3/ 255, 123.0/ 255, 113.9/ 255]

std_transform = {}
std_transform['CIFAR10'] = [63.0/ 255, 62.1/ 255, 66.7/ 255]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='./saved_models/cifar10/model_best.pth')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='MNIST',
                      help='cifar10')

    args = parser.parse_args()
    
    
    if args.dataset == "CIFAR10":
        transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean_transform[args.dataset],
                                                            std_transform[args.dataset])])
        init_channel = 3

    ##### MNIST

    if args.dataset == "MNIST":
        transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5,), (1.0,))])
        init_channel = 1
    #########################
    ##### Load the data #####
    #########################

    ##### CIFAR10

    if args.dataset == "CIFAR10":
        #train_data = datasets.CIFAR10('data', train=True,
                                   # download=True, transform=transform)
        test_data = datasets.CIFAR10('data', train=False,
                                  download=True, transform=transform)

    ##### MNIST

    if args.dataset == "MNIST":
        #train_data = datasets.MNIST(root = "data",train = True,
                             #     download = True, transform = transform)
        test_data = datasets.MNIST(root = "data",train = False,
                                download = True, transform = transform)

    if args.dataset == "CIFAR10":
         classes = ['plane', 'auto', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

    if args.dataset == "MNIST":
         classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)
    
    
    path_load = torch.load(args.load_path)
    model = path_load['weights']
    iteration = path_load['it']
    
    print('Iteration for best model', iteration)
    
    if args.dataset == "CIFAR10":
        model_test = CNN13()
    else:
        model_test = Net()
    model_test = model_test.to(device)
    model_test.load_state_dict(model)
    test(model_test,device,test_loader,classes,args.batch_size)


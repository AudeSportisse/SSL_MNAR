##Loading libraries

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

import seaborn as sns

from torchvision import datasets
from torchvision import transforms

from torchvision import transforms

import torch.optim as optim

import pandas as pd

import logging
import warnings

import random

import pickle

import copy

import argparse

import os

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.metrics import mean_squared_error

import medmnist
from medmnist import INFO, Evaluator

from test import *
from train import *
from utils import *



"""# Main"""
def main():
    parser = argparse.ArgumentParser(description='medMNIST')
    
    parser.add_argument('--seed', type=int, default=10,
                      help='seed for generation of missing values')
    
    parser.add_argument('--threshold', type=float, default=0.95,
                      help='pseudo-label threshold')
    parser.add_argument('--lmbd', type=float, default=1.,
                      help='Strength of pseudo-labelling regularization')
    
    parser.add_argument('--n_epochs', type=int, default=200,
                      help='number of epochs')
    parser.add_argument('--n_epochs_pre_training', type=int, default=0,
                      help='number of epochs pre training using IPW MCAR')
    parser.add_argument('--batch_size_l', type=int, default=64,
                      help='batch size for labelled data')
    parser.add_argument('--batch_size_u', type=int, default=128,
                      help='batch size for unlabelled data')
    parser.add_argument('--batch_size_t', type=int, default=20,
                  help='batch size for test data')
    
    parser.add_argument('--beta_cutoff', type=float, default=0.5,
                      help='beta for adaptative cutoff')
    parser.add_argument('--cutoff_meth', type=str, default="ICLR",
                  help='method for adaptative cutoff')
    
    parser.add_argument('--lab_num', nargs="+", type=int, default=[5],
                      help='number of labeled data in each class')
    parser.add_argument('--unlab_num', nargs="+", type=int, default=[5],
                      help='number of unlabeled data in each class')
    parser.add_argument('--prob_estim', nargs="+", type=float, default=[0.5],
                      help='estimated probability of being observed per class')
    
    parser.add_argument('--info_py', type=str, default='not_known',
                      help='By default, the propensities of the classes are not known')
    
    parser.add_argument('--general_method', type=str, default='Pl',
              help='General optimization method to use (oracle, Pl, DePl, ME_withgrad, ME_withougrad, MLE, CADR)')
    parser.add_argument('--SSL_method', type=str, default='PL',
              help='SSL method to use (Pseudo-laballing, Entropy minimization)')
    
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Learning rate')
    parser.add_argument('--dataset', type=str, default='dermamnist',
                      help='dermamnist')

    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    
    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    
    
    logger.info('Using gpu: %s ' % torch.cuda.is_available())



    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.seed()



    data_flag = args.dataset
    # data_flag = 'breastmnist'
    download = True

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    #### attention this code works only for n_classes= 7 (change the dictionaries dic_new of backups otherwise)

    DataClass = getattr(medmnist, info['python_class'])
    
    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
        
    # load the data
    train_data = DataClass(split='train', transform=data_transform, download=download)
    test_data = DataClass(split='test', transform=data_transform, download=download)

    number_per_class_test = [0]*n_classes
    for u in range(n_classes):
        for v in range(len(test_data.labels)):
            if test_data.labels[v]==u:
                number_per_class_test[u]+=1
    number_per_class_test

    py_test=[number_per_class_test[u]/sum(number_per_class_test) for u in range(len(number_per_class_test))]

    #######################
    ##### Dataloaders #####
    #######################

    valid_num = 9*n_classes

    labeled_idx, valid_idx, unlabeled_idx = sample_labeled_data(train_data.imgs,np.array(train_data.labels),args.lab_num,valid_num,n_classes,args.unlab_num,py=py_test)
    train_idx = labeled_idx + unlabeled_idx

    #######################
    ##### Dataloaders #####
    #######################

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    labeled_sampler = SubsetRandomSampler(labeled_idx)
    unlabeled_sampler = SubsetRandomSampler(unlabeled_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_u,
        sampler=train_sampler)
    labeled_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_l,
        sampler=labeled_sampler)
    unlabeled_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_u,
        sampler=unlabeled_sampler)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_l, 
        sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_t)


    logger.warning("batch_size_l",args.batch_size_l)
    logger.warning("batch_size_u",args.batch_size_u)
    
    compt = [0]*n_classes
    for u in range(n_classes):
        for idx in labeled_idx: 
            if train_data.labels[idx]==u:
                compt[u]+=1

    compt_un = [0]*n_classes
    for u in range(n_classes):
        for idx in unlabeled_idx: 
            if train_data.labels[idx]==u:
                compt_un[u]+=1
    prob_true = []
    for u in range(n_classes):
        prob_true+=[round(compt[u]/(compt_un[u]+compt[u]),4)]

    classes = ['0', '1', '2', '3', '4', '5', '6']
        
    logger.info('Prob_true: %s', prob_true)

    logger.info('Size of the train set: %s', len(train_idx))
    
    
    logger.warning('Size of the labeled set: %s', len(labeled_idx))
    logger.warning('Size of the unlabeled set: %s', len(unlabeled_idx))
    logger.warning('Size of the valid set: %s ', len(valid_idx))
    logger.warning('Size of the test set: %s ', len(test_data))

    perc_lab = len(labeled_idx)/(len(labeled_idx) + len(unlabeled_idx))
    logger.warning('Percentage of labeled data: %s', perc_lab)

    n_glob = len(labeled_idx) + len(unlabeled_idx)
    nl_glob = len(labeled_idx)
    
    py_true = [(compt_un[u]+compt[u])/n_glob for u in range(n_classes)]


    save_dir = './saved_models'
    save_name = args.dataset + f'_{args.general_method}_{nl_glob}_{args.seed}_{args.lmbd}_{args.beta_cutoff}_{args.cutoff_meth}_{args.info_py}'
    
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    
   

    
    model = medNet(n_channels, n_classes)
    model.to(device)
    model_min = get_weights_copy(model,save_name)
    model_min_acc = get_weights_copy(model,save_name)


    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    #optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    valid_loss_min = 100000
    valid_accuracy_max = 0

    
    prob_MCAR = [perc_lab]*n_classes
    
    if args.method_mecha == "oracle":
        prob_res = prob_true
    else:
        prob_res = prob_MCAR
        
    
    if args.info_py == "known":
        py_res = py_true
    else:
        py_res = [1/n_classes]*n_classes
    
    if args.general_method == "MLE":
        prob_res = prob_estim
        
    
    epoch_loss_min=0
    epoch_accuracy_max=0
    
    dic_sav = []
    
    for epoch in range(args.n_epochs_pre_training):
        logger.warning("Epoch: %s", epoch)
        train_loss,train_loss_l,train_loss_u,valid_loss,valid_accuracy,valid_loss_min,valid_accuracy_max,model_min,model_min_acc,py_res,prob_res, save_perc, p_cutoff, epoch_loss_min, epoch_accuracy_max = train(epoch,logger,model,model_min,model_min_acc,device,optimizer,
          "MCAR_IPW",valid_loss_min,valid_accuracy_max,labeled_loader,valid_loader,
          n_glob,nl_glob, save_path, save_name, loader_u=unlabeled_loader,prob=None,method=None,lamb=0,p_in=p_in,n_classes=n_classes,epoch_loss_min=epoch_loss_min,epoch_accuracy_max=epoch_accuracy_max)
        
        mse = mean_squared_error(prob_true,prob_MCAR)
        
        dic_new = {'train_loss': train_loss, 'valid_loss': valid_loss, 'valid_accuracy': valid_accuracy, 'valid_loss_list': valid_loss, 'mse_mechanism': mse, 'mse_py': mse_py, 'acc_0': save_perc[0], 'acc_1': save_perc[1], 'acc_2': save_perc[2], 'acc_3': save_perc[3], 'acc_4': save_perc[4], 'acc_5': save_perc[5], 'acc_6': save_perc[6]}
        
    
    logger.warning("END PRETRAINING IPW MCAR")

    nlab_perclass = compt
    nlab_perclass = torch.tensor(nlab_perclass).to(device)
    
    for epoch in range(args.n_epochs):
        logger.warning("Epoch: %s", epoch+args.n_epochs_pre_training)
       
        
        prob = torch.tensor(prob_res).to(device)
        p_in = torch.tensor(py_res).to(device)
        train_loss,train_loss_l,train_loss_u,valid_loss,valid_accuracy,valid_loss_min,valid_accuracy_max,model_min,model_min_acc,py_res,prob_res, save_perc, p_cutoff, epoch_loss_min, epoch_accuracy_max = train(epoch+args.n_epochs_pre_training,logger,model,model_min,model_min_acc,device,optimizer,args.general_method,valid_loss_min,valid_accuracy_max,labeled_loader,valid_loader,
          n_glob,nl_glob, save_path, save_name, loader_u=unlabeled_loader,prob=prob,method=args.SSL_method,lamb=args.lmbd,p_in=p_in,nlab_perclass=nlab_perclass,cutoff_meth=args.cutoff_meth,beta_cutoff=args.beta_cutoff,info=args.info_py,n_classes=n_classes,epoch_loss_min=epoch_loss_min,epoch_accuracy_max=epoch_accuracy_max)
        
        
        mse = mean_squared_error(prob_true,prob_res)
        logger.warning('Mechanism: %s',prob_res)
        logger.warning('Mean squared error probMNAR: %s', mse)
        
        mse_py = mean_squared_error(py_true,py_res)
        logger.warning('Py: %s',py_res)
        logger.warning('Mean squared error py: %s', mse_py)
    
        dic_new = {'train_loss': train_loss, 'valid_loss': valid_loss, 'valid_accuracy': valid_accuracy, 'valid_loss_list': valid_loss, 'mse_mechanism': mse, 'mse_py': mse_py, 'acc_0': save_perc[0], 'acc_1': save_perc[1], 'acc_2': save_perc[2], 'acc_3': save_perc[3], 'acc_4': save_perc[4], 'acc_5': save_perc[5], 'acc_6': save_perc[6]}
            
        dic_sav.append(dic_new)
    
    dic_sav = pd.DataFrame.from_records(dic_sav)
    
    logger.warning("best accuracy")
    logger.warning('epoch_accuracy_max',epoch_accuracy_max)
    model_test = medNet(n_channels, n_classes)
    model_test = model_test.to(device)
    model_test.load_state_dict(model_min_acc)
    loss_ba ,save_perc_MNAR,acc_tot = test(model_test,device,test_loader,classes,args.batch_size_t,n_classes)


    logger.warning("best loss")
    logger.warning('epoch_loss_min',epoch_loss_min)
    model_test = medNet(n_channels, n_classes)
    model_test = model_test.to(device)
    model_test.load_state_dict(model_min)
    loss ,_,_ = test(model_test,device,test_loader,classes,args.batch_size_t,n_classes)
    
    
    dic_sav.to_csv(os.path.join(save_path, 'save_results.csv'), index = False, header=True)
    

if __name__ == '__main__':
    main()
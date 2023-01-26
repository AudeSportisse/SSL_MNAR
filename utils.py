###### Function to have the same number of samples in each class in the valid set

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import bernoulli
import numpy as np
import os

from sklearn.metrics import mean_squared_error

def sample_labeled_data(data, target, 
                         num_labels,
                         num_val,
                         num_classes,
                         num_unlabels=[0],
                         index=None,
                         py=None):
    '''
    samples for labeled data
    '''
    if len(num_labels) == 1:
        assert num_labels[0] % num_classes == 0
        samples_per_class = int(num_labels[0] / num_classes)
    
    assert num_val % num_classes == 0
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    if not py is None: 
        samples_per_eval_class = [int(num_val * py[u]) for u in range(len(py))]
    else:
        samples_per_eval_class = int(num_val * 1/num_classes)
    
    lb_data = []
    lbs = []
    lb_idx = []
    ulb_data = []
    ulbs = []
    ulb_idx = []
    lb_data_eval = []
    lbs_eval = []
    lb_idx_eval = []
    for c in range(num_classes):
        if len(num_labels) != 1:
            samples_per_class = int(num_labels[c])
        #print(samples_per_class+samples_per_eval_class)
        #print(samples_per_class)
        idx_all = np.where(target == c)[0]
        #print(len(idx_all))
        if not py is None: 
            idx = np.random.choice(idx_all, samples_per_class+samples_per_eval_class[c], False)
        else:
            idx = np.random.choice(idx_all, samples_per_class+samples_per_eval_class, False)
            
        idx_train = idx[:samples_per_class]
        idx_eval = idx[samples_per_class:]
        
        idx_ulb = list(set(idx_all) - set(idx))
        if len(num_unlabels) != 1:
            idx_ulb = idx_ulb[0:num_unlabels[c]]
        
        lb_idx.extend(idx_train)

        lb_data.extend(data[idx_train])
        lbs.extend(target[idx_train])

        lb_idx_eval.extend(idx_eval)

        lb_data_eval.extend(data[idx_eval])
        lbs_eval.extend(target[idx_eval])
         
        ulb_idx.extend(idx_ulb)

        ulb_data.extend(data[idx_ulb])
        ulbs.extend(target[idx_ulb])
        
        
    return lb_idx, lb_idx_eval, ulb_idx

def sample_test_data(data, target,num_classes,py=None):
    '''
    samples for test data
    '''
    
    number_per_class_test = [0]*num_classes
    for u in range(num_classes):
        for v in range(len(target)):
            if target[v]==u:
                number_per_class_test[u]+=1
    
    num_test = min(number_per_class_test)/max(py)

    if not py is None: 
        samples_per_test_class = [int(num_test * py[u]) for u in range(len(py))]
    else:
        samples_per_test_class = int(num_test * 1/num_classes)
    
    test_idx = []
    for c in range(num_classes):
        idx_all = np.where(target == c)[0]
        if not py is None: 
            idx_test = np.random.choice(idx_all, samples_per_test_class[c], False)
        else:
            idx_test = np.random.choice(idx_all, samples_per_test_class, False)
            
        
        test_idx.extend(idx_test)
        
        
    return test_idx



def sigmoid(x):
    dis = torch.distributions.normal.Normal(loc=0, scale=1)
    return dis.cdf(x) #1/(1+torch.exp(-x))




def mask(unlabeled_perc,train_idx,train_data,mecha,prob=None):
    if mecha == "MCAR":
        split_lab = int(np.floor(unlabeled_perc * len(train_idx)))
        labeled_idx, unlabeled_idx = train_idx[split_lab:], train_idx[:split_lab]
    else: 
        r=[bernoulli.rvs(prob[yi]) for yi in [train_data.targets[u] for u in train_idx]]
        unlabeled_idx = [train_idx[i] for i in [idx for idx in np.where([ri == 0 for ri in r])[0]]]
        labeled_idx = [train_idx[i] for i in [idx for idx in np.where([ri == 1 for ri in r])[0]]]
    return(unlabeled_idx,labeled_idx)


def get_weights_copy(model,save_name):
    weights_path = save_name + f'_weights_temp.pt'
    torch.save(model.state_dict(), weights_path)
    return torch.load(weights_path)


def save_model(save_name, save_path, model, epoch):
    save_filename = os.path.join(save_path, save_name)
    torch.save({'weights': model,
                'it': epoch}, save_filename)
    


def pseudolabels_loss(logits_w, weight, targets_true, threshold=[0.95]):
    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    if len(threshold)>1:
        mask = max_probs.ge(threshold[max_idx]).float()
        select = max_probs.ge(threshold[max_idx]).long()
    else: 
        mask = max_probs.ge(threshold[0]).float()
        select = max_probs.ge(threshold[0]).long()        
    selec_list = [select.detach().cpu().numpy()[u]!=0 for u in range(len(select))]
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    if weight==None:
        return (cross_entropy(logits_w, max_idx.detach()) * mask), select, max_idx.long(), sum(selec_list)
    else:
        return (cross_entropy(logits_w, max_idx.detach()) *weight[targets_true] * mask), select, max_idx.long(), sum(selec_list)

### Classic CNN

class Net(torch.nn.Module):
  #Change init_channel for black and white images
    def __init__(self,init_channel=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(init_channel, 6, 5) 
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120) 
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
class medNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(medNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class Loss():
    
    def __init__(self,prob=None,method=None):
        self.prob = prob
        self.method = method

    def loss_oracle(self,pred_l,target_l):
        cross_entropy = nn.CrossEntropyLoss()
        loss = cross_entropy(pred_l, target_l) 
        return(loss)
    
    def loss_supervised(self,pred_l,pred_u,target_l):
        cross_entropy = nn.CrossEntropyLoss(weight=1/self.prob,reduction='none')
        loss = cross_entropy(pred_l, target_l).mean()
        return(loss)
        
    def loss_unsupervised(self,pred_l,pred_u,target_l):
        if self.method == "PL":
            threshold_vec = 0.95*(self.prob/max(self.prob))**0.1
            loss_uns, select_uns, pseudo_lb_uns, nb_selec_uns = pseudolabels_loss(pred_u, weight = None, targets_true = None,threshold=threshold_vec)
            #loss_uns, select_uns, pseudo_lb_uns, nb_selec_uns = pseudolabels_loss(pred_u, weight = None, targets_true = None)
            loss_sup, select_sup, pseudo_lb_sup, nb_selec_sup = pseudolabels_loss(pred_l, weight = (1-torch.tensor(self.prob))/torch.tensor(self.prob),targets_true=target_l,threshold=threshold_vec)
            #if nb_selec_uns != 0:
             #   print ("selection labels uns:", nb_selec_uns)
            #if nb_selec_sup != 0:
             #   print ("selection labels sup:", nb_selec_sup)
        else:
            loss_uns = (-F.softmax(pred_u,1) * torch.log(F.softmax(pred_u,1)+ 1e-5)).sum(1)
            loss_sup = (((1-self.prob)/self.prob)[target_l] * (-F.softmax(pred_l,1) *  torch.log(F.softmax(pred_l,1) + 1e-5)).sum(1))
        return(loss_uns.mean(), loss_sup.mean())
        
    def loss_unsupervised_biased(self,pred_l,pred_u,target_l):
        if self.method == "PL":
            threshold_vec = 0.95*(self.prob/max(self.prob))**0.1
            #loss_uns, select_uns, pseudo_lb_uns, nb_selec = pseudolabels_loss(pred_u, weight = None, targets_true = None,threshold=threshold_vec)
            loss_uns, select_uns, pseudo_lb_uns, nb_selec = pseudolabels_loss(pred_u, weight = None, targets_true = None)
            #if nb_selec != 0:
                #print ("selection labels uns:", nb_selec)
        else:    
            loss_uns = (-F.softmax(pred_u,1) * torch.log(F.softmax(pred_u,1) + 1e-5)).sum(1)
        loss = loss_uns.mean()
        return(loss)
    
    def loss_supervised_MCAR(self,pred_l,pred_u,target_l):
        cross_entropy = nn.CrossEntropyLoss(reduction='none')
        loss = cross_entropy(pred_l, target_l).mean()
        return(loss)
    
    def loss_unsupervised_MCAR(self,pred_l,pred_u,target_l):
        if self.method == "PL":
            loss_uns, select_uns, pseudo_lb_uns, nb_selec_uns = pseudolabels_loss(pred_u, weight = None, targets_true = None)
            loss_sup, select_sup, pseudo_lb_sup, nb_selec_sup = pseudolabels_loss(pred_l, weight = None,targets_true=target_l)
            #if nb_selec_uns != 0:
             #   print ("selection labels uns:", nb_selec_uns)
            #if nb_selec_sup != 0:
             #   print ("selection labels sup:", nb_selec_sup)
        else:
            loss_uns = (-F.softmax(pred_u,1) * torch.log(F.softmax(pred_u,1)+ 1e-5)).sum(1)
            loss_sup = (-F.softmax(pred_l,1) *  torch.log(F.softmax(pred_l,1) + 1e-5)).sum(1)
        return(loss_uns.mean(), loss_sup.mean())
 
    def loss_unsupervised_biased_correc(self,pred_l,pred_u,target_l):
        prob_r0_given_x = 1 - (self.prob*F.softmax(pred_u,1)).sum(1)
        if self.method == "PL":
            loss_uns, select_uns, pseudo_lb_uns, nb_selec = pseudolabels_loss(pred_u, weight = None, targets_true = None)
            #if nb_selec != 0:
                #print ("selection labels uns:", nb_selec)
        else:    
            loss_uns = (-F.softmax(pred_u,1) * torch.log(F.softmax(pred_u,1) + 1e-5)).sum(1)
        loss = loss_uns.mean()
        return(loss)

    def loss_doubly_robust(self,pred_l,pred_u,target_l):
        prob_r0_given_x = 1 - (self.prob*F.softmax(pred_u,1)).sum(1)
        loss_uns = (1/prob_r0_given_x) * (-F.softmax(pred_u,1) * torch.log(F.softmax(pred_u,1) + 1e-5)).sum(1)
        loss = loss_uns.sum() / (len(pred_l)+len(pred_u))
        return(loss)
    

    


def categorical_crossentropy_logdomain_pytorch(log_predictions, targets):
    return -torch.sum(targets * log_predictions, axis=1)


    


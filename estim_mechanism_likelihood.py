import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import bernoulli
import numpy as np
from utils_informative_labels_SSL_new import *
from sklearn.metrics import mean_squared_error
import copy



def estim_mecha2(epoch,logger,nb_epoch,model,psi,model_min,model_min_acc,device,optimizer_data,optimizer_mecha,prob_true,psi_list,train_loss_list,train_loss_mecha_list,valid_loss_list,valid_accuracy_list,valid_loss_min,valid_accuracy_max,loader,valid_loader,loader_u,save_path,save_name,validation=False,meca_save_acc=None,meca_save_loss=None,epoch_acc=None,validation_loss_mean=None):
    
    valid_loss = 0
    
    model.train()
    if epoch % nb_epoch == 0:

        psi.requires_grad = False
        for par in model.parameters():
            par.requires_grad = True
        

        train_loss = 0
        train_mecha = 0
        num = 0
        
        for (im_l, target_l), (im_u, target_u) in zip(loader, loader_u):
            
            optimizer_data.zero_grad()    

            im_l = im_l.type(torch.FloatTensor)
            im_u = im_u.type(torch.FloatTensor)
            
            im_l = im_l.to(device)
            target_l = target_l.to(device)
            im_u = im_u.to(device)
            target_u = target_u.to(device)

            target_l = target_l.squeeze().long()
            target_u = target_u.squeeze().long()

            pred_l = model(im_l)
            pred_u = F.softmax(model(im_u),1)
        
            cross_entropy = torch.nn.CrossEntropyLoss(reduction='sum')
            loss_data = cross_entropy(pred_l,target_l)
            loss_mecha_l = torch.sum(torch.log(sigmoid(psi)[target_l]))
            loss_mecha_u = torch.sum(torch.logsumexp(torch.log(torch.reshape((1-sigmoid(psi)),(n_classes,1)))+torch.log(torch.transpose(pred_u,0,1)),0))
            
            
            loss_data = loss_data - loss_mecha_l
            loss_mecha = - loss_mecha_u

            prior = 0
            loss =  (loss_data + loss_mecha  + prior) / (len(target_l)+len(target_u))
            

            num += 1
            train_loss += loss.item()
            train_mecha += loss_mecha.item()

            loss.backward()
            optimizer_data.step()
            
        train_loss = train_loss / num
        train_mecha = train_mecha / num
        train_loss_list.append(train_loss)
        train_loss_mecha_list.append(train_mecha)

    
    else:

        psi.requires_grad = True
        for par in model.parameters():
            par.requires_grad = False

        train_loss = 0
        train_mecha = 0
        num = 0

        
        for (im_l, target_l), (im_u, target_u) in zip(loader, loader_u):
            
            optimizer_mecha.zero_grad()

            im_l = im_l.type(torch.FloatTensor)
            im_u = im_u.type(torch.FloatTensor)
            
            im_l = im_l.to(device)
            target_l = target_l.to(device)
            im_u = im_u.to(device)
            target_u = target_u.to(device)

            target_l = target_l.squeeze().long()
            target_u = target_u.squeeze().long()

            pred_l = model(im_l).detach()
            pred_u = F.softmax(model(im_u),1).detach()

            cross_entropy = torch.nn.CrossEntropyLoss(reduction='sum')
            loss_data = cross_entropy(pred_l,target_l)
            loss_mecha_l = torch.sum(torch.log(sigmoid(psi)[target_l]))
            loss_mecha_u = torch.sum(torch.logsumexp(torch.log(torch.reshape((1-sigmoid(psi)),(n_classes,1)))+torch.log(torch.transpose(pred_u,0,1)),0))
            
            loss_data = loss_data - loss_mecha_l
            loss_mecha = - loss_mecha_u
            
            prior = 0
            loss = (loss_data + loss_mecha  + prior) / (len(target_l)+len(target_u))
            
          

            num += 1
            train_loss += loss.item()
            train_mecha += loss_mecha.item()
            
            mdmm_return = mdmm_module(loss)
            mdmm_return.value.backward()
            optimizer_mecha.step()

        train_loss = train_loss / num
        train_mecha = train_mecha / num
        train_loss_list.append(train_loss)
        train_loss_mecha_list.append(train_mecha)  
    

    print("Loss data",loss_data)
    print("Loss mecha",loss_mecha)

    if validation == True: 
        
        ####################################    
        #############VALIDATION#############
        ####################################  

        model.eval()

        class_correct = list(0. for i in range(n_classes))
        class_total = list(0. for i in range(n_classes))
        
        num = 0
        
        with torch.no_grad():

            for im, target in valid_loader:

                im = im.type(torch.FloatTensor)

                im = im.to(device)
                target = target.to(device)

                target = target.squeeze().long()

                pred_val = model(im)

                cross_entropy_val = torch.nn.CrossEntropyLoss()
                loss = cross_entropy_val(pred_val,target)

                num += 1
                valid_loss += loss.item()

                _, predic = torch.max(pred_val, 1)    

                correct_tensor = predic.eq(target.data.view_as(predic))
                correct = np.squeeze(correct_tensor.cpu().numpy())

                for i in range(im.size(0)):
                    label = target.data[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
        

        valid_loss = valid_loss / num
        valid_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)

        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)
        
        
        validation_loss = np.mean(validation_loss_mean[epoch-5:epoch-1])

        if validation_loss <= valid_loss_min:
            if logger is not None: 
                logger.warning('Validation loss decreased: %s', valid_loss)
            else: 
                print('Validation loss decreased: ', valid_loss)
            model_min = get_weights_copy(model,save_name)
            valid_loss_min = valid_loss
            save_model('model_best.pth', save_path, model_min, epoch)
            meca_save_loss = sigmoid(psi).detach().cpu().numpy()
            epoch_acc = epoch

        validation_accuracy_max = np.mean(valid_accuracy_list[epoch-5:epoch-1])
        validation_loss_mean.append(valid_accuracy)
        if valid_accuracy >= valid_accuracy_max:
            if logger is not None: 
                logger.warning('Validation accuracy increased: %s', valid_accuracy)
            else: 
                print('Validation accuracy increased: ', valid_accuracy)
            model_min_acc = get_weights_copy(model,save_name)
            valid_accuracy_max = valid_accuracy
            save_model('model_best_accuracy.pth', save_path, model_min_acc,epoch)
            meca_save_acc = sigmoid(psi).detach().cpu().numpy()
            

    if prob_true is not None:
        py_norm = sigmoid(psi).detach().cpu().numpy()
        psi_error = mean_squared_error(prob_true,py_norm)
        print(psi_error)
    else:
        psi_error = 1000
    psi_list.append(psi_error)
    return(psi_error,psi_list,train_loss_list,train_loss_mecha_list,valid_loss_list,valid_accuracy_list,valid_loss_min,valid_accuracy_max,model_min,model_min_acc,meca_save_acc,meca_save_loss,epoch_acc,validation_loss_mean)




def sigmoid(x):
    dis = torch.distributions.normal.Normal(loc=0,scale=1)
    return torch.exp(dis.log_prob(x)) #1/(1+torch.exp(-x))


def sigmoid_inv(x):
    return torch.log(x)-torch.log(1-x)


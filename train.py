from utils import *
import numpy as np
import copy
import os
import pandas as pd

def train(epoch,logger,model,model_min,model_min_acc,device,optimizer,estim,valid_loss_min,valid_accuracy_max,loader,valid_loader,n_glob,nl_glob,save_path,save_name,loader_u=None,prob=None,method=None,lamb=None,p_in=None,pred_for_new=None,cutoff_meth="ICLR",beta_cutoff=0.5,info="not_known",n_classes=10,meca=None,epoch_loss_min=0,epoch_accuracy_max=0):
    
    train_loss = 0
    train_loss_u = 0
    train_loss_l = 0
    valid_loss = 0
    valid_accuracy = 0
    
    
    ###############################  
    #############TRAIN#############
    ###############################
    
    if estim == "oracle":

        model.train()
        
        cl_loss = Loss()
        num = 0
        
        for im, target in loader:
            
            optimizer.zero_grad()

            im = im.to(device)
            target = target.to(device)
            
            target = target.squeeze().long()

            pred = model(im)
            loss = cl_loss.loss_oracle(pred,target)
            loss.backward()
            optimizer.step()

            num += 1
            train_loss += loss.item()
  
        train_loss = train_loss/num
        #train_loss_list.append(train_loss)
    
    elif estim == "MCAR_SSL_unbiased": 
    
        model.train()
        cl_loss = Loss(prob=prob,method=method)
        num = 0

        for (im_l, target_l), (im_u, target_u) in zip(loader, loader_u):

            optimizer.zero_grad()
            
            im_l = im_l.to(device)
            target_l = target_l.to(device)
            im_u = im_u.to(device)
            target_u = target_u.to(device)
            
            target_l = target_l.squeeze().long()
            target_u = target_u.squeeze().long()
            
            pred_l = model(im_l)
            pred_u = model(im_u)

            loss_l = cl_loss.loss_supervised_MCAR(pred_l,pred_u,target_l)
            loss_u_, loss_u_anti = cl_loss.loss_unsupervised_MCAR(pred_l,pred_u,target_l)
            loss_u = (loss_u_ - loss_u_anti)

            loss = loss_l  + lamb * loss_u
            
            loss.backward()
            optimizer.step()

            num += 1
            train_loss += loss.item()
            train_loss_u += loss_u.item()
            train_loss_l += loss_l.item()
        
        train_loss_u = train_loss_u/num
        train_loss_l = train_loss_l/num
        train_loss = train_loss/num
 
        p_cutoff = 0
        prob = prob.cpu().numpy()
        p_in = p_in.cpu().numpy()

    elif estim == "MCAR_SSL_biased":

        model.train()
        cl_loss = Loss(prob=prob,method=method)
        num = 0

        for (im_l, target_l), (im_u, target_u) in zip(loader, loader_u):

            optimizer.zero_grad()
            
            im_l = im_l.to(device)
            target_l = target_l.to(device)
            im_u = im_u.to(device)
            target_u = target_u.to(device)
            
            target_l = target_l.squeeze().long()
            target_u = target_u.squeeze().long()

            pred_l = model(im_l)
            pred_u = model(im_u)

            loss_l = cl_loss.loss_supervised_MCAR(pred_l,pred_u,target_l)
            loss_u = cl_loss.loss_unsupervised_biased(pred_l,pred_u,target_l)

            loss = loss_l + lamb * loss_u
            
            loss.backward()
            optimizer.step()

            num += 1
            train_loss += loss.item()
            train_loss_u += loss_u.item()
            train_loss_l += loss_l.item()
        
        train_loss_u = train_loss_u/num
        train_loss_l = train_loss_l/num
        train_loss = train_loss/num
        
        p_cutoff = 0
        prob = prob.cpu().numpy()
        p_in = p_in.cpu().numpy()
        
        
    elif estim == "SSL_withgrad":
 
        model.train()
        num = 0
        p_in_list = [0]*n_classes
        prob_list = [0]*n_classes
        
        for (im_l, target_l), (im_u, target_u) in zip(loader, loader_u):
            
            optimizer.zero_grad()
            
            im_l = im_l.to(device)
            target_l = target_l.to(device)
            im_u = im_u.to(device)
            target_u = target_u.to(device)
            
            target_l = target_l.squeeze().long()
            target_u = target_u.squeeze().long()

            pred_l = model(im_l)
            pred_u = model(im_u)
            
            prediction_l = torch.tensor(pred_for_new).to(device)
            
            pred_all = torch.cat((pred_l,pred_u),0)
            prediction_all = torch.softmax(pred_all, dim=-1).sum(0)
            
            if info=="not_known":
                
                if epoch >50:
                    
                    p_y = prediction_all/(len(target_u)+len(target_l)) 
                    p_in = p_y

                    prob = (1/p_y)*prediction_l/n_glob
                    for u in range(n_classes):
                        if (prob[u]>0.99):
                            prob[u]=nl_glob/n_glob

                    p_in_list = [p_in_list[u]+p_in.detach().cpu().numpy()[u] for u in range(len(p_in))]
                    prob_list =  [prob_list[u]+prob.detach().cpu().numpy()[u] for u in range(len(prob))]

                else:
                    p_y = p_in
                    prob = (1/p_y)*prediction_l/n_glob
                    for u in range(n_classes):
                        if (prob[u]>0.99):
                            prob[u]=nl_glob/n_glob
                

                    p_in_list = [p_in_list[u]+p_in.detach().cpu().numpy()[u] for u in range(len(p_in))]
                    prob_list =  [prob_list[u]+prob.detach().cpu().numpy()[u] for u in range(len(prob))]
                
            else:
                p_y = p_in 
                prob = (1/p_y)*torch.tensor(pred_for_new).to(device)/n_glob
        
        
        
            if cutoff_meth == "meca":
                p_cutoff = 0.95*(prob.detach()/torch.max(prob.detach()))**beta_cutoff
            else:
                p_cutoff = 0.95*(p_y.detach()/torch.max(p_y.detach()))**beta_cutoff

            
            
            inv_prob = 1/prob[target_l]
            cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
            loss_l = (cross_entropy(pred_l, target_l) * inv_prob).mean()

            weight_anti = (1-prob[target_l])/prob[target_l]
            pseudo_label = torch.softmax(pred_l.detach(), dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(p_cutoff[max_idx]).float()
            select = max_probs.ge(p_cutoff[max_idx]).long()        
            selec_list = [select.detach().cpu().numpy()[u]!=0 for u in range(len(select))]
            cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
            loss_u_anti = (cross_entropy(pred_l, max_idx.detach()) * weight_anti * mask).mean()
            


            loss_uns, select_uns, pseudo_lb_uns, nb_selec_uns = pseudolabels_loss(pred_u, weight = None, targets_true = None,threshold=p_cutoff)

            unsup_loss = loss_uns.mean()

            loss_u = (unsup_loss -   loss_u_anti * nl_glob/(n_glob-nl_glob))
            loss = loss_l * nl_glob/n_glob + lamb * loss_u
            
            
            loss.backward()
            optimizer.step()

            num += 1
            train_loss += loss.item()
            train_loss_u += loss_u.item() 
            train_loss_l += loss_l.item() * nl_glob/n_glob
        
        train_loss_u = train_loss_u/num
        train_loss_l = train_loss_l/num
        train_loss = train_loss/num
        
        prob = [prob_list[u]/num for u in range(len(prob_list))]
        p_in = [p_in_list[u]/num for u in range(len(p_in))]

           
        
            
    elif estim == "SSL_withoutgrad":
 
        model.train()
        num = 0
        p_in_list = [0]*n_classes
        prob_list = [0]*n_classes
        
        for (im_l, target_l), (im_u, target_u) in zip(loader, loader_u):
            
            optimizer.zero_grad()
            
            im_l = im_l.to(device)
            target_l = target_l.to(device)
            im_u = im_u.to(device)
            target_u = target_u.to(device)
            
            target_l = target_l.squeeze().long()
            target_u = target_u.squeeze().long()

            pred_l = model(im_l)
            pred_u = model(im_u)
            
            prediction_l = torch.tensor(pred_for_new).to(device)
            
            pred_all = torch.cat((pred_l,pred_u),0)
            prediction_all = torch.softmax(pred_all, dim=-1).sum(0)
            
            if info=="not_known":
                
                    
                p_y = prediction_all.detach()/(len(target_u)+len(target_l)) 
                update_p = 0.99 * p_in + (1-0.99) * p_y
                p_y = update_p
                p_in = p_y

                prob = (1/p_y)*prediction_l/n_glob
                for u in range(n_classes):
                    if (prob[u]>0.99):
                        prob[u]=nl_glob/n_glob

                #p_in_list = [p_in_list[u]+p_in.detach().cpu().numpy()[u] for u in range(len(p_in))]
                prob_list =  [prob_list[u]+prob.detach().cpu().numpy()[u] for u in range(len(prob))]

                
            else:
                p_y = p_in
                prob = (1/p_y)*torch.tensor(pred_for_new).to(device)/n_glob
        
        
            if cutoff_meth == "meca":
                p_cutoff = 0.95*(prob.detach()/torch.max(prob.detach()))**beta_cutoff
            else:
                p_cutoff = 0.95*(p_y.detach()/torch.max(p_y.detach()))**beta_cutoff

            
            
            inv_prob = 1/prob[target_l]
            cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
            loss_l = (cross_entropy(pred_l, target_l) * inv_prob).mean()

            weight_anti = (1-prob[target_l])/prob[target_l]
            pseudo_label = torch.softmax(pred_l.detach(), dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(p_cutoff[max_idx]).float()
            select = max_probs.ge(p_cutoff[max_idx]).long()        
            selec_list = [select.detach().cpu().numpy()[u]!=0 for u in range(len(select))]
            cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
            loss_u_anti = (cross_entropy(pred_l, max_idx.detach()) * weight_anti * mask).mean()
            


            loss_uns, select_uns, pseudo_lb_uns, nb_selec_uns = pseudolabels_loss(pred_u, weight = None, targets_true = None,threshold=p_cutoff)

            unsup_loss = loss_uns.mean()

            loss_u = (unsup_loss -   loss_u_anti * nl_glob/(n_glob-nl_glob))
            loss = loss_l * nl_glob/n_glob + lamb * loss_u
            
            
            loss.backward()
            optimizer.step()

            num += 1
            train_loss += loss.item()
            train_loss_u += loss_u.item() 
            train_loss_l += loss_l.item() * nl_glob/n_glob
        
        train_loss_u = train_loss_u/num
        train_loss_l = train_loss_l/num
        train_loss = train_loss/num
        
        if info=="not_known":
            prob = [prob_list[u]/num for u in range(len(prob_list))]
        else:
            prob = prob.cpu().numpy()
        p_in = p_in.cpu().numpy()     
    
    elif estim == "SSL_oldmeca":
 
        model.train()
        num = 0
        
        for (im_l, target_l), (im_u, target_u) in zip(loader, loader_u):
            
            optimizer.zero_grad()
            
            im_l = im_l.to(device)
            target_l = target_l.to(device)
            im_u = im_u.to(device)
            target_u = target_u.to(device)
            
            target_l = target_l.squeeze().long()
            target_u = target_u.squeeze().long()

            pred_l = model(im_l)
            pred_u = model(im_u)
            
            prediction_u = torch.softmax(pred_u, dim=-1).sum(0)
            prediction_l_l = torch.softmax(pred_l,dim=-1).sum(0)
            prediction_l = torch.tensor(pred_for_new).to(device)
            
            
            prob = torch.tensor(meca).to(device)
            p_y = (1/prob)*(pred_for_new/n_glob)
            p_in = p_y
        
            if cutoff_meth == "meca":
                p_cutoff = 0.95*(prob.detach()/torch.max(prob.detach()))**beta_cutoff
            else:
                p_cutoff = 0.95*(p_y.detach()/torch.max(p_y.detach()))**beta_cutoff

            
            
            inv_prob = 1/prob[target_l]
            cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
            loss_l = (cross_entropy(pred_l, target_l) * inv_prob).mean()

            weight_anti = (1-prob[target_l])/prob[target_l]
            pseudo_label = torch.softmax(pred_l.detach(), dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(p_cutoff[max_idx]).float()
            select = max_probs.ge(p_cutoff[max_idx]).long()        
            selec_list = [select.detach().cpu().numpy()[u]!=0 for u in range(len(select))]
            cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
            loss_u_anti = (cross_entropy(pred_l, max_idx.detach()) * weight_anti * mask).mean()

            loss_uns, select_uns, pseudo_lb_uns, nb_selec_uns = pseudolabels_loss(pred_u, weight = None, targets_true = None,threshold=p_cutoff)

            unsup_loss = loss_uns.mean()

            loss_u = (unsup_loss -   loss_u_anti * nl_glob/(n_glob-nl_glob))
            loss = loss_l * nl_glob/n_glob + lamb * loss_u
            
            
            loss.backward()
            optimizer.step()

            num += 1
            train_loss += loss.item()
            train_loss_u += loss_u.item() 
            train_loss_l += loss_l.item() * nl_glob/n_glob
        
        train_loss_u = train_loss_u/num
        train_loss_l = train_loss_l/num
        train_loss = train_loss/num
        
        
        prob = prob.cpu().numpy()
        p_in = p_in.cpu().numpy()       
        
    
    
    elif estim == "SSL_ICLR":
 
        model.train()
        cl_loss = Loss(prob=prob,method=method)
        num = 0
        prob_list = [0]*n_classes
        
        for (im_l, target_l), (im_u, target_u) in zip(loader, loader_u):

            optimizer.zero_grad()
            
            im_l = im_l.to(device)
            target_l = target_l.to(device)
            im_u = im_u.to(device)
            target_u = target_u.to(device)

            target_l = target_l.squeeze().long()
            target_u = target_u.squeeze().long()
            
            pred_l = model(im_l)
            pred_u = model(im_u)
            
            pred_all_detach = torch.cat((pred_l.detach(),pred_u.detach()),0)
            pred_all = torch.cat((pred_l,pred_u),0)
            
            if info=="not_known":
                probability_batch = torch.softmax(pred_all_detach, dim=-1)
                p_each_cls = probability_batch.mean(0)
                update_p = 0.99 * p_in + (1-0.99) * p_each_cls
                p_in = update_p
                prob = (1/p_in)*torch.tensor(pred_for_new).to(device)/n_glob
                
                prob_list =  [prob_list[u]+prob.detach().cpu().numpy()[u] for u in range(len(prob))]
            else:
                probability_batch = torch.softmax(pred_all_detach, dim=-1)
                p_each_cls = p_in
                update_p = p_in

            
            if cutoff_meth == "meca":
                p_cutoff = 0.95*(prob/max(prob))**beta_cutoff
            else:
                p_cutoff = 0.95*(update_p/max(update_p))**beta_cutoff
            
            
            #### comment if py is known
            probability_batch_inv = torch.pinverse(probability_batch)
            bs_weight = torch.mm(p_in.view(1,n_classes),probability_batch_inv)
            bs_weight = torch.clamp(bs_weight,0.0,5.0)
            p_each_cls = torch.mm(bs_weight,probability_batch)
            p_each_cls = torch.squeeze(p_each_cls)

            target_one_hot_l = F.one_hot(target_l,n_classes) 
            py_l = torch.sum(torch.mul(target_one_hot_l, p_each_cls.view(1,n_classes)),1) 
            iclr_weight_l = target_l.shape[0] * (1/py_l)/torch.sum(1/py_l)
            iclr_weight_l =  torch.unsqueeze(iclr_weight_l,1).expand(-1,n_classes)
            log_P = F.log_softmax(pred_l, dim=-1) + torch.log(iclr_weight_l)
            loss_l = categorical_crossentropy_logdomain_pytorch(log_P,target_one_hot_l).mean()


            #Unsupervised losses
            pseudo_labels_u = torch.softmax(pred_l.detach(), dim=-1)
            max_probs, max_idx = torch.max(pseudo_labels_u, dim=-1)
            target_one_hot_u = F.one_hot(max_idx,n_classes)

            py_u = torch.sum(torch.mul(target_one_hot_u, p_each_cls.view(1,n_classes)),1) 
            iclr_weight_u = target_l.shape[0] * (1/py_u)/torch.sum(1/py_u)
            iclr_weight_u =  torch.unsqueeze(iclr_weight_u,1).expand(-1,n_classes)
            log_P_aug =  torch.log(iclr_weight_u)
            ce_loss_aug = categorical_crossentropy_logdomain_pytorch(log_P_aug, target_one_hot_u)

            if len(p_cutoff)>1:
                mask = max_probs.ge(p_cutoff[max_idx]).float()
            else: 
                mask = max_probs.ge(p_cutoff[0]).float()
            anti_unsup_loss = -(ce_loss_aug*mask).mean()
            
            
            loss_uns, select_uns, pseudo_lb_uns, nb_selec_uns = pseudolabels_loss(pred_all, weight = None, targets_true = None,threshold=p_cutoff)
                    
            unsup_loss = loss_uns.mean()
            
            loss_u = (unsup_loss -  anti_unsup_loss)
            loss = loss_l + lamb * loss_u
            
            loss.backward()
            optimizer.step()

            num += 1
            train_loss += loss.item()
            train_loss_u += loss_u.item() 
            train_loss_l += loss_l.item()
        
        train_loss_u = train_loss_u/num
        train_loss_l = train_loss_l/num
        train_loss = train_loss/num
        
        if info=="not_known":
            prob = [prob_list[u]/num for u in range(len(prob_list))]
        else:
            prob = prob.cpu().numpy()
        p_in = p_in.cpu().numpy()       
    
    ####################################    
    #############VALIDATION#############
    ####################################  
    
    model.eval()
    
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    cl_loss = Loss()
    num = 0
    
    with torch.no_grad():
    
        for im, target in valid_loader:

            im = im.to(device)
            target = target.to(device)
            
            target = target.squeeze().long()

            pred_val = model(im)

            loss = cl_loss.loss_oracle(pred_val, target)

            num += 1
            valid_loss += loss.item()

            _, predic = torch.max(pred_val, 1)    
            correct_tensor = predic.eq(target.data.view_as(predic))
            correct = np.squeeze(correct_tensor.cpu().numpy())

            for i in range(im.size(0)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        
        save_perc = np.zeros(n_classes)
        for i in range(n_classes):
            if class_total[i] > 0:
                save_perc[i] = 100 * class_correct[i] / class_total[i]
        
        valid_loss = valid_loss/num
        valid_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)

        
        if valid_loss <= valid_loss_min:
            if logger is not None: 
                logger.warning('Validation loss decreased: %s', valid_loss)
            else: 
                print('Validation loss decreased: ', valid_loss)
            model_min = get_weights_copy(model,save_name)
            valid_loss_min = valid_loss
            epoch_loss_min = epoch
            save_model('model_best.pth', save_path,model_min,epoch)

        if valid_accuracy >= valid_accuracy_max:
            if logger is not None: 
                logger.warning('Validation accuracy increased: %s', valid_accuracy)
            else: 
                print('Validation accuracy increased: ', valid_accuracy)
            model_min_acc = get_weights_copy(model,save_name)
            valid_accuracy_max = valid_accuracy
            epoch_accuracy_max = epoch
            save_model('model_best_accuracy.pth', save_path, model_min_acc,epoch)
            
        
    
    return(train_loss,train_loss_l,train_loss_u,valid_loss,valid_accuracy,valid_loss_min,valid_accuracy_max,model_min,model_min_acc,p_in, prob, save_perc, p_cutoff, epoch_loss_min, epoch_accuracy_max)
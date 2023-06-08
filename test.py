from utils import *
import numpy as np
from sklearn.metrics import roc_auc_score

def test(model,device,test_loader,classes,batch_size,n_classes=10):

    cl_loss = Loss()

    test_loss = 0.0
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    
    model.eval()
    with torch.no_grad():

        logits_list = []
        labels_list = []
        
        for im, target in test_loader:

            im = im.to(device)
            target = target.to(device)
            
            target = target.squeeze().long()
            
            pred_test = model(im)

            loss = cl_loss.loss_oracle(pred_test, target)

            test_loss += loss.item()*im.size(0)

            _, predic = torch.max(pred_test, 1)    


            correct_tensor = predic.eq(target.view_as(predic))
            correct = np.squeeze(correct_tensor.cpu().numpy())

            for i in range(len(target)):
                label = target[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

            logits_list.append(pred_test)
            labels_list.append(target)

        logits = torch.cat(logits_list).cpu()
        labels = torch.cat(labels_list).cpu()
        test_loss = test_loss/len(test_loader.dataset)
        print('Test Loss:', test_loss)

        save_perc = np.zeros(n_classes)
        for i in range(n_classes):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
                save_perc[i] = 100 * class_correct[i] / class_total[i]
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
        
        auc = 0
        for i in range(logits.shape[1]):
            label_auc = roc_auc_score((labels == i).float(), logits.softmax(dim=-1)[:, i])
            auc += label_auc
        auc = auc / logits.shape[1]
        
        print('AUC', auc)
        
        acc_tot = 100. * np.sum(class_correct) / np.sum(class_total)
        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
        
        
    return(test_loss,save_perc,acc_tot)
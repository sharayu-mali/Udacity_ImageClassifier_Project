
import torch
from torch import nn,optim
import numpy as np
import copy
from time import time

def train_model(model,trainloader,testloader, criterion, optimizer, device, num_epochs=25):
    '''
        Train the model on training datatset and evaluate on validation dataset

        Parameters:
            model
            trainloader
            testloader
            criterion
            optimizer
            num_epochs

        Returns:
            Trained model
    '''
    since = time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_losses, test_losses,test_accuracies = [], [],[]
    for epoch in range(num_epochs):
        start=time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        tot_train_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            images,labels=images.to(device),labels.to(device)
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            tot_train_loss += loss.item()
            
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            
            loss.backward()
            optimizer.step()
        else:
            tot_test_loss = 0
            test_correct = 0 
            with torch.no_grad():
                for images, labels in testloader:                    
                    images,labels=images.to(device),labels.to(device)
                    log_ps = model(images)
                    loss = criterion(log_ps, labels)
                    tot_test_loss += loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    test_correct += equals.sum().item()


        train_loss = tot_train_loss / len(trainloader.dataset)
        test_loss = tot_test_loss / len(testloader.dataset)
        test_acc=test_correct / len(testloader.dataset)
        
        # At completion of epoch
        train_losses.append(train_loss)
        test_losses.append(test_loss)        
        test_accuracies.append(test_acc)
        
        # deep copy the model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
                
        print("Training Loss: {:.3f}.. ".format(train_loss),
                  "Validation Loss: {:.3f}.. ".format(test_loss),
                  "Validation Accuracy: {:.3f}".format(test_acc),
                  "Time per batch: {:.3f} s\n".format((time() - start)))
        print("--------------------------------------------------------\n")
            
    time_elapsed = time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model,testloader,device,criterion= nn.CrossEntropyLoss()):
    '''
        Validate trained model on the test set
        
        Parameters:
            model
            testloader
            device
            criterion

        Returns:
            validation_loss,validation_accuracy - Obtained on Testing dataset
    '''
    test_loss = 0.0
    test_correct = 0.0
    for images,labels in testloader:
        images,labels=images.to(device),labels.to(device)
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        test_loss += loss.item()
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_correct += equals.sum().item()
    
    # calculate avg test loss
    n=len(testloader.dataset)
    val_loss = test_loss/n
    val_acc= (test_correct/n)*100
    
    print('Validation Loss: {:.6f}\n'.format(val_loss))
    print('Validation Accuracy on Testing Dataset: %2.2f%% (%d/%d)' % (val_acc,test_correct, n))
    
    return val_loss,val_acc
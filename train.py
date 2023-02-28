# Imports here

import matplotlib.pyplot as plt

import torch
from torch import nn,optim
from torchvision.models import resnet50,alexnet,vgg11
from PIL import Image
import numpy as np

from get_input_args import get_input_args_train
from load_data import get_data_loaders
from model_training import train_model,test_model



def __main__():
    in_args = get_input_args_train()
    print('Model architecture : ',in_args.arch)
    image_sizes={'vgg11':224,'resnet50':224,'alexnet':256}
    model=None
    hidden_units=in_args.hidden_units
    
    device = torch.device("cuda" if (torch.cuda.is_available() and in_args.gpu)  else "cpu")

     # Initialize the data loaders
    trainloader,testloader,valloader,n_classes = get_data_loaders(in_args.filename,image_sizes[in_args.arch]) 
    print('Loading Data ... ')
    print('\tNumber of images in Training Dataset - ',len(trainloader))
    print('\tNumber of images in Testing Dataset - ',len(testloader))
    print('\tNumber of images in Validation Dataset - ',len(valloader))
    print('Data loaded successfully ...')
    
    print('Loading Pretrained Model and Initializing the classifier ... ')
    # Load pretrained weights and initialize the model
    if in_args.arch=='vgg11':
        image_size=224
        model=vgg11(pretrained=True)
        # Freeze model parameters while training model
        for param in model.parameters():
            param.requires_grad = False
        #print(model)
        input_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(nn.Linear(input_features,hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(hidden_units, n_classes),
                                        nn.LogSoftmax(dim=1))
    elif in_args.arch=='alexnet':
        image_size=256
        model=alexnet(pretrained=True)
        # Freeze model parameters while training model
        for param in model.parameters():
            param.requires_grad = False
            
        
        input_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Linear(input_features,hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(hidden_units, n_classes),
                                        nn.LogSoftmax(dim=1))
   
    else:
        image_size=224
        model=resnet50(pretrained=True)
        # Freeze model parameters while training model
        for param in model.parameters():
            param.requires_grad = False
        input_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(input_features,hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(hidden_units, n_classes),
                                        nn.LogSoftmax(dim=1))

    
    
    params=[]

    for param in model.parameters():
        if param.requires_grad:
            params.append(param)
            
   
    
    model= model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(params, lr=in_args.learning_rate)
    print('Training the Model ...')
    model= train_model(model,trainloader,valloader, criterion,optimizer_ft,device,num_epochs=in_args.epochs)
    print('Model training successful ...')
    print('Testing the Model ...')
    test_loss,test_accuracy=test_model(model,testloader,device)
    print('Model testing successful ...')
    #print(model)
    # Save the checkpoint 
    print('Saving the Model ...')
    path = in_args.save_dir+in_args.arch+".pt"

    torch.save({
                'epoch': in_args.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_ft.state_dict(),
                'hidden_units':in_args.hidden_units,
                'n_classes':n_classes,
                'image_size':image_size
                }, path)
    
    print('Model saved successfully ...')
    return

__main__()
import matplotlib.pyplot as plt

import torch
from torch import nn,optim
from torchvision.models import resnet50,alexnet,vgg11,densenet121
from PIL import Image
import numpy as np

from get_input_args import get_input_args_predict
from load_data import get_data_loaders
from model_training import train_model,test_model

from load_data import process_image,imshow,get_label
def get_model(model_name,hidden_units=512,n_classes=102):
    '''
        Get model instance based on required architecture 
        
        Parameters:
        
            model_name - Name of model - resnet50/vgg11/alexnet
            hidden_units - Number of hidden units in classifier
            n_classes - Number of units in the output layer of classifier
            
        Returns: 
                model
    '''
    model=None
    model_name= model_name.split('/')[-1]
    if model_name=='vgg11.pt':
        model=vgg11(pretrained=False)
        #print(model)
        input_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(nn.Linear(input_features,hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(hidden_units, n_classes),
                                        nn.LogSoftmax(dim=1))
    elif model_name=='alexnet.pt':
        model=alexnet(pretrained=False)
        
        input_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Linear(input_features,hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(hidden_units, n_classes),
                                        nn.LogSoftmax(dim=1))
   
    else:
        model=resnet50(pretrained=False)
        input_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(input_features,hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(hidden_units, n_classes),
                                        nn.LogSoftmax(dim=1))
    return model

def load_model(model_name,PATH):
    '''
        Load and initialize model weights from checkpoint and return model
        
        Parameters:
        
            model_name - Name of model - resnet50/vgg11/alexnet
            PATH - Path to the checkpoint
            
        Returns: 
                model
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    checkpoint = torch.load(PATH, map_location= map_location)
    hidden_units=checkpoint['hidden_units']
    n_classes=checkpoint['n_classes']
    model=get_model(PATH,hidden_units,)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    #print(checkpoint.keys())
    return model,checkpoint['image_size']


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image=Image.open(image_path)
    image=process_image(image)
    pil_img=image.to(device)
    pil_img=pil_img.unsqueeze(0)
    
    output = torch.exp(model.forward(pil_img))
    probs=output.topk(topk)[0].squeeze().tolist()
    classes=output.topk(topk)[1].squeeze().tolist()
    
    return probs,classes
    
def print_results(image_path, model,topk,category_name_file):
    
    probs, classes = predict(image_path, model,topk)
    class_names=[get_label(category_name_file,idx) for idx in classes]
    print(probs,classes,class_names,sep='\n')
    image=Image.open(image_path)
    image=process_image(image)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(15,10), nrows=2)
    imshow(image,ax=ax1)
    ax1.set_title(class_names[0])
    ax2.barh(y=class_names,width=probs)
    plt.yticks(rotation=30)
    plt.show()
    
    return

def __main__():
    in_args = get_input_args_predict()
    model,image_size=load_model(in_args.checkpoint,'./models/'+in_args.checkpoint)
    #print_results(in_args.input, model,in_args.top_k,in_args.category_names)
    probs, classes = predict(in_args.input, model,in_args.top_k)
    class_names=[get_label(in_args.category_names,idx) for idx in classes]
    print(probs,classes,class_names,sep='\n')
    return

__main__()
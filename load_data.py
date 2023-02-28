
from torchvision import datasets, transforms
import json
import numpy as np
import torch
def get_data_loaders(data_dir,IMAGE_SIZE=224):
    '''
        Initialize data loaders for loading training, testing and validation dataset
        
        Parameters:
            data_dir - Path of directory containing dataset folders - train,test,valid
            IMAGE_SIZE - Image size required for model as input
            
        Returns:
                trainloader,testloader,valloader,n_classes
    
    '''
    
    # Define Transforms for Training, Testing and Validation Sets                             
    train_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                        transforms.RandomRotation(30),
                                        transforms.CenterCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                    transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                std  = (0.229, 0.224, 0.225))]) 

    test_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                    transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                std  = (0.229, 0.224, 0.225))])

    val_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                        transforms.CenterCrop(IMAGE_SIZE),
                                        transforms.ToTensor(),
                                    transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                std  = (0.229, 0.224, 0.225))])


    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    val_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

    # Define loaders for  each subset  of data
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=64)

    n_classes=len(train_data.classes)

    return trainloader,testloader,valloader,n_classes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transform=transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                        transforms.ToTensor(),transforms.Normalize(mean = (0.485, 0.456, 0.406),
                        std  = (0.229, 0.224, 0.225))])
    tensor_img = transform(image)
    tensor_img
    return tensor_img
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def get_label(category_name_file,idx):
    '''
    Get Category name for predicted class Id

    Parameters:
        category_name_file
        idx
    Returns:
        Category name corresponding to the index
    '''
    with open(category_name_file, 'r') as f:
        cat_to_name = json.load(f)
    classes_mapping={0: '1', 1: '10', 2: '100', 3: '101', 4: '102', 5: '11', 6: '12', 7: '13', 8: '14', 9: '15', 10: '16', 11: '17', 12: '18', 13: '19', 14: '2', 15: '20', 16: '21', 17: '22', 18: '23', 19: '24', 20: '25', 21: '26', 22: '27', 23: '28', 24: '29', 25: '3', 26: '30', 27: '31', 28: '32', 29: '33', 30: '34', 31: '35', 32: '36', 33: '37', 34: '38', 35: '39', 36: '4', 37: '40', 38: '41', 39: '42', 40: '43', 41: '44', 42: '45', 43: '46', 44: '47', 45: '48', 46: '49', 47: '5', 48: '50', 49: '51', 50: '52', 51: '53', 52: '54', 53: '55', 54: '56', 55: '57', 56: '58', 57: '59', 58: '6', 59: '60', 60: '61', 61: '62', 62: '63', 63: '64', 64: '65', 65: '66', 66: '67', 67: '68', 68: '69', 69: '7', 70: '70', 71: '71', 72: '72', 73: '73', 74: '74', 75: '75', 76: '76', 77: '77', 78: '78', 79: '79', 80: '8', 81: '80', 82: '81', 83: '82', 84: '83', 85: '84', 86: '85', 87: '86', 88: '87', 89: '88', 90: '89', 91: '9', 92: '90', 93: '91', 94: '92', 95: '93', 96: '94', 97: '95', 98: '96', 99: '97', 100: '98', 101: '99'}

    return cat_to_name[str(classes_mapping[int(idx)])]
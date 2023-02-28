import argparse

def get_input_args_train():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as first positional argument - Path to the dataset directory
      2. CNN Model Architecture as --arch with default value 'resnet50'
      3. Learning rate for model training as --learning_rate with default value 0.01
      4. Hidden units for classifier as --hidden_units with defalt value 512
      5. Epochs for model training as --epochs with defalt value 20
      6. Directory path for saving for saving trained model as --save_dir with default value as './'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(
                    prog = 'Train',
                    description = 'Train Image Classifier for classifying flower images')
   
    parser.add_argument('filename',type=str,help="path to the folder containing images for training, validation and testing model")  
    parser.add_argument('--save_dir',default="./models/",type=str,help="path to the folder for saving trained model")  
    parser.add_argument('--arch',default="resnet50",type=str,help="CNN Model Architecture to be used for model training - vgg16,resnet50,alexnet")
    parser.add_argument('--learning_rate',default=0.01,type=float,help="Learning rate for model training")
    parser.add_argument('--hidden_units',default=512,type=int,help="Hidden units for classifier for model training")
    parser.add_argument('--epochs',default=20,type=int,help="Epochs for model training")
    parser.add_argument('--gpu', action='store_true',help="True if gpu is to be used for model training")
     
    return parser.parse_args()


def get_input_args_predict():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Input Image File as first positional argument- input - Path to the input image directory
      2. Checkpoint of model as checkpoint
      3. Category mapping for predicted class as --category_names with default value cat_to_name.json
      4. Number of classes predicted with highest probability by classifier as --top_k with defalt value 5
      5. Whether to use GPU for model inference or not as --gpu
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(
                    prog = 'Predict',
                    description = 'Use trained Image Classifier for classifying flower image')
   
    parser.add_argument('input',type=str,help="path to the folder containing images for training, validation and testing model") 
    parser.add_argument('checkpoint',type=str,help="path to the checkpoint of model to be used")   
    parser.add_argument('--category_names',default="cat_to_name.json",type=str,help="path to the file that contains category mapping for predicted class ids")  
    parser.add_argument('--top_k',default=5,type=int,help="Top k classes to which the input image is predicted to belong")
    parser.add_argument('--gpu', action='store_true',help="True if gpu is to be used for model inference")
     
    return parser.parse_args()
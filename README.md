# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

Directory Structure:

- assets: Directory that stores figures used in Jupyter Notebook file.
- models: Directory that stores model checkpoints
-- Contains dummy file model.pt
- cat_to_name.json: JSON File containing category name mappings to class ids
- Image Classifier Project.ipynb: Jupyter Notebook to implement an image classifier with PyTorch
- get_input_args.py:  Python file containing functions used for parsing command line arguments
- load_data.py: Python file containing functions used for loading data
- model_training.py: Python file containing functions used for training model
- predict.py: Driver code for model inference
- train.py: Driver code for training model

## Project Part I: Developing an Image Classifier with Deep Learning

Use Jupyter Notebook to implement an image classifier with PyTorch
## Project Part II: Building the command line application

1. Train
- Train a new network on a data set with train.py

- Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
- Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "vgg13"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use GPU for training: python train.py data_dir --gpu
2. Predict
- Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

- Basic usage: python predict.py /path/to/image checkpoint
- Options:
Return top K most likely classes: python predict.py input checkpoint --top_k 3
- Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
- Use GPU for inference: python predict.py input checkpoint --gpu
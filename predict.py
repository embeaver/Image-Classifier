# load checkpoint and predict
from PIL import Image
from collections import OrderedDict
#from torch.autograd import Variable
import numpy as np

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import time
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import argparse
import json
from collections import OrderedDict

## PARSE THE ARGUMENTS ##
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('/path/to/image', help = 'path to image to predict')
    parser.add_argument('checkpoint', default = 'checkpoints/checkpoint.pth', help = 'saved model checkpoint')
    
    parser.add_argument('--top_k', default = 1, help = 'top 5 predictions')
    parser.add_argument('--category_names', default = 'cat_to_name.json', help = 'name of cat_to_name.json')
    parser.add_argument('--gpu', action = 'store_true', default = True, help = 'use gpu for loading checkpoint and predicting')
    args = parser.parse_args()
    args = vars(args)
    
    return args

## LABEL MAPPING
def label_mapping(category_names):
    with open(str(category_names), 'r') as f:
        cat_to_name = json.load(f)
    
    types_count = int(len(cat_to_name))
    print('number of categories =', types_count)
    return types_count, cat_to_name

## LOADING THE CHECKPOINT ##
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    new_model = models.densenet121(pretrained = True)
    for parameters in new_model.parameters():
        parameters.requires_grad = False
    in_features = checkpoint['in_features']
    hidden_units = int(checkpoint['hidden_units'])
    hidden_sizes = [hidden_units, 128]
    output_size = 102

    classifier = nn.Sequential(OrderedDict([
                        ('inputs', nn.Linear(in_features, hidden_sizes[0])),
                        ('dropout1', nn.Dropout(p = 0.2)),
                        ('hidden_layer1', nn.Linear(hidden_sizes[0], hidden_sizes[1])),# in_features matches the last lime of the pre-trained model classifier
                        ('relu1', nn.ReLU()),
                        ('dropout2', nn.Dropout(p = 0.2)),
                        ('hidden_layer2', nn.Linear(hidden_sizes[1], output_size)), # 102 is the number of outputs (number of categories of flowers)
                        ('output', nn.LogSoftmax(dim = 1))
                        ]))
    new_model.classifier = checkpoint['classifier']
    new_model.class_to_idx = checkpoint['class_to_idx']
    #new_model.load_state_dict(checkpoint['state_dict'])
    new_model.load_state_dict = checkpoint['state_dict']
    new_model.optimizer = checkpoint['optimizer']
    new_model.epochs = checkpoint['epoch']

    return new_model

## PROCESS IMAGE USING PIL, RETURN NUMPY ARRAY ##
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    width, height = pil_image.size
    #print(width, height)

    if width < height:
        new_width = 256
        new_height = int((new_width * height)/width)
    elif height < width:
        new_height = 256
        new_width = int((new_height * width)/height)
    #print(new_width, new_height)
    
    # resize image to new height and new width
    pil_img_resize = pil_image.resize((new_width, new_height))

    # crop center of image to 224x224 pixels
    left, top, right, bottom = ((new_width/2 - (224/2)), (new_height/2 - (224/2)), 
                            (new_width/2 + (224/2)), (new_height/2 + (224/2)))
    #print('left: ', left, 'top:', top, 'right:', right, 'bottom:', bottom)
    pil_img_crop = pil_img_resize.crop((left, top, right, bottom))
    #print(pil_img_crop.size)
    
    # convert color channels of images into floats between 0-1 instead of integers between 0-255
    numpy_img = np.array(pil_img_crop)/255
    #print('numpy_img shape: ', numpy_img.shape)

    # normalize the images to match the network
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (numpy_img - mean)/std

    # pytorch expects color channel to be first dimension but it is the third dimension in the PIL image and Numpy array
    # reorder these using ndarray.transpose
    #numpy_img = image.transpose((1, 0, 2))
    #numpy_img = image.transpose((2, 1, 0))
    #numpy_img = image.transpose((2, 0, 1))
    #numpy_img = image.transpose((1, 2, 0))
    numpy_img = image.transpose((0, 1, 2))
    #print('new_numpy_img shape: ', numpy_img.shape)

    return numpy_img

## PREDICT THE IMAGE USING PRETRAINED MODEL ##
def predict(image_path, model, topk, category_names, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    # for option 2 (numpy array) in process_image
    numpy_img = process_image(image_path)
    
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    image = tensor_transform(numpy_img).float()
    if gpu == 'gpu':
        # device will automatically use CUDA if available/enabled
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')     
    model.eval() # turns off dropouts for evaluation
    image = torch.FloatTensor(image).to(device)
    inputs = image.unsqueeze_(0)
    model.to(device)
    
    log_ps = model.forward(inputs) # forward pass
    types_count, cat_to_name = label_mapping(category_names)

    ps = torch.exp(log_ps) # output of model is LogSoftmax(), exp to get probs
    probs, classes = ps.topk(k = int(topk), dim = 1) # top probabilities, top classes

    probs = probs.squeeze().tolist()
    #print('\nProbabilities:', probs)
    classes = classes.squeeze().tolist()
    #print('\nClasses:', classes)  # , type(classes))
    
    # translate class to flower using class_to_idx
    index_to_class = {value: key for key, value in model.class_to_idx.items()}
    #print('index_to_class successful')
    if classes == 1:
        names = cat_to_name[index_to_class[classes]]
    elif len(classes) > 1:
        names = [cat_to_name[index_to_class[item]] for item in classes]
    #print('\nNames:', names)

    return probs, classes, names

## MAIN PROGRAM ##
def main():
    args = parse_args()
    # Loading checkpoint
    print('\nloading model checkpoint')
    model = load_checkpoint(args['checkpoint']) 

    ## pre-process and predict the image
    print('\nprocessing and predicting the image')
    probs, classes, names = predict(args['/path/to/image'], 
                                    model, 
                                    args['top_k'], 
                                    args['category_names'],
                                    args['gpu'])

    print('\n\nFinished!')
    print('\nProbabilities:', probs)
    print('\nNames', names)
    
    return probs, names

if __name__ == '__main__':
    main()


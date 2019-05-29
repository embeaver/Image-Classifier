# Imports the libraries
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
from time import time
import matplotlib.pyplot as plt
import json
import numpy as np
from collections import OrderedDict
import argparse


## PARSE THE ARGUMENTS ##
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help = 'path to image folder')
    parser.add_argument('--save_dir', default = 'checkpoints', help = 'directory to save checkpoint')
        
    parser.add_argument('--arch', choices = ['densenet121', 'vgg13', 'alexnet'], 
                        default = 'densenet121', help = 'model type/architecture')
    parser.add_argument('--learning_rate', default = 0.001, help = 'learning rate of model ex: 0.001')
    parser.add_argument('--hidden_units', default = 128, help = 'hidden units in model')
    parser.add_argument('--epochs', default = 10, help = 'number of epochs for training model')
    parser.add_argument('--gpu', action = 'store_true', default = True, help = 'use gpu for training')
    
    args = parser.parse_args()
    args = vars(args)
    
    return args
    

## LOAD  AND TRANSFORM THE DATA FOR TRAINING
def load_data(data_dir = 'flowers'): #, batch_size = 16):
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    # define a transform to normalize the data
    #data_transforms = 

    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                            std = [0.229, 0.224, 0.225])])

    validation_transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                            std = [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                            std = [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    # download the test and training set
    # image_datasets = image_datasets.ImageFolder(data_dir, transform = data_transforms)

    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    validation_data = datasets.ImageFolder(valid_dir, transform = validation_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    #dataloaders = 
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 16, shuffle = True) # batch_size = 32
    validloader = torch.utils.data.DataLoader(validation_data, batch_size = 16) # batch_size = 32
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 16) # batch_size = 32
    
    return trainloader, validloader, testloader, train_data, validation_data, test_data


## LABEL MAPPING
def label_mapping():
    with open(str('cat_to_name.json'), 'r') as f:
        cat_to_name = json.load(f)
    
    types_count = int(len(cat_to_name))
    print('number of categories =', types_count)
    return types_count


## TRAINING THE MODEL ##
def train_model(arch, gpu, trainloader, validloader, learning_rate, epochs, hidden_units, types_count): ####
    
    ## BUILD MODEL ##
    if gpu == True:
        # device will automatically use CUDA if available/enabled
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    # Setting up the model classifier, criterion, and optimizer
    if arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        in_features = 1024
    elif arch == 'vgg13':
        model = models.vgg13(pretrained = True)
        in_features = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        in_features = 9216
        
    # Freeze parameters so we don't backpropogate through them
    for param in model.parameters():
        param.requires_grad = False # error with resnet18 model
    # replace the current classifier with our classifier
    #in_features = 1024 # matches in_features in model architecture
    model.classifier = nn.Sequential(OrderedDict([
                            ('inputs', nn.Linear(in_features, 256)),
                            ('dropout1', nn.Dropout(p = 0.2)),
                            ('hidden_layer1', nn.Linear(256, int(hidden_units))), 
                            ('relu1', nn.ReLU()),
                            ('dropout2', nn.Dropout(p = 0.2)),
                            ('hidden_layer2', nn.Linear(int(hidden_units), types_count)), # 102 is the number of outputs (categories of flowers)
                            ('output', nn.LogSoftmax(dim = 1))
                                        ]))
    # Criterion
    criterion = nn.NLLLoss()
    # Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr = float(learning_rate))
    model.to(device) # sends model to gpu if available, if not jsut uses CPU (much slower)
    
    ## TRAIN MODEL ##
    epochs = int(epochs)
    train_losses, valid_losses = [], []

    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device) # send inputs and labels to gpu if available
            optimizer.zero_grad() # zero the gradients
        
            log_ps = model(inputs) # forward pass
            loss = criterion(log_ps, labels) # calculate loss
            loss.backward() # backward pass to update weights and biases
            optimizer.step() # optimizer
            running_loss += loss.item() # calculate running loss
        
        else:
            valid_loss = 0
            accuracy = 0
        
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval() # turns off dropout for evaluating model
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device) # send to gpu if available
                    log_ps = model(inputs) # forward pass
                    valid_loss += criterion(log_ps, labels) # validation data loss
                
                    ps = torch.exp(log_ps) # probabilities, take exp because the output of the model is LogSoftmax
                    top_p, top_class = ps.topk(1, dim=1) # top probability, top class
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
        
            model.train() # turns on dropout for training model
                
            train_losses.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
        
    #plt.plot(train_losses, label = 'Training Loss')
    #plt.plot(valid_losses, label = 'Validation Loss')
    #plt.legend(frameon = False)
    
    return model, epochs, in_features

##  TEST THE MODEL ##
def test_network(model, dataloader, gpu):
    if gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
        
    model.to(device)
    criterion = nn.NLLLoss()
    # Testing the network on the test data 
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        model.eval() # turns off dropout for evaluating model
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # get class probabilities
            log_ps = model(inputs) # forward pass
            test_loss += criterion(log_ps, labels)
                
            ps = torch.exp(log_ps) # probabilities
            top_p, top_class = ps.topk(1, dim=1) # top predictions
            equals = top_class == labels.view(*top_class.shape) # does the prediction the label
            accuracy += torch.mean(equals.type(torch.FloatTensor)) # calculate accuracy

    print("Test Loss: {:.3f}.. ".format(test_loss/len(dataloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(dataloader)))
 
## SAVE THE MODEL AS A CHECKPOINT ##
def save_checkpoint(model, in_features, hidden_units, train_data, save_dir, epochs, arch, learning_rate):
    model.class_to_idx = train_data.class_to_idx ### uses train_data
    # Saving the model
    #epoch = epochs + 1
    #epoch = str(epochs)
    optimizer = optim.Adam(model.classifier.parameters(), lr = float(learning_rate))
    print(epochs, type(epochs))
    arch = str(arch)
    print(arch, type(arch))
    torch.save({
        'in_features': in_features,
        'hidden_units': hidden_units,
        'epoch': epochs + 1,
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
    }, save_dir + '/' + 'checkpoint.pth')
    
    print('Model saved')
    print(model.classifier)
    return 'checkpoint.pth'


## MAIN PROGRAM ##
def main():
    print("Checking runtime...")
    start_time = time()
    
    args = parse_args()
    # Loading data
    trainloader, testloader, validloader, train_data, test_data, validation_data = load_data(args['data_dir'])
    
    ## Label mapping, counting categories of images
    types_count = label_mapping()
    
    # Build and train the model def train_model(model, gpu, learning_rate, epochs, hidden_units, types_count): ####
    print('\nbuilding and training the model, could take up to 10 mins per epoch on GPU')
    model, epochs, in_features = train_model(args['arch'], # used but not tested on other models
                                    args['gpu'], # used
                                    trainloader,  
                                    validloader,
                                    args['learning_rate'], #used
                                    args['epochs'], # used
                                    args['hidden_units'], # used
                                    types_count)
    
    print('\ntesting network on the test set')
    test_network(model, testloader, args['gpu'])
    
    # Saving the checkpoint          
    print('\nsaving trained model as a checkpoint')
    save_checkpoint(model, 
                    in_features,
                    args['hidden_units'],
                    train_data, 
                    args['save_dir'], 
                    epochs, 
                    args['arch'],
                    args['learning_rate'])
    
    print('\nfinished!')
    end_time = time()
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

if __name__ == '__main__':
    main()

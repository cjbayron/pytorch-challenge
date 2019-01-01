'''Script for training models
'''
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, transforms, datasets

import utils as ut

TRN_DATA_DIR = "assets/flower_data/train"
VAL_DATA_DIR = "assets/flower_data/valid"
BATCH_SIZE = 32
NUM_EPOCHS = 1
LOAD_MODEL = True

# for pipeline testing
TEST_DATA = ['assets/flower_data/train/74/image_01218.jpg',
             'assets/flower_data/train/94/image_07453.jpg']

# 'inception', 'resnet18', 'resnet152',
# 'densenet121', 'densenet201'
MODEL_TO_USE = 'inception'
INPUT_SIZE = {'inception': (299, 299, 3),
              'resnet': (224, 224, 3),
              'densenet': (224, 224, 3)}

MODEL_DIR = 'checkpoints'
MODEL_FN = 'inception_1e.pth'

def count_parameters(model):
    '''Count trainable parameters in model. From:
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_parameter_count():
    '''Print parameter count of state-of-the-art image models
    '''
    image_models = [models.alexnet(),
                    models.vgg11(), models.vgg19(),
                    models.resnet18(), models.resnet152(),
                    models.squeezenet1_0(), models.squeezenet1_1(),
                    models.densenet121(), models.densenet201(),
                    models.inception_v3()]

    for model in image_models:
        print(model.__class__.__name__, count_parameters(model))

def get_data_gen(input_size):
    '''Initialize data loader
    '''
    side_len = min(input_size[:2])
    # for training
    train_transforms = transforms.Compose([transforms.Resize(side_len),
                                           transforms.RandomCrop(side_len),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(TRN_DATA_DIR, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=BATCH_SIZE, shuffle=True)

    # for validation
    valid_transforms = transforms.Compose([transforms.Resize(side_len),
                                           transforms.RandomCrop(side_len),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

    valid_data = datasets.ImageFolder(VAL_DATA_DIR, transform=valid_transforms)
    validloader = torch.utils.data.DataLoader(valid_data,
                                              batch_size=BATCH_SIZE, shuffle=True)

    return trainloader, validloader

def visualize_transforms():
    '''Display effects of torchvision transforms
    '''
    for img in TEST_DATA:
        orig_img = Image.open(img)
        min_size = min(np.array(orig_img).shape[:2])

        trans_funcs = [transforms.RandomResizedCrop(224),
                       transforms.Resize(224),
                       transforms.RandomCrop(min_size),
                       transforms.RandomRotation(30)]

        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))

        ax[0][0].imshow(orig_img)
        ax[0][0].set_title('Original')

        for i, sb in enumerate(ax[1:].flatten()):
            if i >= len(trans_funcs):
                break

            im = trans_funcs[i](orig_img)
            sb.imshow(im)
            sb.set_title(trans_funcs[i].__class__.__name__)

        plt.show()

def train(model_name, model):
    '''Perform model training
    '''
    # prepare data generator
    input_size = INPUT_SIZE[model_name]
    trainloader, validloader = get_data_gen(input_size)

    if LOAD_MODEL:
        ckpt = torch.load(os.path.join(MODEL_DIR, MODEL_FN))
        model.load_state_dict(ckpt)

    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('Training on GPU.')
        model.cuda()
    else:
        print('No GPU available, training on CPU.')

    # prepare loss and optimizer functions
    loss_f = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                     patience=1, verbose=True)

    # perform training
    train_losses, val_losses = [], []
    best_val_loss = np.inf
    for e in range(NUM_EPOCHS):
        running_loss = 0
        model.train()
        for images, labels in tqdm(trainloader):
            # move data to gpu
            if train_on_gpu:
                images, labels = images.cuda(), labels.cuda()
            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()
            # Forward pass, then backward pass, then update weights
            logits = model.forward(images)
            loss = loss_f(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            with torch.no_grad():
                total = 0
                corrects = 0
                val_loss = 0
                model.eval()
                for images, labels in tqdm(validloader):
                    # move data to gpu
                    if train_on_gpu:
                        images, labels = images.cuda(), labels.cuda()

                    logits = model.forward(images)
                    val_loss += loss_f(logits, labels)

                    probs = F.softmax(logits, dim=1)
                    preds = probs.cpu().numpy().argmax(axis=1)
                    preds = torch.from_numpy(preds)
                    if train_on_gpu:
                        preds = preds.cuda()
                    corrects += torch.sum(preds == labels).type(torch.FloatTensor)
                    total += len(labels)

            accuracy = float(corrects / total)
            train_losses.append(running_loss/len(trainloader))
            val_losses.append(val_loss/len(validloader))

            print("Epoch: {}/{}.. ".format(e+1, NUM_EPOCHS),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Validation Loss: {:.3f}.. ".format(val_loss/len(validloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy))

            print(f"Training loss: {running_loss/len(trainloader)}")

            if val_loss < best_val_loss:
                # save model
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, MODEL_FN))
                print("Saved to %s." % os.path.join(MODEL_DIR, MODEL_FN))
                best_val_loss = val_loss

            # adjust learning rate based on validation loss
            scheduler.step(val_loss)

    # graph training and validation loss
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

def main():
    '''Main
    '''
    global MODEL_TO_USE
    # Print parameter count of available models. This is for model selection.
    #print_model_parameter_count()

    if MODEL_TO_USE == 'inception':
        model = ut.get_modified_inception()
    elif 'resnet' in MODEL_TO_USE:
        sz = int(MODEL_TO_USE.strip('resnet'))
        model = ut.get_modified_resnet(sz)
        MODEL_TO_USE = 'resnet'
    elif 'densenet' in MODEL_TO_USE:
        sz = int(MODEL_TO_USE.strip('densenet'))
        model = ut.get_modified_densenet(sz)
        MODEL_TO_USE = 'densenet'
    else:
        print("Unsupported model type!")
        return -1

    # Visualize transforms. This is for transforms selection.
    # visualize_transforms()

    train(MODEL_TO_USE, model)

    return 0

if __name__ == "__main__":
    main()

'''Script for testing trained models
'''
import os
from heapq import nlargest

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets

import utils as ut

TRN_DATA_DIR = "assets/flower_data/train"
VAL_DATA_DIR = "assets/flower_data/valid"
BATCH_SIZE = 32

# 'inception', 'resnet18', 'resnet152',
# 'densenet121', 'densenet201'
MODEL_TO_USE = 'densenet201'
#MODEL_TO_USE = 'resnet152'
INPUT_SIZE = {'inception': (299, 299, 3),
              'resnet': (224, 224, 3),
              'densenet': (224, 224, 3)}

MODEL_DIR = 'checkpoints'
MODEL_FN = 'densenet201_classifier_20e_final.pth'

NUM_TEST = 5

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

def get_top_misclassified(labels, preds, k=5):
    '''Get top K misclassified classes
    '''
    labels = np.array(labels)
    preds = np.array(preds)

    # get labels where there is mismatch
    mis_labels = np.unique(labels[labels != preds])
    # for tracking of misclassification
    err_list = [0] * ut.OUTPUT_SIZE
    err_detail_map = {}
    mis_dict = {}

    for lab in mis_labels:
        lab_idxs = (labels == lab)
        total = lab_idxs.sum()

        preds_for_lab = preds[lab_idxs]
        mis_preds = (preds_for_lab != lab)
        wrong = mis_preds.sum()

        err_list[lab-1] = float(wrong / total)
        err_detail_map[lab] = {'wrong': wrong, 'total': total}
        mis_dict[lab] = preds_for_lab[mis_preds]

    # get indices,value of elements with largest values
    idx_err_pairs = nlargest(k, enumerate(err_list), key=lambda x: x[1])
    for idx, err in idx_err_pairs:
        if err == 0:
            continue

        lab = idx+1
        print("%d: %0.2f (%d/%d), Preds: " % (lab, err, err_detail_map[lab]['wrong'],
                                              err_detail_map[lab]['total']), mis_dict[lab])

def test(model_name, model):
    '''Perform model training
    '''
    # prepare data generator
    input_size = INPUT_SIZE[model_name]
    trainloader, validloader = get_data_gen(input_size)

    # load model
    ckpt = torch.load(os.path.join(MODEL_DIR, MODEL_FN))
    model.load_state_dict(ckpt)

    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('Testing on GPU.')
        model.cuda()
    else:
        print('No GPU available, testing on CPU.')

    # prepare loss function
    loss_f = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []

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

            all_labels.extend(labels.cpu().numpy().squeeze().tolist())
            all_preds.extend(preds.cpu().numpy().squeeze().tolist())

    accuracy = float(corrects / total)

    print("Validation Loss: {:.3f}.. ".format(val_loss/len(validloader)),
          "Validation Accuracy: {:.3f}".format(accuracy))

    # display confusion matrix (WARNING: SLOW!)
    # print("Plotting confusion matrix...")
    # plot_confusion_matrix(all_labels, all_preds, normalize=True)
    # plt.show()

    #get_top_misclassified(all_labels, all_preds)
    return accuracy

def main():
    '''Main
    '''
    global MODEL_TO_USE

    if MODEL_TO_USE == 'inception':
        model = ut.get_modified_inception(pretrained=False)
    elif 'resnet' in MODEL_TO_USE:
        sz = int(MODEL_TO_USE.strip('resnet'))
        model = ut.get_modified_resnet(sz, pretrained=False)
        MODEL_TO_USE = 'resnet'
    elif 'densenet' in MODEL_TO_USE:
        sz = int(MODEL_TO_USE.strip('densenet'))
        model = ut.get_modified_densenet(sz, pretrained=False)
        MODEL_TO_USE = 'densenet'
    else:
        print("Unsupported model type!")
        return -1

    acc = 0
    for i in range(NUM_TEST):
        acc += test(MODEL_TO_USE, model)

    acc /= NUM_TEST
    print("Average Accuracy: {:.3f}".format(acc))

    return 0

if __name__ == "__main__":
    main()

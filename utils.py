'''Utility functions
'''
from torch import nn
from torchvision import models

OUTPUT_SIZE = 102

def get_modified_inception(pretrained=True):
    '''Return inception3 with frozen layers
    '''
    model = models.inception_v3(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False

    # replace fc
    classifier = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(1024, OUTPUT_SIZE)
    )

    model.fc = classifier
    # work around to remove aux from output
    model.aux_logits = False

    return model

def get_modified_resnet(size, pretrained=True):
    '''Return resnet with frozen layers
    '''
    if size == 18:
        model = models.resnet18(pretrained=pretrained)
        in_features = 512
    elif size == 152:
        model = models.resnet152(pretrained=pretrained)
        in_features = 2048
    else:
        raise Exception("Unsupported model type!")

    for param in model.parameters():
        param.requires_grad = False

    # replace fc
    classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(512, OUTPUT_SIZE)
    )

    model.fc = classifier

    return model

def get_modified_densenet(size, pretrained=True):
    '''Return densenet with frozen layers
    '''
    if size == 121:
        model = models.densenet121(pretrained=pretrained)
        in_features = 1024
    elif size == 201:
        model = models.densenet201(pretrained=pretrained)
        in_features = 1920
    else:
        raise Exception("Unsupported model type!")

    for param in model.parameters():
        param.requires_grad = False

    # replace classifier
    classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(512, OUTPUT_SIZE)
    )

    model.classifier = classifier

    return model

from torch import nn
from torch import optim

def get_criterion(cfg):
    if cfg['training']['criterion']['name'] == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()

def get_optimizer(cfg, model_parameters):
    OPTIM_NAME = cfg['training']['optimizer']['name'].lower()
    LR = cfg['training']['learning_rate']
    if OPTIM_NAME == 'sgd':
        return optim.SGD(model_parameters, lr=LR, momentum=0.9, weight_decay=5e-4)
    elif OPTIM_NAME == 'adam':
        return optim.Adam(model_parameters, lr=LR)

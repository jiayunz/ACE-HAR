import torch

from .models.lenet import LeNet
from .models.resnet import ResNet18, ResNet_1layer

def build_model(model_name, task, n_class, device):
    if model_name == 'ResNet18':
        model = ResNet18(n_class)
    elif model_name == 'ResNet_1layer':
        model = ResNet_1layer(num_blocks=[2], num_classes=n_class)
    elif model_name == 'LeNet':
        model = LeNet(n_class)
    else:
        raise ValueError('Wrong model name:', model_name)

    if task.startswith('cifar10'):
        dummy_input = torch.randn(1, 3, 32, 32, device=device)

    model.dummy_input = dummy_input

    return model


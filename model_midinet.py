from model_resnet import ResNet50

import torch.nn as nn
import torch.optim as optim

class MidiNet(nn.Module):
    def __init__(self):
        super().__init__()

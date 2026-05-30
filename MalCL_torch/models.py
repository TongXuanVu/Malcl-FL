import numpy as np
import os
import os.path as opth
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input_dim = 64
        self.output_features = 33
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_features),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def reinit(self):
        self.apply(weights_init)

    def forward(self, input):
        return self.fc(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_features = 33
        self.output_dim = 1
        self.latent_dim = 256
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, self.latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_dim),
            nn.Linear(self.latent_dim, self.output_dim),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def reinit(self):
        self.apply(weights_init)

    def forward(self, input):
        x = input.view(-1, 1, self.input_features)
        x = self.conv(x)
        feature = x.view(-1, 32)
        x = self.fc(feature)
        return x.view(-1, 1), feature
    
class Classifier(nn.Module):
    def __init__(self, init_classes=6):
        super(Classifier, self).__init__()
        self.input_features = 33
        self.output_dim = init_classes
        self.drop_prob = 0.5
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.Dropout(self.drop_prob),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.Dropout(self.drop_prob),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc1_f = nn.Flatten()
        self.fc1 = nn.Linear(32, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)
        self.fc1_drop1 = nn.Dropout(self.drop_prob)
        self.fc1_act1 = nn.ReLU()
        self.apply(weights_init)

    def reinit(self):
        self.apply(weights_init)

    def forward(self, x):
        original_shape = x.size()
        batch_size = original_shape[0]
        if len(original_shape) == 3:
            batch_size = original_shape[0] * original_shape[1]
            x = x.view(batch_size, self.input_features)
        x = x.view(batch_size, 1, self.input_features)
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc1_f(x)
        x = self.fc1(x)
        x = self.fc1_bn1(x)
        x = self.fc1_drop1(x)
        x = self.fc1_act1(x)
        if len(original_shape) == 3:
            x = x.view(original_shape[0], original_shape[1], -1)
        return x

    def expand_output_layer(self, init_classes, nb_inc, task, target_dim=None):
        old_fc1 = self.fc1
        old_fc1_bn1 = self.fc1_bn1
        if target_dim is not None:
            self.output_dim = target_dim
        else:
            self.output_dim = init_classes + nb_inc * task
            
        self.fc1 = nn.Linear(old_fc1.in_features, self.output_dim)
        self.fc1_bn1 = nn.BatchNorm1d(self.output_dim)
        with torch.no_grad():
            # Copy old weights up to the minimum of old and new dimensions
            copy_out = min(old_fc1.out_features, self.output_dim)
            self.fc1.weight[:copy_out].copy_(old_fc1.weight.data[:copy_out])
            self.fc1.bias[:copy_out].copy_(old_fc1.bias.data[:copy_out])
            
            copy_bn = min(old_fc1_bn1.num_features, self.output_dim)
            self.fc1_bn1.weight[:copy_bn].copy_(old_fc1_bn1.weight.data[:copy_bn])
            self.fc1_bn1.bias[:copy_bn].copy_(old_fc1_bn1.bias.data[:copy_bn])
        return self

    def predict(self, x_data):
        return self.forward(x_data)

    def get_logits(self, x):
        original_shape = x.size()
        batch_size = original_shape[0]
        if len(original_shape) == 3:
            batch_size = original_shape[0] * original_shape[1]
            x = x.view(batch_size, self.input_features)
        x = x.view(batch_size, 1, self.input_features)
        x = self.block1(x)
        x = self.block2(x)
        logits = self.fc1_f(x)
        return logits.detach()

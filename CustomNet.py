import torch.nn as nn
import torchvision.models
from torchvision.models.resnet import ResNet18_Weights


class encoder_decoder:

    encoder = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = encoder.fc.in_features
    encoder.fc = nn.Linear(num_features,100)

    num_features = encoder.fc.out_features

    frontend = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(100, 2),
        nn.Sigmoid()
    )


class CustomNet(nn.Module):
    def __init__(self, encoder=None):
        super(CustomNet, self).__init__()
        self.encoder = encoder
        self.frontend = encoder_decoder.frontend
        self.frontend_weights(mean=0.0, std=0.01)

        if self.encoder is None:
            self.encoder = encoder_decoder.encoder

        self.mse_loss = nn.MSELoss()

    def frontend_weights(self, mean, std):
        for param in self.frontend.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def forward(self, X):
        X = self.encoder(X)
        X = self.frontend(X)
        return X

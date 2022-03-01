import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.encoder = nn.Sequential(
        nn.Conv2d(channels, 256, 3, padding=1),
        nn.PReLU(256),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.PReLU(256),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.PReLU(256),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(256, 512, 3, padding=1),
        nn.PReLU(512),
        nn.MaxPool2d(2, 2),
        #
        nn.Conv2d(512, 1024, 3, padding=1),
        nn.PReLU(1024),
        #nn.MaxPool2d(2, 2),
        #
        #nn.Flatten(),
        #nn.Linear(1024*4*4, self.hidden),
        )

    def forward(self, x):
        encoded = self.encoder(x) # encoded
        return encoded




class Decoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        # self.dec_linear = nn.Sequential(
        #     nn.Linear(self.hidden,1024*4*4),
        #     nn.PReLU(1024*4*4)
        # )

        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=1),
        nn.PReLU(512),
        nn.ConvTranspose2d(512, 256, 2, stride=2),
        nn.PReLU(256),
        nn.ConvTranspose2d(256, 256, 2, stride=2),
        nn.PReLU(256),
        #
        nn.ConvTranspose2d(256, 256, 2, stride=2),
        nn.PReLU(256),
        #
        nn.ConvTranspose2d(256, channels, 3, stride=1, padding=1),
        nn.Sigmoid()
        )

    def forward(self, x):
        #dec_flatten = self.dec_linear(x)
        #dec_unflatten = dec_flatten.view(x.shape[0], 1024 , 4 , 4)
        decoded = self.decoder(x)
        return decoded
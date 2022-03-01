import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self, hidden, channels):
        super().__init__()
        self.hidden = hidden

        self.encoder = nn.Sequential(
        nn.Conv2d(channels, 64, 3, padding=1),
        nn.PReLU(64),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.PReLU(128),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.PReLU(256),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.PReLU(256),
        nn.MaxPool2d(2, 2),
        #
        #nn.Conv2d(256, 256, 3, padding=1),
        #nn.PReLU(256),
        #nn.MaxPool2d(2, 2),
        #
        nn.Flatten(),
        nn.Linear(256*6*6, self.hidden),
        )

    def forward(self, x):
        encoded = self.encoder(x) # encoded
        return encoded




class Decoder(torch.nn.Module):
    def __init__(self, hidden, channels):
        super().__init__()
        self.hidden = hidden

        self.dec_linear = nn.Sequential(
            nn.Linear(self.hidden,256*6*6),
            nn.PReLU(256*6*6)
        )

        self.decoder = nn.Sequential(
        #nn.ConvTranspose2d(256, 256, 2, stride=2),
        #nn.PReLU(256),
        nn.ConvTranspose2d(256, 256, 2, stride=2),
        nn.PReLU(256),
        nn.ConvTranspose2d(256, 128, 2, stride=2),
        nn.PReLU(128),
        #
        nn.ConvTranspose2d(128, 64, 2, stride=2),
        nn.PReLU(64),
        #
        nn.ConvTranspose2d(64, channels, 3, stride=1, padding=1),
        nn.Sigmoid()
        )

    def forward(self, x):
        dec_flatten = self.dec_linear(x)
        dec_unflatten = dec_flatten.view(x.shape[0], 256 , 6 , 6)
        decoded = self.decoder(dec_unflatten)
        return decoded
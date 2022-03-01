import torch
import torch.nn as nn

class Encoder(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden

        self.encoder = nn.Sequential(
        nn.Linear(48*48*9, 10000),
        nn.ReLU(True),
        nn.Linear(10000, self.hidden),
        #nn.BatchNorm1d(self.hidden),
        nn.ReLU(True),
        )

    def forward(self, x):
        batch, ch, w, h = x.shape
        x = x.view(batch, ch * w * h)
        encoded = self.encoder(x) # encoded
        return encoded




class Decoder(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden

        self.decoder = nn.Sequential(
        nn.Linear(self.hidden, 10000),
        nn.ReLU(True),
        nn.Linear(10000, 48*48*9),
        #nn.BatchNorm1d(48*48*9),
        nn.ReLU(True),
        )

    def forward(self, x, ch, w, h):
        decoded = self.decoder(x)
        decoded = decoded.view(x.shape[0], ch , w , h)
        return decoded
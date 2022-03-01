import rasterio
import torch
import numpy as np
import torch.nn.functional as F
import utils
import torch.nn as nn

# with rasterio.open("1012_20170817.tif") as tif:
#   x = tif.read()
# torch_x = torch.FloatTensor(x)
# print(x.shape)
# print(torch_x.shape)
# print(np.sum(x - torch_x.numpy()))

class SimpleAutoencoder(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden

        self.encoder = nn.Sequential(
        nn.Linear(48*48*9, 10000),
        nn.BatchNorm1d(10000),
        nn.ReLU(True),
        nn.Linear(10000, self.hidden),
        #nn.BatchNorm1d(self.hidden),
        nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
        nn.Linear(self.hidden, 10000),
        nn.BatchNorm1d(10000),
        nn.ReLU(True),
        nn.Linear(10000, 48*48*9),
        #nn.BatchNorm1d(48*48*9),
        nn.ReLU(True),
        )

    def forward(self, x):
        batch, ch, w, h = x.shape
        x = x.view(batch, ch * w * h)
        encoded = self.encoder(x) # encoded
        decoded = self.decoder(encoded)
        decoded = decoded.view(batch, ch , w , h)
        return encoded, decoded

# model = SimpleAutoencoder()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# criterion = torch.nn.MSELoss()

# epochs = 150
# data = torch_x.unsqueeze(0)
# print(data.shape)
# for epoch in range(epochs):
#   print(f'{epoch+1}/{epochs}')
#   encoded, decoded = model(data)
#   loss = criterion(data, decoded)
#   optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()
#   utils.save_rgb(decoded,'/home/super/rlagrassa/inaf_compression_project/experiments/aae_two_splitted_DeconvTest/results_slice_1000/test.png')
#
#   print('\t loss:', loss.item())
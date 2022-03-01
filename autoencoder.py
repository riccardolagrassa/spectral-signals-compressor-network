from torch import nn



class Net(nn.Module):
    def __init__(self, output_aut):
        super(Net, self).__init__()
        self.output_aut=output_aut

        self.encoder = nn.Sequential(
            nn.Conv2d(9, 128, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, self.output_aut, 3, stride=2, padding=1),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.output_aut, 256, 2, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=4, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=3, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 9, 2, stride=2, padding=0),
            nn.ReLU(True)
            # nn.Sigmoid()
        )

    def forward(self, x):
        code = self.encoder(x)
        output = self.decoder(code)
        return output, code

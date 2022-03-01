import torch
import torch.nn as nn

# class SNet(torch.nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.channels = channels
#
#         self.snet = nn.Sequential(
#         nn.Conv2d(9, 64, 3, padding=1),
#         nn.ReLU(True),
#         nn.Conv2d(64, 64, 3, padding=1),
#         nn.ReLU(True),
#         nn.MaxPool2d(2, 2),
#         nn.Flatten(),
#         nn.Linear(64*24*24, 9),
#         nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         selection_bands = self.snet(x)
#         test, sorted_bands = torch.topk(selection_bands, self.channels, dim=1)
#         a = torch.index_select(x[0], 0, sorted_bands[0]).unsqueeze(0)
#         for j in range(1, x.shape[0]):
#             a=torch.vstack((a, torch.index_select(x[j], 0, sorted_bands[j]).unsqueeze(0)))
#         return a, sorted_bands


class Encoder(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.fc_merge= nn.Linear(self.hidden*2, self.hidden)

        self.encoder_branch1 = nn.Sequential(
        nn.Conv2d(9, 64, 3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(256*6*6, self.hidden)
        )

        self.encoder_branch2 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(9, 3, 3), stride=(1, 1, 1), padding=(4, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(64, 128, kernel_size=(9, 3, 3), stride=(1, 1, 1), padding=(4, 1, 1)),
            nn.ReLU(True),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(128, 256, kernel_size=(9, 3, 3), stride=(1, 1, 1), padding=(4, 1, 1)),
            nn.ReLU(True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6 * 2, self.hidden)
        )



    def forward(self, x):
        encoded1 = self.encoder_branch1(x) # encoded1
        encoded2 = self.encoder_branch2(x.unsqueeze(1)) # encoded2
        encoded_merged = torch.hstack((encoded1, encoded2))
        final_encoded=self.fc_merge(encoded_merged)
        return final_encoded




class Decoder(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.dec_linear = nn.Linear(self.hidden,256*6*6)
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(256, 256, 2, stride=2),
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, 2, stride=2),
        nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 2, stride=2),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 9, 3, stride=1, padding=1),
        nn.Sigmoid()
        )

    def forward(self, x):
        dec_flatten = self.dec_linear(x)
        dec_unflatten = dec_flatten.view(x.shape[0], 256 , 6 , 6)
        decoded = self.decoder(dec_unflatten)
        return decoded
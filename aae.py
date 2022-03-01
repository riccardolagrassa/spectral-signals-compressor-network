import argparse
import os

import piq
import rasterio
import shutil

import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torch.backends import cudnn
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from dataset import BaseDataset_wth_folders

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=250, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=512, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=48, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=9, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

torch.cuda.manual_seed_all(0)                       # Set random seed.
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

if torch.cuda.is_available():
    cudnn.enabled = True
    cudnn.benchmark  = True
else:
    print("Error in GPU.")
    exit()


def save_tiff(path, data):
    data_unnormalize = data * torch.tensor([21527]).cuda()  #19017 for 1k and 5k, 21527 for 100k
    data = data_unnormalize.cpu().numpy()
    d=rasterio.open(path, 'w', driver='GTiff', crs='EPSG:32632', nodata=None, height=data.shape[1], width=data.shape[2], count=9, dtype=np.uint16)

    for i in range(9):
        d.write(data[i],i+1)



def validate() -> float:
    # Calculate how many iterations there are under epoch.
    batches = len(valid_dataloader)
    # Set generator model in verification mode.
    encoder.eval()
    decoder.eval()
    # Initialize the evaluation index.
    total_psnr_value, total_ssim,  = 0.0, 0.0
    total_psnr_bands=[0 for i in range(9)]
    total_ssim_bands=[0 for i in range(9)]

    with torch.no_grad():
        for index, (data, filename_packed) in enumerate(valid_dataloader):
            data = data.cuda()
            encoded_imgs = encoder(data)
            output = decoder(encoded_imgs)

            for j, filename, real_data in zip(output, filename_packed,data):
                #torchvision.utils.save_image(j, results_dir+'/'+filename+'.png')
                #torchvision.utils.save_image(real_data, results_dir+'/'+filename+'_GT_'+'.png')

                save_tiff(results_dir+'/'+filename, j)
                #save_tiff(results_dir+'/'+filename+'_GT', real_data)


            output = output * torch.tensor([21527]).cuda()  # 19017 for 1k and 5k, 21527 for 100k
            data = data * torch.tensor([21527]).cuda()  # 19017 for 1k and 5k, 21527 for 100k

            pq_psnr = piq.psnr(output, data, data_range=21527.)
            pq_ssim = piq.ssim(output, data, data_range=21527.)
            total_psnr_value += pq_psnr
            total_ssim += pq_ssim

            for b in range(9):
                pq_psnr = piq.psnr(output[:, None, b, :, :], data[:, None,b, :, :], data_range=21527.)
                pq_ssim=piq.ssim(output[:, None,b, :, :], data[:, None,b, :, :], data_range=21527.)
                total_psnr_bands[b]+=pq_psnr
                total_ssim_bands[b]+=pq_ssim


        avg_psnr_value = total_psnr_value / batches
        avg_ssim_value = total_ssim / batches

        avg_total_psnr_bands=[i/batches for i in total_psnr_bands]
        avg_total_ssim_bands=[i/batches for i in total_ssim_bands]

        print(f"Valid stage: Epoch[{epoch + 1:04d}]")
        print(f"avg PSNR: {avg_psnr_value:.3f} avg SSIM: {avg_ssim_value:.3f}")
        print("Bands evaluation")
        for b in range(9):
             print(f"avg PSNR: {avg_total_psnr_bands[b]:.3f} avg SSIM: {avg_total_ssim_bands[b]:.3f}")
        print("**************************")

    return avg_psnr_value, avg_ssim_value

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            # nn.Linear(int(np.prod(img_shape)), 512),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(9, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(3, stride=3),
        )

        self.mu = nn.Linear(256, opt.latent_dim)
        self.logvar = nn.Linear(256, opt.latent_dim)

    def forward(self, img):
        #img_flat = img.view(img.shape[0], -1)
        x = self.model(img)
        x_flat = x.view(img.shape[0], -1)
        mu = self.mu(x_flat)
        logvar = self.logvar(x_flat)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Sigmoid()
            #nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


dataset_slice=10000


# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCEWithLogitsLoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

encoder.cuda()
decoder.cuda()
discriminator.cuda()
adversarial_loss.cuda()
pixelwise_loss.cuda()

# Configure data loader
# train_dir = '/home/super/datasets-nas/mars_dataset/train/'
# valid_dir = '/home/super/datasets-nas/mars_dataset/valid/'

train_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/train/'
valid_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/test/'

results_dir = '/home/super/rlagrassa/inaf_compression_project/experiments/aae_2/results_slice_'+str(dataset_slice)

if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)


train_dataset = BaseDataset_wth_folders(train_dir, "train")
valid_dataset = BaseDataset_wth_folders(valid_dir, "validate")

train_dataloader = DataLoader(train_dataset, opt.batch_size, True)
valid_dataloader = DataLoader(valid_dataset, opt.batch_size, False)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor #if cuda else torch.FloatTensor


def sample_image(n_row, batches_done, path):
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    gen_imgs = decoder(z)
    save_image(gen_imgs.data, path+'/%d.png' % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------
encoder.train()
decoder.train()
discriminator.train()

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(train_dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.0005 * adversarial_loss(discriminator(encoded_imgs), valid) + 1.0 * pixelwise_loss(
            decoded_imgs, real_imgs
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(train_dataloader) + i
        if batches_done % opt.sample_interval == 0:
            avg_psnr_value, avg_ssim_value=validate()
            encoder.train()
            decoder.train()
            #sample_image(n_row=10, batches_done=batches_done, path=results_dir)

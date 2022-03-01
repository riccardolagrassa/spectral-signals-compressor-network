import argparse
import os
import time

from osgeo import gdal
from torch.optim import lr_scheduler
from torchvision.utils import save_image

import Ds_build
import piq
import shutil

import numpy as np
import math
import itertools
import torchvision
import torchvision.transforms as transforms
from torch.backends import cudnn

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import utils
from CAE_threeD_model import Decoder, Encoder
from dataset import BaseDataset_wth_folders

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=400, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=1000, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=48, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=9, help="number of image channels")
parser.add_argument("--save_path_model", type=str, default='None', help="save path model")
parser.add_argument("--resume", type=bool, default=False, help="resume training")
parser.add_argument("--model_path_encoder", type=str, default='None', help="path pretrained")
parser.add_argument("--model_path_decoder", type=str, default='None', help="path pretrained")
parser.add_argument("--model_path_selector", type=str, default='None', help="path pretrained")

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
img_v_shape = (4, opt.img_size, opt.img_size)
img_ext_shape = (5, opt.img_size, opt.img_size)


torch.cuda.manual_seed_all(0)                       # Set random seed.
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'

if torch.cuda.is_available():
    cudnn.enabled = True
    cudnn.benchmark  = True
else:
    print("Error in GPU.")
    exit()



def save_tiff_gdal(path, data, ds):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path, ds.RasterXSize, ds.RasterYSize, ds.RasterCount, gdal.GDT_Float32)
    outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  ##sets same projection as input

    for i in range(9):
        outdata.GetRasterBand(i + 1).WriteArray(data[i, :, :])

    outdata.FlushCache()


def validate():
    # Calculate how many iterations there are under epoch.
    batches = len(valid_dataloader)
    # Set generator model in verification mode.
    encoder.eval()
    decoder.eval()
    # Initialize the evaluation index.
    total_psnr_value, total_ssim,  = 0.0, 0.0
    total_psnr_bands=[0 for i in range(9)]
    total_ssim_bands=[0 for i in range(9)]
    total_fsim_bands=[0 for i in range(9)]

    with torch.no_grad():
        for index, (data, filename_packed) in enumerate(valid_dataloader):
            data = data.cuda()
            encoded = encoder(data)
            decoded = decoder(encoded)
            for j, filename, real_data in zip(decoded, filename_packed,data):
                ds=Ds_build.main(valid_dir+filename)

                #utils.save_rgb(j.cpu().numpy(), results_dir+'/'+filename+'.png')
                #torchvision.utils.save_image(j, results_dir+'/'+filename+'.png')
                #torchvision.utils.save_image(real_data, results_dir+'/'+filename+'_GT_'+'.png')

                #save_tiff_gdal(results_dir+'/'+filename, j.cpu().numpy(), ds)

                #save_tiff_gdal(results_dir+'/'+filename+'_GT',real_data.reshape(9, 48, 48), ds)




            #data = (data- torch.min(data))/(torch.max(data) - torch.min(data))
            #decoded = (decoded- torch.min(decoded))/(torch.max(decoded) - torch.min(decoded))

            pq_psnr = piq.psnr(decoded, data, data_range=1.)
            pq_ssim = piq.ssim(decoded, data, data_range=1.)

            total_psnr_value += pq_psnr
            total_ssim += pq_ssim

            for b in range(9):
                pq_psnr = piq.psnr(decoded[:, None, b, :, :], data[:, None,b, :, :], data_range=1.)
                pq_ssim=piq.ssim(decoded[:, None,b, :, :], data[:, None,b, :, :], data_range=1.)
                pq_fsim=piq.fsim(decoded[:, None,b, :, :], data[:, None,b, :, :], data_range=1., chromatic=False)

                total_psnr_bands[b]+=pq_psnr
                total_ssim_bands[b]+=pq_ssim
                total_fsim_bands[b]+=pq_fsim


        avg_psnr_value = total_psnr_value / batches
        avg_ssim_value = total_ssim / batches

        avg_total_psnr_bands=[i/batches for i in total_psnr_bands]
        avg_total_ssim_bands=[i/batches for i in total_ssim_bands]
        avg_total_fsim_bands=[i/batches for i in total_fsim_bands]


        print(f"Valid stage: Epoch[{epoch + 1:04d}]")
        print(f"avg PSNR: {avg_psnr_value:.3f} avg SSIM: {avg_ssim_value:.3f}")
        print("Bands evaluation")
        for b in range(9):
             print(f"avg PSNR: {avg_total_psnr_bands[b]:.3f} avg SSIM: {avg_total_ssim_bands[b]:.3f}, avg FSIM: {avg_total_fsim_bands[b]:.3f}""")
        print("**************************")
        return avg_psnr_value.cpu().numpy(),avg_ssim_value.cpu().numpy()



dataset_slice=100000


criterion = torch.nn.BCELoss()
encoder = Encoder(opt.latent_dim)
decoder = Decoder(opt.latent_dim)
criterion.cuda()


train_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/train/'
valid_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/test/'

results_dir = '/home/super/rlagrassa/inaf_compression_project/experiments/CAE3D_100k/results_slice_'+str(dataset_slice)+'_'+str(opt.latent_dim)
save_path_model=opt.save_path_model

if os.path.exists(save_path_model):
    shutil.rmtree(save_path_model)
os.makedirs(save_path_model)

if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)


train_dataset = BaseDataset_wth_folders(train_dir, "train")
valid_dataset = BaseDataset_wth_folders(valid_dir, "validate")

train_dataloader = DataLoader(train_dataset, opt.batch_size, True, num_workers=8)
valid_dataloader = DataLoader(valid_dataset, opt.batch_size, False, num_workers=8)

encoder = nn.DataParallel(encoder).cuda()
decoder = nn.DataParallel(decoder).cuda()

if opt.resume:
    state_dict_encoder = torch.load(opt.model_path_encoder)
    encoder.load_state_dict(state_dict_encoder)
    state_dict_decoder = torch.load(opt.model_path_decoder)
    decoder.load_state_dict(state_dict_decoder)

optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr
)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer_G, opt.n_epochs)

Tensor = torch.cuda.FloatTensor


def sample_image(n_row, batches_done, path):
    data, _ = next(iter(valid_dataloader))
    #selection_bands, _ = snet(data)
    encoded = encoder(data)
    decoded=decoder(encoded)
    gen_imgs = decoded[:,0:3,:,:]
    save_image(gen_imgs.data, path+'/best_decoded_grid.png', nrow=n_row, normalize=False)


# ----------
#  Training
# ----------


def train(epoch):
    for i, (imgs, filename) in enumerate(train_dataloader):
        real_imgs = imgs.cuda()

        encoded  = encoder(real_imgs)
        decoded  = decoder(encoded)

        g_loss =  criterion(decoded,real_imgs)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_dataloader), g_loss.item())
        )


encoder.train()
decoder.train()
#snet.train()
start = time.time()


def save_model(avg_psnr_value, avg_ssim_value, epoch, avg_score_values):
    if avg_ssim_value > avg_score_values:
        print("Epoch: ", epoch," SSIM: ", avg_ssim_value, " PSNR: ",
              avg_psnr_value)
        torch.save(encoder.state_dict(), os.path.join(opt.save_path_model, "CAE3D_encoder_best.pth"))
        torch.save(decoder.state_dict(), os.path.join(opt.save_path_model, "CAE3D_decoder_best.pth"))
        avg_score_values = avg_ssim_value
        sample_image(n_row=10, batches_done=epoch, path=results_dir)
    return avg_score_values



avg_score_values = 0.0

for epoch in range(opt.n_epochs):
    train(epoch)
    if (epoch + 1) % 1 == 0:
        avg_psnr_value,avg_ssim_value=validate()
        avg_score_values=save_model(avg_psnr_value, avg_ssim_value, epoch, avg_score_values)
        encoder.train()
        decoder.train()
    scheduler.step()
print("Training time: ", time.time() - start)
import argparse
import os
import time
from osgeo import gdal
from torch.optim import lr_scheduler
from torchvision.utils import save_image

import CAE_model
import Ds_build
import matplotlib.pyplot as plt
import SSCNet_lastconv
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
from dataset_load_imagenet import MyDataset_plus_aug
import utils
from dataset import BaseDataset_wth_folders
plt.rcParams.update({'figure.max_open_warning': 0})
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=432, help="number of image channels")
parser.add_argument("--save_path_model", type=str, default='None', help="save path model")
parser.add_argument("--dataset_name", type=str, default='', help="dataset")

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


torch.cuda.manual_seed_all(0)                       # Set random seed.
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if torch.cuda.is_available():
    cudnn.enabled = True
    cudnn.benchmark  = True
else:
    print("Error in GPU.")
    exit()

def perchannel_unnormalize(data):
    data= (train_dataset.max_channels_repeated - train_dataset.min_channels_repeated) * data + train_dataset.min_channels_repeated
    return data

def save_tiff_gdal(path, data):
    # min = -305.55
    # max = 1279.7
    data=perchannel_unnormalize(data)
    data=data.numpy()
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path, data.shape[1], data.shape[2], data.shape[0], gdal.GDT_Float32)
    #outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
    #outdata.SetProjection(ds.GetProjection())  ##sets same projection as input

    for i in range(opt.channels):
        outdata.GetRasterBand(i + 1).WriteArray(data[i, :, :])
    outdata.FlushCache()


def validate(encoder, decoder, valid_dataloader, results_dir, valid_dir, bands) -> float:
    # Calculate how many iterations there are under epoch.
    batches = len(valid_dataloader)
    # Set generator model in verification mode.
    # Initialize the evaluation index.
    total_psnr_value, total_ssim, total_ms_ssim  = 0.0, 0.0, 0.0
    total_psnr_bands=[0 for i in range(bands)]
    total_ssim_bands=[0 for i in range(bands)]
    total_fsim_bands=[0 for i in range(bands)]

    with torch.no_grad():
        for index, (data, filename_packed) in enumerate(valid_dataloader):
            data = data.cuda()
            encoded = encoder(data)
            decoded = decoder(encoded)
            #for j, filename, real_data in zip(decoded, filename_packed,data):
                #ds=Ds_build.main(valid_dir+filename)
                #utils.save_rgb(j.cpu().numpy(), results_dir+'/'+filename+'.png')
                #torchvision.utils.save_image(j, results_dir+'/'+filename+'.png')
                #torchvision.utils.save_image(real_data, results_dir+'/'+filename+'_GT_'+'.png')
                #save_tiff_gdal(results_dir+'/'+filename, j.cpu())
                #save_tiff_gdal(results_dir+'/'+filename+'_GT',real_data.cpu())

            pq_psnr = piq.psnr(decoded, data, data_range=1.)
            pq_ssim = piq.ssim(decoded, data, data_range=1.)
            #pq_mssim = piq.multi_scale_ssim(decoded, data, kernel_size=3,  data_range=1.)

            total_psnr_value += pq_psnr
            total_ssim += pq_ssim
            #total_ms_ssim += pq_mssim


            for b in range(bands):
                pq_psnr = piq.psnr(decoded[:, None, b, :, :], data[:,None,b, :, :], data_range=1.)
                pq_ssim=piq.ssim(decoded[:,None,b, :, :], data[:,None,b, :, :], data_range=1.)
                pq_fsim=piq.fsim(decoded[:,None,b, :, :], data[:,None,b, :, :], data_range=1., chromatic=False)

                total_psnr_bands[b]+=pq_psnr
                total_ssim_bands[b]+=pq_ssim
                total_fsim_bands[b]+=pq_fsim


        avg_psnr_value = total_psnr_value / batches
        avg_ssim_value = total_ssim / batches
        #avg_ms_ssim_value = total_ms_ssim / batches


        avg_total_psnr_bands=[i/batches for i in total_psnr_bands]
        avg_total_ssim_bands=[i/batches for i in total_ssim_bands]
        avg_total_fsim_bands=[i/batches for i in total_fsim_bands]


        print(f"Valid stage: Epoch[{epoch + 1:04d}]")
        print(f"avg PSNR: {avg_psnr_value:.6f} avg SSIM: {avg_ssim_value:.6f}") # avg MS-SSIM: {avg_ms_ssim_value:.6f}")
        print("Bands evaluation")
        for b in range(bands):
            print(f"avg PSNR: {avg_total_psnr_bands[b]:.6f} avg SSIM: {avg_total_ssim_bands[b]:.6f}, avg FSIM: {avg_total_fsim_bands[b]:.6f}""")
        print("**************************")
        return avg_psnr_value.cpu().numpy(),avg_ssim_value.cpu().numpy()




criterion = torch.nn.BCELoss()
#cosine_criterion = torch.nn.CosineSimilarity(dim=2, eps=1e-8)
#pixelwise_loss = torch.nn.MSELoss()
encoder = CAE_model.Encoder(opt.latent_dim, opt.channels)
decoder = CAE_model.Decoder(opt.latent_dim, opt.channels)

# train_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/train/'
# valid_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/test/'
#train_dir = "/home/super/datasets-nas/dataset_rosetta_tiles/"+'train/'
#valid_dir =  "/home/super/datasets-nas/dataset_rosetta_tiles/"+'test/'
#train_dir = "/home/super/tmp_datasets/Imagenet-ILSVRC2012/train/"
#valid_dir =  "/home/super/tmp_datasets/kodak_test/"

results_dir = '/home/super/rlagrassa/inaf_compression_project/experiments/SSCNet_lastconv/results_slice_'+str(opt.latent_dim)#+str(dataset_slice)+'_'+str(opt.latent_dim)
save_path_model=opt.save_path_model

if os.path.exists(save_path_model):
    shutil.rmtree(save_path_model)
os.makedirs(save_path_model)

if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)

if opt.dataset_name == 'rosetta':
    train_dir = "/home/super/datasets-nas/dataset_rosetta_tiles/"+'train/'
    valid_dir =  "/home/super/datasets-nas/dataset_rosetta_tiles/"+'test/'
if opt.dataset_name == 'lombardia':
    dataset_slice = 100000
    train_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/train/'
    valid_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/test/'

if opt.dataset_name == 'rosetta' or opt.dataset_name == 'lombardia':
    train_dataset = BaseDataset_wth_folders(train_dir, "train" , opt.channels, opt.img_size, opt.dataset_name)
    valid_dataset = BaseDataset_wth_folders(valid_dir, "validate", opt.channels, opt.img_size, opt.dataset_name)
    train_dataloader = DataLoader(train_dataset, opt.batch_size, True, num_workers=8)
    valid_dataloader = DataLoader(valid_dataset, opt.batch_size, False, num_workers=8)



if opt.dataset_name == 'Imagenet-ILSVRC2012':
    train_dir = "/home/super/tmp_datasets/Imagenet-ILSVRC2012/train/"
    valid_dir =  "/home/super/tmp_datasets/kodak_test/"
    train_obj = MyDataset_plus_aug(train_dir, opt.channels, opt.img_size, opt.dataset_name)
    valid_obj = MyDataset_plus_aug(valid_dir, opt.channels, opt.img_size, opt.dataset_name)
    train_dataset=train_obj.train_dataset
    valid_dataset=valid_obj.test_dataset
    train_dataloader = DataLoader(train_dataset, opt.batch_size, True, num_workers=8)
    valid_dataloader = DataLoader(valid_dataset, opt.batch_size, False, num_workers=8)
    print("Full Dataset size: ", len(train_dataset), len(valid_dataset.samples))


encoder = nn.DataParallel(encoder).cuda()
decoder = nn.DataParallel(decoder).cuda()

optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr
)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer_G, opt.n_epochs)
Tensor = torch.cuda.FloatTensor

def multiple_plot(path):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        data, filename = next(iter(valid_dataloader))
        encoded = encoder(data)
        decoded = decoder(encoded)
        columns = 10
        decoded = decoded.permute(0, 2, 3, 1)
        decoded = decoded.cpu().data.numpy()
        for idx_cube, (img, s_filename) in enumerate(zip(decoded, filename)):
            if idx_cube > 20:
                break
            fig = plt.figure(figsize=(data.shape[2] / 2, data.shape[2] / 2))
            for band in range(decoded.shape[3]):
                plt.subplot(img.shape[2] / columns + 1, columns, band+1)
                plt.imshow(img[:, :, band], cmap='magma')
                plt.axis('off')
            fig.savefig(path+'/'+s_filename+'.png', dpi=120)
            plt.close(fig)

def sample_image(n_row, batches_done, path):
    data, _ = next(iter(valid_dataloader))
    encoded = encoder(data)
    decoded=decoder(encoded)
    gen_imgs = decoded[:,0:3,:,:]
    save_image(gen_imgs.data, path+'/best_decoded_grid.png', nrow=n_row, normalize=False)


def train(epoch):
    for i, (imgs, filename) in enumerate(train_dataloader):
        real_imgs = imgs.cuda()
        encoded  = encoder(real_imgs)
        decoded  = decoder(encoded)

        #c_loss =   0.08 * cosine_criterion(decoded.view(-1,9,48*48), real_imgs.view(-1,9,48*48)).abs().mean()
        g_loss =  criterion(decoded,real_imgs)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        if (i + 1) % 10 == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [bce loss: %f]"
                % (epoch, opt.n_epochs, i, len(train_dataloader), g_loss.item())
            )




encoder.train()
decoder.train()

start = time.time()


def save_model(avg_psnr_value, avg_ssim_value, epoch, avg_score_values):
    if avg_ssim_value > avg_score_values:
        print("Epoch: ", epoch," SSIM: ", avg_ssim_value, " PSNR: ",avg_psnr_value)
        torch.save(encoder.state_dict(), os.path.join(opt.save_path_model, "CAE_encoder_best.pth"))
        torch.save(decoder.state_dict(), os.path.join(opt.save_path_model, "CAE_decoder_best.pth"))
        avg_score_values = avg_ssim_value
        #sample_image(n_row=10, batches_done=epoch, path=results_dir)
        #multiple_plot(path=results_dir)
    return avg_score_values



avg_score_values = 0.0

for epoch in range(opt.n_epochs):
    train(epoch)
    if (epoch + 1) % 50  == 0:
        avg_psnr_value,avg_ssim_value=validate(encoder, decoder, valid_dataloader,results_dir,valid_dir,opt.channels)
        avg_score_values=save_model(avg_psnr_value, avg_ssim_value, epoch, avg_score_values)
        encoder.train()
        decoder.train()
    scheduler.step()
print("Training time: ", time.time() - start)
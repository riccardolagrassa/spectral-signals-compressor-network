import os

import piq
import rasterio
import shutil
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from dataset import BaseDataset_wth_folders
from obj_fun import SAD

torch.autograd.set_detect_anomaly(True)
from models import VAE_GAN, Discriminator



def save_tiff(path, data):
    data_unnormalize = data * torch.tensor([19017]).cuda()  #19017 for 1k and 5k
    data = data_unnormalize.cpu().numpy()
    #data= data.cpu().numpy()
    d=rasterio.open(path, 'w', driver='GTiff', crs='EPSG:32632', nodata=None, height=data.shape[1], width=data.shape[2], count=9, dtype=np.uint16)

    for i in range(9):
        d.write(data[i],i+1)



def validate() -> float:
    # Calculate how many iterations there are under epoch.
    batches = len(valid_dataloader)
    # Set generator model in verification mode.
    gen.eval()
    # Initialize the evaluation index.
    total_psnr_value, total_ssim,  = 0.0, 0.0
    total_psnr_bands=[0 for i in range(9)]
    total_ssim_bands=[0 for i in range(9)]

    with torch.no_grad():
        for index, (data, filename_packed) in enumerate(valid_dataloader):
            data = data.cuda()
            _,_,output = gen(data)

            for j, filename, real_data in zip(output, filename_packed,data):
                torchvision.utils.save_image(j, results_dir+'/'+filename+'.png')
                torchvision.utils.save_image(real_data, results_dir+'/'+filename+'_GT_'+'.png')

                #save_tiff(results_dir+'/'+filename, j)
                #save_tiff(results_dir+'/'+filename+'_GT', real_data)

            pq_psnr = piq.psnr(output, data, data_range=1.)
            pq_ssim = piq.ssim(output, data, data_range=1.)
            total_psnr_value += pq_psnr
            total_ssim += pq_ssim

            # for b in range(9):
            #     pq_psnr = piq.psnr(output[:, None, b, :, :], data[:, None,b, :, :], data_range=1.)
            #     pq_ssim=piq.ssim(output[:, None,b, :, :], data[:, None,b, :, :], data_range=1.)
            #     total_psnr_bands[b]+=pq_psnr
            #     total_ssim_bands[b]+=pq_ssim


        avg_psnr_value = total_psnr_value / batches
        avg_ssim_value = total_ssim / batches

        #avg_total_psnr_bands=[i/batches for i in total_psnr_bands]
        #avg_total_ssim_bands=[i/batches for i in total_ssim_bands]

        print(f"Valid stage: Epoch[{epoch + 1:04d}]")
        print(f"avg PSNR: {avg_psnr_value:.3f} avg SSIM: {avg_ssim_value:.3f}")
        #print("Bands evaluation")
        # for b in range(9):
        #     print(f"avg PSNR: {avg_total_psnr_bands[b]:.3f} avg SSIM: {avg_total_ssim_bands[b]:.3f}")
        # print("**************************")

    return avg_psnr_value, avg_ssim_value






num_epochs = 1000
batch_size = 256
learning_rate = 1e-4
dataset_slice=1000
lambda_loss= 1
adversarial_lambda=5e-3

# # standard compression ratio = 2 by standard implementation because of we have a maxpool before the latent code space
# compression_ratio = 2 # after 2 (maxpool) so 4:1 ratio
# output_aut= int((9*48*48)/compression_ratio)
torch.cuda.manual_seed_all(0)                       # Set random seed.
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if torch.cuda.is_available():
    cudnn.enabled = True
    cudnn.benchmark  = True                    # If the dimension or type of the input data of the network does not change much, turn it on, otherwise turn it off.
else:
    print("Error in GPU.")
    exit()

# train_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/train/'
# valid_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/test/'

train_dir = '/home/super/datasets-nas/mars_dataset/train/'
valid_dir = '/home/super/datasets-nas/mars_dataset/valid/'

results_dir = '/home/super/rlagrassa/inaf_compression_project/experiments/vae_gan/results_slice_'+str(dataset_slice)

if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)


train_dataset = BaseDataset_wth_folders(train_dir, "train")
valid_dataset = BaseDataset_wth_folders(valid_dir, "validate")

train_dataloader = DataLoader(train_dataset, batch_size, True)
valid_dataloader = DataLoader(valid_dataset, batch_size, False)


gen = VAE_GAN().cuda()
discrim = Discriminator().cuda()


pixel_loss = nn.L1Loss().cuda()
criterion = nn.BCEWithLogitsLoss().cuda()
optim_E = torch.optim.Adam(gen.encoder.parameters(), lr=learning_rate)
optim_D = torch.optim.Adam(gen.decoder.parameters(), lr=learning_rate)
optim_Dis = torch.optim.Adam(discrim.parameters(), lr=learning_rate)

gen.train()
discrim.train()

for epoch in range(num_epochs):
    prior_loss_list, gan_loss_list, recon_loss_list = [], [], []
    dis_real_list, dis_fake_list, dis_prior_list = [], [], []
    for i, (data, _) in enumerate(train_dataloader, 0):
        bs = data.size()[0]

        ones_label = torch.full([bs, 1], 1.0, dtype=data.dtype, requires_grad=False).cuda()
        zeros_label = torch.full([bs, 1], 0.0, dtype=data.dtype, requires_grad=False).cuda()
        #zeros_label1 = Variable(torch.zeros(64, 1)).cuda()
        datav = data.cuda()


        mean, logvar, rec_enc = gen(datav)

        #Decoder
        rec_loss = pixel_loss(rec_enc, datav)
        err_dec = rec_loss
        recon_loss_list.append(err_dec.item())
        optim_D.zero_grad()
        err_dec.backward(retain_graph=True)
        optim_D.step()

        # real_output = discrim(datav.detach())
        # reconstructed_output = discrim(rec_enc)
        # loss_GAN=criterion(reconstructed_output - real_output.mean(0, keepdim=True), ones_label)
        # gan_loss = adversarial_lambda * loss_GAN + err_dec
        # gan_loss_list.append(gan_loss.item())
        #
        # optim_Dis.zero_grad()
        # gan_loss.backward(retain_graph=True)
        # optim_Dis.step()
        #
        #
        # optim_Dis.zero_grad()
        #
        # real_output = discrim(datav)
        # reconstructed_output = discrim(rec_enc.detach())
        #
        # d_loss_hr = criterion(real_output - reconstructed_output.mean(0, keepdim=True), ones_label)
        # d_loss_sr = criterion(reconstructed_output - real_output.mean(0, keepdim=True), zeros_label)
        #
        # d_loss = (d_loss_hr + d_loss_sr) / 2
        # d_loss.backward(retain_graph=True)
        # optim_Dis.step()

        # output = discrim(datav)[0]
        # errD_real = criterion(output, ones_label)
        # output = discrim(rec_enc)[0]
        # errD_rec_enc = criterion(output, zeros_label)
        # output = discrim(x_p_tilda)[0]
        # errD_rec_noise = criterion(output, zeros_label1)
        # gan_loss = adversarial_lambda * (errD_real + errD_rec_enc + errD_rec_noise)

        #Decoder
        #x_l_tilda = discrim(rec_enc)[1]
        #x_l = discrim(datav)[1]
        #rec_loss = ((rec_enc - datav) ** 2).mean()


        #Encoder
        mean, logvar, rec_enc = gen(datav)
        x_l_tilda = discrim(rec_enc)[1]
        x_l = discrim(datav.detach())[1]
        rec_loss = ((x_l_tilda - x_l) ** 2).mean()
        prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
        prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data)
        prior_loss_list.append(prior_loss.item())
        err_enc = prior_loss + 5 * rec_loss
        optim_E.zero_grad()
        err_enc.backward(retain_graph=True)
        optim_E.step()

        if i % 1 == 0:
            #print('[%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_prior: %.4f\tRec_loss: %.4f\tdis_loss: %0.4f\t'
            print('[%d/%d][%d/%d]\tLoss pixel: %.4f'
                % (epoch, num_epochs, i, len(train_dataloader),err_dec.item()))#, gan_loss.item(), prior_loss.item(), err_dec.item(), d_loss.item()))

    if i % 1 == 0:
        avg_psnr_value, avg_ssim_valu=validate()
        gen.train()


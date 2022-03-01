import os
import time

import rasterio
import shutil

import numpy
import piq
from torch.backends import cudnn

from autoencoder import  Net
from dataset import BaseDataset_wth_folders
import torch
from torch import nn
from torch.utils.data import DataLoader


def save_tiff(path, data):
    data_unnormalize = data * torch.tensor([19017]).cuda()
    data = data_unnormalize.cpu().numpy()
    d=rasterio.open(path, 'w', driver='GTiff', crs='EPSG:32632', nodata=None, height=data.shape[1], width=data.shape[2], count=9, dtype=numpy.uint16)
    for i in range(9):
        d.write(data[i],i+1)



def validate(max_range) -> float:
    # Calculate how many iterations there are under epoch.
    batches = len(valid_dataloader)
    # Set generator model in verification mode.
    model.eval()
    # Initialize the evaluation index.
    total_psnr_value, total_ssim,  = 0.0, 0.0
    total_psnr_bands=[0 for i in range(9)]
    total_ssim_bands=[0 for i in range(9)]

    with torch.no_grad():
        for index, (data, filename_packed) in enumerate(valid_dataloader):
            data = data.cuda()
            output,_ = model(data)

            for j, filename in zip(output, filename_packed):
                save_tiff(results_dir+'/'+filename, j)

        #     pq_psnr = piq.psnr(output, data, data_range=1.)
        #     pq_ssim = piq.ssim(output, data, data_range=1.)
        #     total_psnr_value += pq_psnr
        #     total_ssim += pq_ssim
        #
        #     for b in range(9):
        #         pq_psnr = piq.psnr(output[:, None, b, :, :], data[:, None,b, :, :], data_range=1.)
        #         pq_ssim=piq.ssim(output[:, None,b, :, :], data[:, None,b, :, :], data_range=1.)
        #         total_psnr_bands[b]+=pq_psnr
        #         total_ssim_bands[b]+=pq_ssim
        #
        #
        # avg_psnr_value = total_psnr_value / batches
        # avg_ssim_value = total_ssim / batches
        #
        # avg_total_psnr_bands=[i/batches for i in total_psnr_bands]
        # avg_total_ssim_bands=[i/batches for i in total_ssim_bands]
        #
        # print(f"Valid stage: Epoch[{epoch + 1:04d}]")
        # print(f"avg PSNR: {avg_psnr_value:.3f} avg SSIM: {avg_ssim_value:.3f}")
        # print("Bands evaluation")
        # for b in range(9):
        #     print(f"avg PSNR: {avg_total_psnr_bands[b]:.3f} avg SSIM: {avg_total_ssim_bands[b]:.3f}")
        print("**************************")





def train():
    total_loss,max_range= 0, 0
    for data, _ in train_dataloader:
        img = data.cuda()
        output,_ = model(img)
        loss = criterion(output, img) * lambda_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss+=loss.data

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss))
    if (epoch + 1) % 1 == 0:
        validate(max_range)
        model.train()

num_epochs = 800
batch_size = 128
learning_rate = 1e-3
dataset_slice=1000
lambda_loss= 0.1

# standard compression ratio = 2 by standard implementation because of we have a maxpool before the latent code space
compression_ratio = 2 # after 2 (maxpool) so 4:1 ratio
output_aut= int((9*48*48)/compression_ratio)
torch.cuda.manual_seed_all(0)                       # Set random seed.
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if torch.cuda.is_available():
    cudnn.enabled = True
    cudnn.benchmark  = True                    # If the dimension or type of the input data of the network does not change much, turn it on, otherwise turn it off.
else:
    print("Error in GPU.")
    exit()

train_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/train/'
valid_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/test/'

results_dir = '/home/super/rlagrassa/inaf_compression_project/experiments/autoencoder_base/results_slice_'+str(dataset_slice)

if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)


train_dataset = BaseDataset_wth_folders(train_dir, "train")
valid_dataset = BaseDataset_wth_folders(valid_dir, "validate")

train_dataloader = DataLoader(train_dataset, batch_size, True)
valid_dataloader = DataLoader(valid_dataset, batch_size, False)



model = Net(output_aut).cuda()
criterion = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

total_loss=0

model.train()

start = time.time()
for epoch in range(num_epochs):
    train()
print("Training Time: ", time.time() - start)


#torch.save(model.state_dict(), './sim_autoencoder.pth')

import argparse
import os
import shutil
import time
import torch.nn as nn
import piq
import torch
import torchsummary as torchsummary
from osgeo import gdal
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np

import Ds_build
from dataset import BaseDataset_wth_folders
from dataset_load_imagenet import MyDataset_plus_aug


def save_tiff_gdal(path, data, ds):
    data = data * 21527
    data = data.cpu().numpy()
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path, ds.RasterXSize, ds.RasterYSize, ds.RasterCount, gdal.GDT_UInt16) #GDT_UInt16
    outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  ##sets same projection as input

    for i in range(data.shape[0]):
        outdata.GetRasterBand(i + 1).WriteArray(data[i, :, :])

    outdata.FlushCache()

def save_tiff_gdal_singleband(path, data,ds):
    data = data * 21527
    data = data.cpu().numpy()
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path, data.shape[0], data.shape[1], 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ds.GetGeoTransform())
    outdata.SetProjection(ds.GetProjection())
    outdata.GetRasterBand(1).WriteArray(data)
    outdata.FlushCache()


def sample_image(n_row, title, path, data):
    gen_imgs = data[:,0:3,:,:]
    grid = make_grid(gen_imgs.data, nrow=n_row, normalize=False)
    save_image(grid, path+'/grid_'+title+'.png', normalize=False)

def multiple_plot(path, data):
    plt.imshow(data, cmap='magma', vmin=0, vmax=np.max(data))
    plt.axis('off')
    plt.colorbar()
    plt.savefig(path+'.png', dpi=120)
    plt.close()

def perchannel_unnormalize(data,valid_dataset):
    data= (valid_dataset.max_channels_repeated - valid_dataset.min_channels_repeated) * data + valid_dataset.min_channels_repeated
    return data

def save_tiff_gdal_rosetta(path, data,valid_dataset):
    data=data.cpu()
    data=perchannel_unnormalize(data,valid_dataset)
    data=data.numpy()
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path, data.shape[1], data.shape[2], data.shape[0], gdal.GDT_Float32)
    #outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
    #outdata.SetProjection(ds.GetProjection())  ##sets same projection as input

    for i in range(432):
        outdata.GetRasterBand(i + 1).WriteArray(data[i, :, :])
    outdata.FlushCache()

def validate(encoder, decoder, valid_dataloader, results_dir, valid_dir, channels,valid_dataset) -> float:
    batches = len(valid_dataloader)
    encoder.eval()
    decoder.eval()

    total_psnr_value, total_ssim, total_ms_ssim  = 0.0, 0.0, 0.0
    # total_psnr_bands=[0 for i in range(channels)]
    # total_ssim_bands=[0 for i in range(channels)]
    # total_fsim_bands=[0 for i in range(channels)]

    with torch.no_grad():
        for index, (data, filename_packed) in enumerate(valid_dataloader):
            data = data.cuda()
            encoded = encoder(data)
            decoded = decoder(encoded)
            ##if index == 0:
            sample_image(n_row=6, title='decoded', path=results_dir, data=decoded)
            sample_image(n_row=6, title='source', path=results_dir, data=data)
            for j, filename, real_data in zip(decoded, filename_packed,data):
                save_image(j, results_dir + '/' + filename + '.png')
                save_image(real_data, results_dir + '/' + filename + '_GT_' + '.png')
                #ds=Ds_build.main(valid_dir+'/'+filename)
                ##utils.save_rgb(j.cpu().numpy(), results_dir+'/'+filename+'.png')
                ##torchvision.utils.save_image(j, results_dir+'/'+filename+'.png')
                ##torchvision.utils.save_image(real_data, results_dir+'/'+filename+'_GT_'+'.png')
                #dist = real_data[:, :, :] - j[:, :, :]
                # if filename == 'v1_00377420372.cal@0':
                #     for d_channels in range(0, real_data.shape[0]):
                #         print(filename," band@",d_channels," psnr: ", piq.psnr(j[None, None, d_channels, : , :], real_data[None, None, d_channels, : , :], data_range=1.))
                #         print(filename," band@",d_channels," ssim: ", piq.ssim(j[None, None, d_channels, : , :], real_data[None, None, d_channels, : , :], data_range=1.))
                        #multiple_plot(results_dir+'/'+filename + '_error_band@'+str(d_channels), dist[d_channels,:,:].cpu().numpy())
                        #save_tiff_gdal(results_dir + '/' + name + '_error_band@' + str(j), dist)
                        #save_tiff_gdal_singleband(results_dir+'/'+filename + '_error_band@'+str(d_channels), dist[d_channels, :, :], ds)
                        #save_tiff_gdal_rosetta(results_dir+'/'+filename+'@'+str(d_channels), j[d_channels,:,:],valid_dataset)
                        #save_tiff_gdal_rosetta(results_dir+'/'+filename+'RealDataNorm@'+str(d_channels), real_data[d_channels,:,:],valid_dataset)

                #save_tiff_gdal(results_dir+'/'+filename, j, ds)
                #save_tiff_gdal(results_dir+'/'+filename+'_GT',real_data, ds)


            pq_psnr = piq.psnr(decoded, data, data_range=1.)
            pq_ssim = piq.ssim(decoded, data, data_range=1.)
            pq_mssim = piq.multi_scale_ssim(decoded, data, kernel_size=3,  data_range=1.)


            total_psnr_value += pq_psnr
            total_ssim += pq_ssim
            total_ms_ssim += pq_mssim

            # for b in range(channels):
            #     pq_psnr = piq.psnr(decoded[:, None, b, :, :], data[:, None,b, :, :], data_range=1.)
            #     pq_ssim=piq.ssim(decoded[:, None,b, :, :], data[:, None,b, :, :], data_range=1.)
            #     pq_fsim=piq.fsim(decoded[:, None,b, :, :], data[:, None,b, :, :], data_range=1., chromatic=False)
            #
            #     total_psnr_bands[b]+=pq_psnr
            #     total_ssim_bands[b]+=pq_ssim
            #     total_fsim_bands[b]+=pq_fsim


        avg_psnr_value = total_psnr_value / batches
        avg_ssim_value = total_ssim / batches
        avg_ms_ssim_value = total_ms_ssim / batches

        # avg_total_psnr_bands=[i/batches for i in total_psnr_bands]
        # avg_total_ssim_bands=[i/batches for i in total_ssim_bands]
        # avg_total_fsim_bands=[i/batches for i in total_fsim_bands]


        # print(f"avg PSNR: {avg_psnr_value:.3f} avg SSIM: {avg_ssim_value:.3f}")
        print(f"avg PSNR: {avg_psnr_value:.6f} avg SSIM: {avg_ssim_value:.6f} avg MS-SSIM: {avg_ms_ssim_value:.6f}")
        # for b in range(channels):
        #      print(f"Band {b}   avg PSNR: {avg_total_psnr_bands[b]:.3f} avg SSIM: {avg_total_ssim_bands[b]:.3f} avg FSIM: {avg_total_fsim_bands[b]:.3f}""")
        # print("**************************")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=48, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--dataset_slice", type=int, default=100000, help="dataset dimension")
    parser.add_argument("--model_path_encoder", type=str, default='None', help="path pretrained")
    parser.add_argument("--model_path_decoder", type=str, default='None', help="path pretrained")
    parser.add_argument("--dataset", type=str, default='lombardia', help="dataset")
    parser.add_argument("--last_layer", type=str, default='linear', help="last layer mode")
    parser.add_argument("--latent_space", type=int, default=13824, help="latent space")

    torch.cuda.manual_seed_all(0)  # Set random seed.
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    if torch.cuda.is_available():
        cudnn.enabled = True
        cudnn.benchmark = True
    else:
        print("Error in GPU.")
        exit()

    opt = parser.parse_args()
    print(opt)
    if opt.dataset == 'lombardia':
        valid_dir = '/home/super/datasets-nas/lombardia_original_slice_' + str(opt.dataset_slice) + '/test/'
        results_dir = '/home/super/rlagrassa/inaf_compression_project/experiments/CAE_PReLU/results_slice_'+str(opt.dataset_slice)+'_'+str(opt.latent_space)
        valid_dataset = BaseDataset_wth_folders(valid_dir, "validate",opt.channels, opt.img_size, opt.dataset)
        valid_dataloader = DataLoader(valid_dataset, opt.batch_size, False, num_workers=8)

    elif opt.dataset == 'rosetta':
        valid_dir = "/home/super/datasets-nas/dataset_rosetta_tiles/" + 'test/'
        results_dir = '/home/super/rlagrassa/inaf_compression_project/experiments/CAE_LastLinearLayer_rosetta/'
        valid_dataset = BaseDataset_wth_folders(valid_dir, "validate", opt.channels, opt.img_size, opt.dataset)
        valid_dataloader = DataLoader(valid_dataset, opt.batch_size, False, num_workers=8)

    elif opt.dataset == 'Imagenet-ILSVRC2012':
        valid_dir = "/home/super/tmp_datasets/kodak_test/"
        results_dir = '/home/super/rlagrassa/inaf_compression_project/experiments/CAE_kodak/'+str(opt.latent_space)
        valid_obj = MyDataset_plus_aug(valid_dir, opt.channels, opt.img_size, opt.dataset)
        valid_dataset = valid_obj.test_dataset
        valid_dataloader = DataLoader(valid_dataset, opt.batch_size, False, num_workers=8)
        print("Test set size: ", len(valid_dataset.samples))


    if os.path.exists(results_dir):
       shutil.rmtree(results_dir)
    os.makedirs(results_dir)



    if opt.dataset == 'lombardia':
        from SSCNet_lomb import Encoder, Decoder
        encoder = Encoder(opt.latent_space,opt.channels)
        decoder = Decoder(opt.latent_space,opt.channels)

    elif opt.dataset == 'rosetta':
        if opt.last_layer == 'conv':
            from SSCNet_lastconv import Decoder, Encoder
            encoder = Encoder(opt.channels)
            decoder = Decoder(opt.channels)
        elif opt.last_layer == 'linear':
            from CAE_model import Decoder, Encoder
            encoder = Encoder(opt.latent_space, opt.channels)
            decoder = Decoder(opt.latent_space, opt.channels)

    elif opt.dataset == 'Imagenet-ILSVRC2012':
        from CAE_model import Decoder, Encoder
        encoder = Encoder(opt.latent_space, opt.channels)
        decoder = Decoder(opt.latent_space, opt.channels)

    encoder = nn.DataParallel(encoder).cuda()
    decoder = nn.DataParallel(decoder).cuda()

    # encoder_parameters = count_parameters(encoder)
    # decoder_parameters = count_parameters(decoder)
    # print("Encoder parameters: ", encoder_parameters)
    # print("Decoder parameters: ", decoder_parameters)
    # print(torchsummary.summary(encoder, (423, 64, 64)))

    state_dict_encoder = torch.load(opt.model_path_encoder)
    encoder.load_state_dict(state_dict_encoder)
    state_dict_decoder = torch.load(opt.model_path_decoder)
    decoder.load_state_dict(state_dict_decoder)

    average_multiple_results = True

    # encoder.half()
    # decoder.half()
        # Set generator model in verification mode.
    start=time.time()
    if average_multiple_results:
        for j in range(0,20):
            validate(encoder, decoder, valid_dataloader,results_dir,valid_dir,opt.channels, valid_dataset)
            valid_obj = MyDataset_plus_aug(valid_dir, opt.channels, opt.img_size, opt.dataset)
            valid_dataset = valid_obj.test_dataset
            valid_dataloader = DataLoader(valid_dataset, opt.batch_size, False, num_workers=8)
    else:
        validate(encoder, decoder, valid_dataloader, results_dir, valid_dir, opt.channels, valid_dataset)
    print("Test time: ", time.time() - start)





if __name__ == "__main__":
    main()

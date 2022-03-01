import shutil
import matplotlib.pyplot as plt
from osgeo import gdal
import os
import torch
import numpy as np
from osgeo import gdal
from VIRTISpy import *


def get_data(mode_sampler, save_path):
    for name in mode_sampler:
        images_name = name.split('/')[-1]
        # folder_name = name.split('/')[-2]
        shutil.copyfile(name, save_path + images_name)


def splittation(data_list):
    # Train val splitting
    validation_split = 0.2
    dataset_size = len(data_list)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(1)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = [data_list[i] for i in train_indices]
    valid_sampler = [data_list[i] for i in val_indices]

    print(len(train_sampler), len(valid_sampler), len(data_list))

    # save train data
    get_data(train_sampler, path_save_file_train)
    # save test data
    get_data(valid_sampler, path_save_file_test)


def patches_extraction(x):
    if x.shape[0] > 3:
        kc, kh, kw = x.shape[0], stride, stride
        dc, dh, dw = x.shape[0], stride, stride  # stride shifting kernel
        patches = x.unfold(0, kc, dc).unfold(1, kh, dh).unfold(2, kw, dw)
        patches = patches.contiguous().view(-1,  kc, kh, kw)
        return patches.numpy()
    # elif x.shape[1] == 1:
    #     kc, kh, kw = 1, 512, 512
    #     dc, dh, dw = 1, stride, stride  # stride
    #     patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    #     patches = patches.contiguous().view(patches.size(0), -1, kc, kh, kw)
    #     return patches|
    else:
        print(x.shape)
        print("Error on channel size")


def save_tiff_gdal(path, data, namefile, data_list):
    driver = gdal.GetDriverByName("GTiff")
    #outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
    #outdata.SetProjection(ds.GetProjection())  ##sets same projection as input
    for idx, tile in enumerate(data):
        outdata = driver.Create(path+namefile+'@'+str(idx), tile.shape[1], tile.shape[2], tile.shape[0], gdal.GDT_Float32)
        data_list.append(path+namefile+'@'+str(idx))
        for i in range(tile.shape[0]):
            outdata.GetRasterBand(i + 1).WriteArray(tile[i, :, :])
        outdata.FlushCache()

def multiple_plot(img_filename):
    fig = plt.figure(figsize=(img.shape[0]/2,img.shape[1]/2))
    columns = 10
    for band in range(img.shape[2]):
        plt.subplot(img.shape[2] / columns + 1, columns, band + 1)
        plt.imshow(img[:, :, band], cmap='gray')
    fig.savefig('/home/super/datasets-nas/rosetta_png_samples/'+img_filename+'.png', dpi=100)
    #plt.show()

stride=64
plt.rcParams.update({'figure.max_open_warning': 0})
data_list=[]
path_source ="/home/super/datasets-nas/rosetta/"

path_save_file_train = "/home/super/datasets-nas/dataset_rosetta_tiles/"+'/train/'
path_save_file_test =  "/home/super/datasets-nas/dataset_rosetta_tiles/"+'/test/'
if os.path.exists(path_save_file_train):
    shutil.rmtree(path_save_file_train)
os.makedirs(path_save_file_train)
if os.path.exists(path_save_file_test):
    shutil.rmtree(path_save_file_test)
os.makedirs(path_save_file_test)


path_destination_tiles = "/home/super/datasets-nas/all_rosetta_tiles/"
if os.path.exists(path_destination_tiles):
    shutil.rmtree(path_destination_tiles)
os.makedirs(path_destination_tiles)

sub_folders=os.listdir(path_source)

gl_min, gl_max = 0, 0
min_channel_list = [0 for i in range(432)]
max_channel_list = [0 for i in range(432)]

for idx, mtpX in enumerate(sub_folders):
    stp_list = os.listdir(path_source+mtpX+'/data/')
    for stp in stp_list:
        all_cubes=os.listdir(path_source+mtpX+'/data/'+stp+'/cal/virtis_m_vis/')
        for cube in all_cubes:
            cb=VIRTISpy(path_source+mtpX+'/data/'+stp+'/cal/virtis_m_vis/'+cube)
            img=cb.getCube()
            #img=cb.getBand(3)
            #img=where(img<=0,0,img)
            #img = np.moveaxis(img, 0, 2)
            #multiple_plot(path_img)
            if img.shape[1] >= stride and img.shape[2] >= stride:
                #frac0,_=np.modf(img.shape[1] / stride) #to check how null information we obtain after the tiles creations
                #frac1,_=np.modf(img.shape[2] / stride)
                #img = ((img - np.min(img)) / (np.max(img) - np.min(img)))  # * 255

                #Local search
                for idx, j in enumerate(img):
                    tmp_min, tmp_max = np.min(j), np.max(j)
                    if tmp_min < min_channel_list[idx]:
                        min_channel_list[idx] = tmp_min
                    if tmp_max > max_channel_list[idx]:
                        max_channel_list[idx] = tmp_max

                #Global search
                tmp_min, tmp_max = np.min(img), np.max(img)
                if tmp_min < gl_min:
                    gl_min = tmp_min
                if tmp_max > gl_max:
                    gl_max = tmp_max
                tiles = patches_extraction(torch.tensor(img))
                s_cube_filename = path_source+mtpX+'/data/'+stp+'/cal/virtis_m_vis/'+cube
                print(s_cube_filename, " Shape: ",tiles.shape)
                save_tiff_gdal(path_destination_tiles,tiles, cube, data_list)
            else:
                break

print("Min: ", gl_min, "Max: ", gl_max)
print("Min per band channels: ", min_channel_list)
print("Max per band channels: ", max_channel_list)
splittation(data_list)
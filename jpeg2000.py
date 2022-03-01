import os
import shutil
import sys
from io import BytesIO
import cv2
import glymur as glymur
import numpy
from PIL import Image
from osgeo import gdal
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import Ds_build
import matplotlib.pyplot as plt


def save_tiff_gdal(path, data):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path, data.shape[0], data.shape[1], 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ds.GetGeoTransform())
    outdata.SetProjection(ds.GetProjection())
    outdata.GetRasterBand(1).WriteArray(data)
    outdata.FlushCache()

def multiple_plot(path, data):
    plt.imshow(data, cmap='magma', vmin=0, vmax=np.max(data))
    plt.axis('off')
    plt.colorbar()
    plt.savefig(path+'.png', dpi=120)
    plt.close()

def jpeg2000_f(path, x):
    example_array = x.ReadAsArray()
    x = ((example_array + 305.545166015625) / (1279.68310546875 + 305.545166015625))
    x = numpy.round(x * (2^32-1)).astype(numpy.int32)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path+'.tif', x.shape[1], x.shape[2], 432, gdal.GDT_Float32)

    for i in range(432):
        outdata.GetRasterBand(i + 1).WriteArray(x[i, :, :])
    outdata.FlushCache()

    x = gdal.Open(path+'.tif')
    jpeg_out=gdal.GetDriverByName('JP2OpenJPEG').CreateCopy(path+'.jp2', x,options=['QUALITY=100'])
    print(jpeg_out.shape)
    exit()

def open_tiff(path_image):
    x = gdal.Open(path_image)
    if dataset_name == 'lombardia':
        example_array = x.GetRasterBand(3).ReadAsArray()
        example_array = numpy.dstack((example_array, x.GetRasterBand(2).ReadAsArray()))
        example_array = numpy.dstack((example_array, x.GetRasterBand(1).ReadAsArray()))
        for i in range(4, 10):
            example_array = numpy.dstack((example_array,
                                          x.GetRasterBand(i).ReadAsArray()))
        return numpy.array(example_array).astype('uint16')
    elif dataset_name == 'rosetta':
        example_array = x.ReadAsArray()
        example_array = example_array.transpose(1,2,0)
    # x = ((example_array + 305.545166015625) / (1279.68310546875 + 305.545166015625))
    # x = numpy.round(x * (2^32-1)).astype(numpy.int32)
        return example_array

dataset_slice = 100000
dataset_name = 'lombardia'
if dataset_name == 'lombardia': valid_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/test/'
elif dataset_name == 'rosetta': valid_dir =  "/home/super/datasets-nas/dataset_rosetta_tiles/"+'test/'

results_dir = '/home/super/rlagrassa/inaf_compression_project/experiments/jpeg_compression10_singlebands/'+dataset_name

if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)

glymur.set_option('lib.num_threads', 8)

total_psnr_bands=[0 for i in range(9)]
total_ssim_bands=[0 for i in range(9)]
final_psnr, final_ssim = 0, 0
for index, img in enumerate(os.listdir(valid_dir)):
    # if index == 100:
    #     break
    avg_ssim, avg_psnr = 0, 0
    x = open_tiff(valid_dir+img)
    name=str(img).split('.')[0]

    # c=glymur.Jp2k(results_dir+'/'+name+'_compressed_band.jp2', data=x, cratios=[40])
    # a=glymur.Jp2k(results_dir+'/'+name+'_band.jp2', data=x, cratios=[1])
    #
    # compressed_size = os.path.getsize(results_dir+'/'+name+'_compressed_band.jp2')
    # source_size= os.path.getsize(results_dir+'/'+name+'_band.jp2')
    #
    # print("Last sample ratio compression bands: ", source_size / compressed_size)

    for j in range(0, 9):
        ds = Ds_build.main(valid_dir + img)
        #x_pil = Image.fromarray(x[:, :, j])#.convert("F")
        #x_pil.save(results_dir + '/' + name + '_compressed_band@'+str(j)+'.jp2', 'JPEG2000', quality_mode='db', quality=[100])
        #dec = Image.open(results_dir + '/' + name + '_compressed_band@'+str(j)+'.jp2')
        c = glymur.Jp2k(results_dir + '/' + name + '_compressed_band@'+str(j)+'.jp2', data=x[:,:,j], cratios=[10])
        dec = numpy.asfarray(c.read())
        compressed_size = os.path.getsize(results_dir + '/' + name + '_compressed_band@'+str(j)+'.jp2')
        #source_size = os.path.getsize(results_dir + '/' + name + '@'+str(j)+'_band.jp2')
        source_size = x[:,:,j].nbytes
        print("Sample Ratio Compression Bands@ ", j, " ", source_size / compressed_size)
        ssim_value = ssim(x[:, :, j], dec, data_range=np.max(x[:,:,j]))
        psnr_value = psnr(x[:, :, j], dec, data_range=np.max(x[:,:,j]))
        #dist=x[:, :, j] - dec
        print("Image ", name, " band: ", j, " psnr: ", psnr_value, " ssim: ", ssim_value)
        #multiple_plot(results_dir + '/' + name + '_error_band@'+str(j), dist)
        #save_tiff_gdal(results_dir + '/' + name + '_error_band@'+str(j), dist)
        total_psnr_bands[j] += psnr_value
        total_ssim_bands[j] += ssim_value
        avg_psnr+=psnr_value
        avg_ssim+=ssim_value

    final_psnr += avg_psnr / x.shape[2]
    final_ssim += avg_ssim / x.shape[2]

final_psnr/=len(os.listdir(valid_dir))
final_ssim/=len(os.listdir(valid_dir))


avg_total_psnr_bands=[i/len(os.listdir(valid_dir)) for i in total_psnr_bands]
avg_total_ssim_bands=[i/len(os.listdir(valid_dir)) for i in total_ssim_bands]

print(f"avg PSNR: {final_psnr:.3f} avg SSIM: {final_ssim:.3f}")
for b in range(9):
    print(f"Band {b}   avg PSNR: {avg_total_psnr_bands[b]:.3f} avg SSIM: {avg_total_ssim_bands[b]:.3f}""")
print("**************************")
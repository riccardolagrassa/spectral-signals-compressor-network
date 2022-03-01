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


def open_tiff(path_image):
    x = gdal.Open(path_image)
    example_array = x.GetRasterBand(3).ReadAsArray()
    example_array = numpy.dstack((example_array, x.GetRasterBand(2).ReadAsArray()))
    example_array = numpy.dstack((example_array, x.GetRasterBand(1).ReadAsArray()))
    for i in range(4, 10):
        example_array = numpy.dstack((example_array,
                                      x.GetRasterBand(i).ReadAsArray()))
    #x = example_array / 21527
    #x = numpy.round(x * 255).astype(numpy.uint8)
    #x = ((x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)))
    return numpy.array(example_array).astype('uint8')

dataset_slice = 100000
valid_dir = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/test/'

results_dir = '/home/super/rlagrassa/inaf_compression_project/experiments/jpeg_compression'


if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)

total_psnr_bands=[0 for i in range(9)]
total_ssim_bands=[0 for i in range(9)]
final_psnr, final_ssim = 0, 0
for index, img in enumerate(os.listdir(valid_dir)):
    avg_ssim, avg_psnr = 0, 0
    x = open_tiff(valid_dir+img)
    name=str(img).split('.')[0]
    for j in range(0, x.shape[2]):
        a=x[:, :, j]
        b=x[:,:,j]

        cv2.imwrite(results_dir+'/'+name+'_compressed_band'+str(j)+'.jpg', a, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        cv2.imwrite(results_dir+'/'+name+'_band'+str(j)+'.jpg', b)

        compressed_size = os.path.getsize(results_dir+'/'+name+'_compressed_band'+str(j)+'.jpg')
        source_size= os.path.getsize(results_dir+'/'+name+'_band'+str(j)+'.jpg')

        #if index == len(os.listdir(valid_dir)) - 1:
        print("Last Sample Ratio Compression bands: ", j+1, " ", source_size / compressed_size)

        original_image = cv2.imread(results_dir+'/'+name+'_band'+str(j)+'.jpg', cv2.IMREAD_GRAYSCALE)
        compressed_image = cv2.imread(results_dir+'/'+name+'_compressed_band'+str(j)+'.jpg', cv2.IMREAD_GRAYSCALE)
        ssim_value = ssim(numpy.asfarray(original_image[:, :]), numpy.asfarray(compressed_image[:, :]), data_range=numpy.max(b[:, :]))
        psnr_value = psnr(numpy.asfarray(original_image[:, :]), numpy.asfarray(compressed_image[:, :]), data_range=numpy.max(b[:, :]))
        total_psnr_bands[j] += psnr_value
        total_ssim_bands[j] += ssim_value
        avg_psnr+=psnr_value
        avg_ssim+=ssim_value

    final_psnr += avg_psnr / x.shape[2]
    final_ssim += avg_ssim / x.shape[2]

final_psnr /= len(os.listdir(valid_dir))
final_ssim /= len(os.listdir(valid_dir))


avg_total_psnr_bands=[i/len(os.listdir(valid_dir)) for i in total_psnr_bands]
avg_total_ssim_bands=[i/len(os.listdir(valid_dir)) for i in total_ssim_bands]

print(f"avg PSNR: {final_psnr:.3f} avg SSIM: {final_ssim:.3f}")
for b in range(9):
    print(f"Band {b}   avg PSNR: {avg_total_psnr_bands[b]:.3f} avg SSIM: {avg_total_ssim_bands[b]:.3f}""")
print("**************************")
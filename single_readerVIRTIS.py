import shutil
import matplotlib.pyplot as plt
from osgeo import gdal
import os
import torch
import numpy as np
from osgeo import gdal
from VIRTISpy import *

def multiple_plot(img_filename):
    #fig = plt.figure(figsize=(img.shape[0],img.shape[1]))
    #columns = 10
    # for band in range(img.shape[2]):
    #     plt.subplot(img.shape[2] / columns + 1, columns, band + 1)
    #     plt.imshow(img[:, :, band], cmap='gray')
    plt.axis('off')
    for j in range(img.shape[2]):
        plt.imshow(img[:, :, j], cmap='gray')
        plt.savefig('/home/super/datasets-nas/rosetta_png_samples/'+img_filename+str(j)+'.png', dpi=130)


path='/home/super/datasets-nas/rosetta/ro-c-virtis-3-esc1-mtp010-v2.0/data/stp034/cal/virtis_m_vis/v1_00377420372.cal'
cb=VIRTISpy(path)
img=cb.getCube()
img = np.moveaxis(img, 0, 2)
multiple_plot(path.split('/')[-1])
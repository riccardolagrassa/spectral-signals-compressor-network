import imageio
import spectral
import numpy as np

def save_rgb(img, save_path):
    img = img.squeeze()
    #img = img.detach().numpy()
    img = np.moveaxis(img, 0, -1)
    rgb = spectral.get_rgb(img, (0, 1, 2))
    #rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')
    imageio.imsave(save_path, rgb)

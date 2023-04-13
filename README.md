# Spectral Signals Compressor network
<img src="./sscnet.png" width="60%">

## Abstract
In Space Science and satellite imagery, better resolution of the data information obtained makes images clearer and interpretation more accurate, however, the huge data volume acquired by the complex onboard satellite instruments become a problem that needs to be managed carefully. To reduce the data volume to be stored and transmitted on-ground the signals received should be compressed allowing a good original source representation in the reconstruction step.
Image compression covers a key role in Space Science and satellite imagery and recently, Deep Learning models have achieved remarkable results in computer vision.
In this paper, we propose a Spectral Signals Compressor network based on Deep Convolutional AutoEncoder (SSCNet) and, we conduct a wide range of experiments over multi/hyper-spectrals and RGB datasets reporting improvements over all baselines used as benchmarker and than the JPEG family algorithm.
Experimental results demonstrate the effectiveness in the compression ratio and spectral signals reconstruction and the robustness with a data type greater than 8 bits, clearly exhibiting better results using the PSNR, SSIM, MS-SSIM evaluations criterion.

# Pretrained models link
https://drive.google.com/drive/folders/1js1e1EtrePl2jYq5HV-AHcdQiklF0fK3?usp=sharing
## Colab (for inference)
https://colab.research.google.com/drive/1pH7DX-CMLdhEneZTFULnzQxQScHtbDTj?usp=sharing


## License

The project is licensed under the GPL-3.0 license 


## BibTeX Citation

If you use the Spectral Signals Compressor network in a scientific publication, we would appreciate using the following citation:

```


@article{la2022hyperspectral,
  title={Hyperspectral Data Compression Using Fully Convolutional Autoencoder},
  author={La Grassa, Riccardo and Re, Cristina and Cremonese, Gabriele and Gallo, Ignazio},
  journal={Remote Sensing},
  volume={14},
  number={10},
  pages={2472},
  year={2022},
  publisher={MDPI}
}

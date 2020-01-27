# Recycle-GAN :Unsupervised Video Retargeting 

This repository provides the code for our work on [unsupervised video retargeting](http://www.cs.cmu.edu/~aayushb/Recycle-GAN/). 

```make
@inproceedings{Recycle-GAN,
  author    = {Aayush Bansal and
               Shugao Ma and
               Deva Ramanan and
               Yaser Sheikh},
  title     = {Recycle-GAN: Unsupervised Video Retargeting},
  booktitle   = {ECCV},
  year      = {2018},
}
```

Acknowledgements: This code borrows heavily from the PyTorch implementation of [Cycle-GAN and Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). A  huge thanks to them!


### John Oliver to Stephen Colbert
[![John Oliver to Stephen Colbert](https://img.youtube.com/vi/VWXFqDdqLbE/0.jpg)](https://www.youtube.com/watch?v=VWXFqDdqLbE)

Click above to see video!


### Video by CMU Media folks
[![Video by CMU folks!](https://img.youtube.com/vi/ehD3C60i6lw/0.jpg)](https://youtu.be/ehD3C60i6lw)

Click above to see video!

## Introduction

We use this formulation in our ECCV'18 paper on unsupervised video retargeting for various domains where space and time information matters such as face retargeting. Without any manual annotation, our approach could learn retargeting from one domain to another.

## Using the Code

The repository contains the code for training a network for retargeting from one domain to another, and use a trained module for this task. Following are the things to consider with this code:

### Requirements
* Linux or MacOS
* Python 3
* Pytorch 0.4
* NVIDIA GPU + CUDA CuDNN

### Python Dependencies
* numpy 1.15.0
* torch 0.4.1.post2
* torchvision 0.2.2
* visdom
* dominate

Run the following command to install automatically:
```pip install requirements.txt```

### Data pre-processing

For each task, create a new folder in "dataset/" directory. The images from two domains are placed respectively in "trainA/" and "trainB/". Each image file consists of horizontally concatenated images, "{t, t+1, t+2}" frames from the video. The test images are placed in "testA/" and "testB/". Since we do not use temporal information at test time, the test data consists of single image "{t}".

## Training

There are two training modules in "scripts/" directory: (1). Recycle-GAN, (2). ReCycle-GAN

### Recycle-GAN

Recycle-GAN is the model described in the paper and is used for most examples in the paper, specifically face to face, flower to flower, clouds and wind synthesis, sunrise and sunset.

### ReCycle-GAN

ReCycle-GAN is mostly similar to Recycle-GAN. Additionally, we also use vanilla cycle-losses from CycleGAN between corresponding source and target frames. We found this module useful for tasks such as unpaired image to labels, and labels to image on VIPER dataset, image to normals, and normals to image on NYU-v2 depth dataset.

## Prediction Model

There are two prediction model used in this work: (1). simple U-Net, (2). higher-capacity prediction.

### unet-128, unet-256

If you want to use this prediction module, please set the flag "--which_model_netP" to "unet_128" and "unet_256" respectively.

### prediction

An advanced version of prediction module is a higher capacity module by setting the flag "--which_model_netP" to "prediction".

## Observation about training: 

We observed that model converges in 20-40 epochs when sufficiently large data is used. For smaller datasets (ones having 1000 images or less), it is suitable to let it train for longer.

## Test

At test time, we do inference per image (as mentioned previously). The test code is based on cycle-gan.


## Data & Trained Models:

Please use following links to download Face, Flowers, and Viper data: 

1. [Faces (10 GB)](https://www.dropbox.com/s/s6kzovbrevin5tr/faces.tar.gz?dl=0) 
2. [Flowers (1.6 GB)](https://www.dropbox.com/s/f8nnulkcsovxwmf/flowers.tar.gz?dl=0)
3. [Viper (3.17 GB)](https://www.dropbox.com/s/qhu29y5cx4lyfun/Viper_data.tar.gz?dl=0)


Please contact [Aayush Bansal](http://cs.cmu.edu/~aayushb) for any specific data or trained models, or for any other information. 

import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from functools import partial
import sys
from torchmetrics.image import PeakSignalNoiseRatio as psnr_layer
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as lpips_layer
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim_layer

sys.path.append("..")

from utils import clip_img



def psnr(im1, im2, sc_type="neg1to1"):
    
    with torch.no_grad():
        im1,im2 = im1.detach(),im2.detach()
        ranges = {"neg1to1":2,"zeroto1":1}
        met_layer = psnr_layer(data_range=ranges[sc_type]).to(im1.device)
    return met_layer(im1,im2)    
        
def ssim(im1, im2, sc_type="neg1to1"):
    
    with torch.no_grad():
        im1,im2 = im1.detach(),im2.detach()
        ranges = {"neg1to1":2,"zeroto1":1}
        met_layer = ssim_layer(data_range=ranges[sc_type]).to(im1.device)
    return met_layer(im1,im2)  

def lpips(im1, im2, sc_type="neg1to1"):
    
    with torch.no_grad():
        im1,im2 = im1.detach(),im2.detach()
        par = {"neg1to1":False,"zeroto1":True}
        met_layer = lpips_layer(normalize=par[sc_type]).to(im1.device)
    return met_layer(im1,im2)  



def batch_PSNR(img, imclean, scaling=True, sc_type="neg1to1", int_dtype=np.uint16):
    with torch.no_grad():

        _clip_img = partial(
            clip_img, scaling=scaling, sc_type=sc_type, int_dtype=int_dtype
        )

        img, imclean = map(_clip_img, [img, imclean])

        Img = img.cpu().detach().numpy()
        Iclean = imclean.cpu().detach().numpy()

        if scaling:

            if sc_type == "neg1to1":
                data_range = 2
            elif sc_type == "zeroto1":
                data_range = 1

        else:
            data_range = np.iinfo(int_dtype).max

        psnr = 0

        for i in range(Img.shape[0]):
            # N x C x H x W
            # H x W x C
            psnr += peak_signal_noise_ratio(
                Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range
            )
    return psnr / Img.shape[0]


def batch_SSIM(img, imclean, scaling=True, sc_type="neg1to1", int_dtype=np.uint16):
    with torch.no_grad():

        _clip_img = partial(
            clip_img, scaling=scaling, sc_type=sc_type, int_dtype=int_dtype
        )

        img, imclean = map(_clip_img, [img, imclean])

        Img = img.cpu().detach().numpy()
        Iclean = imclean.cpu().detach().numpy()

        if scaling:

            if sc_type == "neg1to1":
                data_range = 2
            elif sc_type == "zeroto1":
                data_range = 1

        else:
            data_range = np.iinfo(int_dtype).max

        ssim = 0

        for i in range(Img.shape[0]):
            # treat last dim as channel, hence transpose!
            ssim += structural_similarity(
                np.transpose(Iclean[i, :, :, :], (1, 2, 0)),
                np.transpose(Img[i, :, :, :], (1, 2, 0)),
                data_range=data_range,
                channel_axis=-1,
            )
    return ssim / Img.shape[0]


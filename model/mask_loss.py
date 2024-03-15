from pytorch_wavelets import DWTForward
from torch_topological.nn import (
    SummaryStatisticLoss,
    MultiScaleKernel,
    WassersteinDistance,
    SlicedWassersteinDistance,
    SignatureLoss,
)
from functools import partial
import pandas as pd
import torch
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  
from torch.nn import Upsample



def img_to_wv_bands(img,wave="haar",level=0):
    
    dwt = DWTForward(wave=wave,J=level).to(img.device)
    img_w=  dwt(img)
    img_wa = img_w[0]
    img_wv = img_w[1][level-1][:, :, 0, ...]  # [BS, C, H, W]
    img_wh = img_w[1][level-1][:, :, 1, ...]
    img_wd = img_w[1][level-1][:, :, 2, ...]
    return {"LL":img_wa,"HV":img_wv,"HH":img_wh,"HD":img_wd}

def mmnorm(v): 
    return (v - v.min())/(v.max() - v.min())


    
def mean_hbands(img,wave="haar",level=2):
    wv = img_to_wv_bands(img,wave=wave,level=level)
    meanb = torch.mean(torch.stack([torch.abs(wv["HV"]),torch.abs(wv["HH"]),torch.abs(wv["HD"])]),dim=0)
    return wv["LL"], meanb

def hh_bands(img, wave="haar", level=1):
    dwt = DWTForward(wave=wave, J=level).to(img.device)
    img_w = dwt(img)
    img_wa = img_w[0]
    img_wv = img_w[1][level - 1][:, :, 0, ...]  # [BS, C, H, W]
    img_wh = img_w[1][level - 1][:, :, 1, ...]
    img_wd = img_w[1][level - 1][:, :, 2, ...]
    return torch.stack([img_wv, img_wh, img_wd])


def calc_mask(img, level=1, wave="haar", reduce_mask=True):
    llimgn, _ = mean_hbands(img, wave=wave, level=level)

    upscale_transn = Upsample(
        scale_factor=2 ** (level + 1), mode="bicubic", align_corners=False
    )
    up_llimgn = upscale_transn(llimgn)

    if reduce_mask:
        _,h1lln = mean_hbands(up_llimgn, wave=wave, level=1)
    else:
        h1lln = hh_bands(up_llimgn, wave=wave, level=1)

    mask = mmnorm(torch.abs(h1lln))

    return mask
        
import torch.nn.functional as F
import torch
from typing import Optional
from .pd_loss import topo_loss
from .mask_loss import calc_mask#, calc_mask_hh

from .vggloss import VGGLoss


from torch.cuda.amp import GradScaler, autocast

from torchmetrics.functional import peak_signal_noise_ratio,structural_similarity_index_measure

def normto01(t,tmin=-1,tmax=1):
    return (t-tmin)/(tmax-tmin)

def mse_loss(data, target, noise=None,reduction="mean"):

    
    if noise is not None:
        return F.mse_loss(noise,data-target,reduction=reduction)
    else:
        return F.mse_loss(data, target,reduction=reduction)

def l1_loss(data, target, noise=None,reduction="mean"):
    if noise is not None:
        return F.l1_loss(noise,data-target,reduction=reduction)
    else:
        return F.l1_loss(data, target,reduction=reduction)
    
def psnr_loss(data, target,reduction=None,dim=0):
    return peak_signal_noise_ratio(data.unsqueeze(0),target.unsqueeze(0),data_range=2,dim=dim,reduction=reduction)
    
def ssim_loss(data, target,reduction=None):
    return structural_similarity_index_measure(data,target,data_range=2,return_full_image=True)[1]
    

def smooth_l1_loss(data, target,noise=None,reduction=None):
    if noise is not None:
        return F.smooth_l1_loss(noise,data-target,reduction=reduction)
    else:
        return F.smooth_l1_loss(data, target,reduction=reduction)

def vgg_loss(data, target,noise=None,reduction="mean"):
    data, target = normto01(data),normto01(target)
    vgg = VGGLoss(reduction=reduction).to(target.device)
    
    if noise is not None:
        return vgg(noise,data-target,target_is_features=False)
    else:
        return vgg(data,target,target_is_features=False)
    

def combo_loss(
    data: torch.Tensor,
    target: torch.Tensor,
    noise: torch.Tensor = None,
    alpha: int = 0.93,
    beta: Optional[int] = None,
    base_loss: str = "l1",
    supp_loss: str = "topo",
    **params
) -> torch.Tensor:
    
    base_losses = {"mse": mse_loss, "l1": l1_loss, "sml1": smooth_l1_loss}
    supp_losses = {"topo": topo_loss, "vgg": vgg_loss}
   # print("COMBO****************")
    if beta is None:
        beta = 1 - alpha
    
    
    base_loss_vals = base_losses[base_loss](data, target,noise)
    
    supp_loss_vals =  supp_losses[supp_loss](data, target, **params)
    
    
   # print(f"base vals: {alpha* base_loss_vals.item():.3f}, topo vals: {beta*supp_loss_vals.item():.3f}")
    L = alpha * base_loss_vals + beta * supp_loss_vals

    return L

def tricombo_loss(
    data: torch.Tensor,
    target: torch.Tensor,
    noise: torch.Tensor = None,
    alpha: float = None,
    betas:  list = [0.005,0.0005],
    base_loss: str = "l1",
    supp_losses: list = ["topo","vgg"],
    **params
) -> torch.Tensor:
    
    assert len(betas)==len(supp_losses)#:,"Len(betas) should be = Len(supp_losses)"
    
    alpha = 1 - sum(betas)
    
    base_loss_fn = {"mse": mse_loss, "l1": l1_loss, "sml1": smooth_l1_loss}
    supp_loss_fn = {"topo": topo_loss, "vgg": vgg_loss}
   # print("COMBO****************")

    
    base_loss_vals = base_loss_fn[base_loss](data, target,noise)
    
    supp_loss_vals =  [supp_loss_fn[supp_loss](data, target, **params) for supp_loss in supp_losses]
    
   # print(f"base vals: {alpha* base_loss_vals.item():.3f}, topo vals: {beta*supp_loss_vals.item():.3f}")
    L = alpha * base_loss_vals + sum([beta * supp_loss for beta,supp_loss in zip(betas,supp_loss_vals)])

    return L

def combo_masked_loss(
    data: torch.Tensor,
    target: torch.Tensor,
    base_loss: str = "l1",
    loss_type = "wass",
    wave='haar',
    level=1,
    maxdim=1,
    patch=False,
    reduce_mask=True,
    **params
) -> torch.Tensor:


    base_losses = {"mse": mse_loss, "l1": l1_loss, "sml1": smooth_l1_loss,"psnr": psnr_loss,"ssim":ssim_loss}
    
    
    gt_mask = calc_mask(target,level=level, wave=wave,reduce_mask=reduce_mask)
    #print(gt_mask.shape)
    
    topo_loss_vals = topo_loss(noisy=data, gt=target,
                                 patch=patch,
                                 loss_type=loss_type,
                                 maxdim=maxdim,
                                 reduce="none",
                                 **params)
    
    topo_loss_vals = topo_loss_vals.view(*target.shape[0:2])/max(target.shape[-1],target.shape[-2])
    if base_loss == "ones":
        base_loss_img = torch.ones(data.shape)
    else:
        base_loss_img = base_losses[base_loss](data=data,target=target,reduction="none")
    masked_loss= base_loss_img*(1-gt_mask) +topo_loss_vals.unsqueeze(-1).unsqueeze(-1)*gt_mask
    return masked_loss.mean()

def tricombo_masked_loss(
    data: torch.Tensor,
    target: torch.Tensor,
    base_loss: str = "l1",
    vgg_beta=0,
    loss_type = "wass",
    wave='haar',
    level=1,
    maxdim=1,
    patch=True,
    reduce_mask=True,
    **params
) -> torch.Tensor:


    base_losses = {"mse": mse_loss, "l1": l1_loss, "sml1": smooth_l1_loss}
    
    
    gt_mask = calc_mask(target,level=level, wave=wave,reduce_mask=reduce_mask)
    
    topo_loss_vals = topo_loss(noisy=data, gt=target,
                                 patch=patch,
                                 loss_type=loss_type,
                                 maxdim=maxdim,
                                 reduce="none",
                                 **params)
    
    topo_loss_vals = topo_loss_vals.view(*target.shape[0:2])/max(target.shape[-1],target.shape[-2])
    base_loss_img = base_losses[base_loss](data=data,target=target,reduction="none")
    
    if vgg_beta != 0:
       # upscale = Upsample(scale_factor=2, mode='nearest')
        vgg_loss_img = vgg_loss(data=data,target=target,reduction="mean")
        masked_loss= (base_loss_img*(1-gt_mask) +topo_loss_vals.unsqueeze(-1).unsqueeze(-1)*gt_mask)*(vgg_beta) + (1-vgg_beta)*vgg_loss_img.unsqueeze(-1).unsqueeze(-1)
   
    else:
        masked_loss= base_loss_img*(1-gt_mask) +topo_loss_vals.unsqueeze(-1).unsqueeze(-1)*gt_mask
        
    #print(topo_loss_vals,base_loss_img.mean(),vgg_loss_img)
    return masked_loss.mean()


import torch

from functools import partial
from time import time
from torch_topological.nn import (
    VietorisRipsComplex,
    CubicalComplex,
    SummaryStatisticLoss,
    MultiScaleKernel,
    WassersteinDistance,
    SlicedWassersteinDistance,
    SignatureLoss,
)

from pytorch_wavelets import DWTForward
from .toputils import ch_sample_from_densest


def calc_pd_rieck(
    img,
    maxdim: int = 0,
    patch: bool = False,
    p: int = 2,
    q: int = 2,
    reshape=True,
    **params
):
    """
    calc_pd_rieck _summary_

    _extended_summary_

    Args:
        img (_type_): wdescription_
        maxdim (int, optional): wdescription_. Defaults to 0.
     
        patch (bool, optional): _description_. Defaults to True.
        p (int, optional): _description_. Defaults to 2.
        q (int, optional): _description_. Defaults to 2.
    """

    def reshape_batch(img):
        return img.view(img.shape[0] * img.shape[1], img.shape[2], img.shape[3])
    
    
    if patch:
        img = ch_sample_from_densest(img, reduce=True,**params)
    else:
        img = reshape_batch(img)
    comp = VietorisRipsComplex(dim=maxdim, p=p).to("cuda")
    pd = list(comp(img))
    return pd



def topo_loss(
    noisy: torch.Tensor,
    gt: torch.Tensor,
    maxdim: int = 0,
    patch: bool = True,
    p: int = 2,
    q: int = 2,
    loss_type="wass",
    reduce="mean",
    out_log =True,
    n_samples=1000,
    n_neighbors=300,
    **params
) -> torch.Tensor:
    
    pd = {}
    lossfn = {}

    calc_pd = partial(
        calc_pd_rieck, patch=patch,n_samples=n_samples,n_neighbors=n_neighbors,maxdim=maxdim,**params
    )

    pd["noisy"], pd["clean"] = list(map(calc_pd, [noisy, gt]))
    
    if loss_type == "wass":
        lossfn = WassersteinDistance(p=torch.inf)
    if loss_type == "totpers":
        lossfn = SummaryStatisticLoss(summary_statistic="total_persistence", p=p, q=q)
    if loss_type == "swass":
        lossfn = SlicedWassersteinDistance()
    loss = torch.stack(list(map(lossfn, *list(pd.values()))))
    if patch:
        loss = loss/n_samples
    if out_log:
        loss = torch.log(1+loss)
    if reduce == "mean":
        return loss.mean()
    elif reduce == "none":
        return loss
    
         
 
    
    
    
#### For test purposes, contains all loss types between two pds
    
def calc_losses_pd(pd_img,pd_gt, p=2, q=2,coeff=[1,1],out_log=False):
    pd = {}
    lossfn = {}
    losses = {}
    loss_types = [
            "wass",
            "msk",
            "polynomial_function",
            "total_persistence",
            "p_norm",
            "persistent_entropy",
        ]
    
    pd["noisy"], pd["clean"] = pd_img, pd_gt

    lossfn["wass"] = WassersteinDistance(p=torch.inf)
    lossfn["swass"] = SlicedWassersteinDistance()
    lossfn["msk"] = MultiScaleKernel(1.0)
    lossfn["sign"] = SignatureLoss(
        p=p, normalise=True, dimensions=list(range(len(coeff)))
    )

    for ltype in [
        "persistent_entropy",
        "polynomial_function",
        "total_persistence",
        "p_norm",
    ]:

        lossfn[ltype] = SummaryStatisticLoss(summary_statistic=ltype, p=p, q=q)
        
    for loss_comp in loss_types:


        losses[loss_comp] = torch.mean(
            torch.stack(list(map(lossfn[loss_comp], *list(pd.values()))))
        ).item()
            
    if out_log:    
        return {k: math.log(l) for k, l in losses.items()}
    else:
        return losses



import gudhi
import gudhi.wasserstein


import torch
import math
from typing import Literal

from ripser import Rips

import numpy as np

def ch_sample_from_densest(
    img: torch.tensor,
    top_portion: float = 0.2,
    n_neighbors: int = 300,
    n_samples: int = 500,
    reduce = True,
) -> torch.tensor:
    #print("in")
    top_contrast_patches = get_top_contrast_patches_from_img(
        img, top_portion=top_portion
    )
    
    densest_patches = get_densest_patches(
        patches=top_contrast_patches,
        k=n_neighbors,
        n_samples=n_samples# // img.shape[-3],
    )
    b,c,h,w = densest_patches.shape
    if reduce:
    # concat over channels
        #cat_patches = torch.cat(
        #    densest_patches.tensor_split(densest_patches.shape[-3], dim=-3), dim=-2
        #)
        #print("catp",cat_patches.shape)
        #return cat_patches.squeeze(dim=1)
        return densest_patches.view(b*c, h,w) 
    else:
        return densest_patches


def get_top_contrast_patches_from_img(
    batch_img: torch.tensor, top_portion: float = 0.2
) -> torch.tensor:

    """
    Calculates contrast norm and returns most contrast patches of the image
    
    
    Args:
    batch_img
    top_portion
    
    """

    # extract patches and calculate d_norm
    data = extract_patches_batch(batch_img)
    dnorms = d_norm(data)
    
    #sort by values of d-norms
    sort_idcs = torch.argsort(dnorms)

    sorted_idcs = torch.argsort(dnorms)
    sorted_idcs = torch.broadcast_to(sort_idcs.unsqueeze(-1), data.shape)
    sorted_vals = torch.gather(data, -2, sorted_idcs)
    
    # get top_portion of contrast patches
    top_contrast = sorted_vals[:, :, int(-top_portion * dnorms.shape[-1]) :, :]
    
    #normalize to a sphere
    top_contrast = torch.nn.functional.normalize(
        top_contrast - top_contrast.mean(dim=-1, keepdim=True), dim=-1
    )

    return top_contrast


def get_densest_patches(patches, k, n_samples):
    """
    Parameters:
    patches: B x C x N x D
    k: k in KNN
    n_samples: number of samples to pick from the top densest


    Returns:

    (sorted by distance) pairs of (dist, idx) for k nearest neighbours for input of shape B x C x n_samples x D
   

    """

    cdist = torch.cdist(patches, patches)

    # knn on patches list
    # get k nearest neighbours for each patch
    dists, indices = cdist.topk(k, dim=-1, largest=False, sorted=True)
    # dist, indcs: (B x C x N x k, B x C x N x k)

    # for each patch leave only furthest neigh for each patch
    # and sort accoording to distance to this furthest (m-th) neighbour  (m-th distance)
    sorted_nbrs_idcs = torch.argsort(dists[..., -1])

    # take top ((densest_part)*100)% of densest points
    # (in bottom ((densest_part)*100)% according to the m-distance)
    sorted_nbrs_idcs = torch.broadcast_to(sorted_nbrs_idcs.unsqueeze(-1), patches.shape)
    sorted_nbrs_vals = torch.gather(patches, -2, sorted_nbrs_idcs)
    # perm = torch.randperm(sorted_nbrs_vals.size(-2))
    # idx = perm[:n_samples]
    # samples = tensor[idx]
    # return sorted_nbrs_vals[..., idx, :]
    return sorted_nbrs_vals[..., :n_samples, :]


def d_norm(x):
    """
    d_norm [summary]
    calculates contrast norm for 3x3 patch (9-vector)

    Args:
        x (tensor): flattened patch (9-vector) of size B x C x 9

    Returns:
        tensor: D-norm value
    """
    cmatrix = torch.tensor(
        [
            [2, -1, 0, -1, 0, 0, 0, 0, 0],
            [-1, 3, -1, 0, -1, 0, 0, 0, 0],
            [0, -1, 2, 0, 0, -1, 0, 0, 0],
            [-1, 0, 0, 3, -1, 0, -1, 0, 0],
            [0, -1, 0, -1, 4, -1, 0, -1, 0],
            [0, 0, -1, 0, -1, 3, 0, 0, -1],
            [0, 0, 0, -1, 0, 0, 2, -1, 0],
            [0, 0, 0, 0, -1, 0, -1, 3, -1],
            [0, 0, 0, 0, 0, -1, 0, -1, 2],
        ],
        dtype=x.dtype,
        device=x.device,
    )

    x = x.unsqueeze(-2)

    return torch.sqrt((x @ cmatrix) @ x.transpose(-1, -2)).squeeze()


def prune_pd_torch(pd):
    """
    Prune diagrams (remove inf values)
    """

    return tuple(p[p[:, 1] != math.inf] for p in pd)


def extract_patches_batch(
    batch_img: torch.tensor, psize: int = 3, stride: int = 2
) -> torch.tensor:
    """
    Extracts patches from torch image batch of size N x C x H x W

    Parameters:
    batch_img
    psize
    stride

    Returns:
    torch tensor of collection of flattened patches, size N x C x NPATCHES x psize*psize
    """
    patches2 = batch_img.unfold(2, size=psize, step=stride).unfold(
        3, size=psize, step=stride
    )
    namedpatches2 = patches2.refine_names("N", "C", "PN1", "PN2", "PH", "PW")
    return (
        namedpatches2.flatten(["PN1", "PN2"], "PN")
        .flatten(["PH", "PW"], "PC")
        .rename(None)
    )




def wass_dist(pds):

    wd = F.WassersteinDistance.apply(pds[0], pds[1])
    
    if wd == 0:
        wd = torch.zeros(1)
    return wd



def prune_pd(pd):
    """
    Prune diagrams (remove inf values)
    """
    pd = [p.tolist() for p in pd]
    # print(len(pd),len(pd[0][0]))
    # print(len(pd), len(pd[0]))
    for j in range(len(pd)):
        count = 0
        for i in range(len(pd[j])):
            if pd[j][i - count][1] == math.inf:
                # print(pd[j][i-count])
                del pd[j][i - count]
                count += 1
    return [np.asarray(l) for l in pd]


def get_pd_cpu(
    patches: torch.Tensor, backend: Literal["rips", "ripspp"] = "rips", maxdim: int = 1
) -> np.array:
    """
    Calculates PD for a set of patches
    
    backend: "rips" or "ripspp"
    """

    if isinstance(patches, torch.Tensor):
        patches = patches.cpu().detach().numpy()

    # assert patches.size==2
    if backend == "rips":

        rips = Rips(maxdim=maxdim, verbose=False)

        # begin = time()
        pd = rips.fit_transform(patches, distance_matrix=False)
        # end = time()

        # print(f"Rips time: {end-begin}")

        return prune_pd(pd)

    elif backend == "ripspp":

        import ripserplusplus as rpp

        # begin = time()
        # print(f"rpp location: {rpp.__file__}")
        pd = rpp.run("--format point-cloud", patches)
        # end = time()
        # print(f"Ripspp time: {end-begin}")
        pd = [np.array([list(el) for el in pd[d]]) for d in range(len(pd))]
        return prune_pd(pd)

    else:
        raise ValueError("Wrong backend type! Should be 'rips' or 'ripspp'")

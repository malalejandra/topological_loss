import sys
import os
from pathlib import Path

# import numpy as np

from typing import Union, List
from multiprocessing import cpu_count

# import torch

# from torch import tensor
from torchvision import transforms


from .dataset import (
    CFAImageDataset,
    ToTensor,
    RandomCrop,
    CenterCrop,
    FloatScale,
    DebayerPair,
    ToChannels,
)

from base import BaseDataLoader


sys.path.append("./..")

# mean_noisy = tensor([10375.5342,  9444.5625,  9444.2227,  9381.9883])
# std_noisy = tensor([527.8456, 484.7748, 484.4405, 491.7215])
# mean_gt = tensor([10097.5762,  9211.6338,  9218.2295,  9013.4863])
# std_gt = tensor([521.7043, 480.2397, 480.3188, 483.5645])


class CFADataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size=8,
        n_patches_per_im=1,
        shuffle=True,
        isos: List[int] = [110, 600000],
        nshots: Union[int, List[int]] = None,
        cams: Union[str, List[str]] = None,
        scenes: Union[str, List[str]] = None,
        validation_split=0.2,
        num_workers=8,
        size=(256, 256),
        crop_type="random",
        scaling=True,
        sc_type="neg1to1",
        pin_memory=True,
        debayer=True,
        drop_last = True
    ):
        """
        __init__ [summary]

        [extended_summary]

        Args:
            data_dir ([type]): Directory with images (with data organised as data_dir/scene_i/cam_j/shot_k.tiff)
            batch_size (int, optional): Batch size. Defaults to 8.
            n_patches_per_im (int, optional): number of patches to sample from one image. Defaults to 1 and if crop_type is "center", can't be >1.
            shuffle (bool, optional): Shuffle dataset?. Defaults to True.
            isos (List[int], optional): ISO range [min,max]. Defaults to [110, 600000] (use all).
            nshots (Union[int,List[int]], optional): Shot number used. Defaults to None (use all).
            cams (Union[str,List[str]], optional):  Specify list of strings of cameras. Defaults to None (use all).
            scenes (Union[str,List[str]], optional): Specify list of strings of scenes. Defaults to None (use all).
            validation_split (float, optional): Proportion of validation set. Defaults to 0.2.
            num_workers (int, optional): Number of workers in loader. Defaults to 8.
            size (tuple, optional): Patch size. Defaults to (256,256).
            crop_type (str, optional): Crop transformation type, "center" or "random". Defaults to "random".
            scaling (bool, optional): Scaling output image to a range defined by sc_type. Defaults to True.
            sc_type (str, optional): Scaling type ("zeroto1" or "neg1to1") for output image range to be [0,1] or [-1,1]. Defaults to "neg1to1".
            pin_memory (bool, optional): Pin_memory in torch data loader. data Defaults to True.
            debayer (bool, optional): Debayer result or leave CFA. Defaults to True.
            drop_last (bool, optional): Drop last batch if len<b_size. Defaults to True.

        Raises:
            AssertionError: [description]
            AssertionError: [description]
        """
        if num_workers is None:
            num_workers = len(os.sched_getaffinity(0)) - 1

        if (n_patches_per_im > 1) & (crop_type == "center"):
            raise AssertionError(
                "Either change crop_type to 'random' or set n_patches_per_img=1!"
            )

        self.shuffle = shuffle
        self.scaling = scaling
        self.debayer = debayer
        self.sc_type = sc_type
        self.isos = isos
        self.cams = cams
        self.scenes = scenes
        self.nshots = nshots
        self.drop_last = drop_last

        self.data_dir = Path(data_dir)

        size = tuple(size)

        # print(f"type size {type(size)}")

        crop_transf = {"random": RandomCrop(size), "center": CenterCrop(size)}

        transf_list = [ToTensor(), crop_transf[crop_type]]

        if scaling:
            transf_list.insert(0, FloatScale(sc_type=sc_type))

        if debayer:
            transf_list.append(DebayerPair(scaling))
        else:
            transf_list.insert(0, ToChannels())

        transform = transforms.Compose(transf_list)

        self.dataset = CFAImageDataset(
            isos=self.isos,
            n_patches_per_im=n_patches_per_im,
            cams=self.cams,
            nshots=self.nshots,
            scenes=self.scenes,
            img_dir=self.data_dir,
            transform=transform,
        )

        super().__init__(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_split=validation_split,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=self.drop_last
        )


if __name__ == "__main__":

    isos = "all"

    bs = 16

    ncpus = cpu_count()

    data_dir = Path("../../test-Dataset/3_aligned-matched/")

    dataloader = CFADataLoader(
        isos=isos, data_dir=data_dir, batch_size=bs, shuffle=True, num_workers=4
    )

    print(dataloader)


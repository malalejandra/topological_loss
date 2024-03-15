import torch
from torch.utils.data import Dataset


from pathlib import Path
import numpy as np
from utils import cfa_to_chnls, tiff_read, chnls_to_cfa, Debayer3x3

from typing import Union, List

from skimage import img_as_float32

from torchvision import transforms

import re

from natsort import natsorted


import torchvision.transforms.functional as tf


class CFAImageDataset(Dataset):
    """
    A dataset of noisy and clean CFA image pairs, that consists of images with different a range of iso parameters defined by isos list.
    """

    def __init__(
        self,
        img_dir: Union[Path, str],
        n_patches_per_im: int,
        isos: List[int],
        nshots: Union[int, List[int]] = None,
        cams: Union[str, List[str]] = None,
        scenes: Union[str, List[str]] = None,
        transform=None,
        shuffle=True
    ):

        """
        Args:

        isos - range of iso values or None for all (default)
        
        img_dir - path/to/imgs/ (ls img_dir: scene1 scene2)
        
        transform - transform to apply
        


        Returns:

        [img_noisy, img_clean, label]

        """

        self.img_paths = self.get_img_paths_from_dir(
            img_dir, isos=isos, cams=cams, scenes=scenes, nshots=nshots
        )

        self.labels = self.get_img_labels(self.img_paths)
        
        #self.shuffle = shuffle
        
       # if self.shuffle:
        #   # print("shuffling")
        #    self.indices = np.arange(len(self.img_paths))
           # np.random.shuffle(self.indices)
            
            #self.img_paths = np.array(self.img_paths)[self.indices]
            #self.labels = np.array(self.labels)[self.indices]
            #print(self.labels)
        

        self.n_patches_per_im = n_patches_per_im

        self.transform = transform

    def get_img_paths_from_dir(
        self,
        img_dir: Path,
        isos: List[int],
        cams: Union[str, List[str]],
        scenes: Union[str, List[str]],
        nshots: Union[int, List[int]],
    ) -> list:

        img_dir = Path(img_dir)
        img_paths = list(img_dir.glob(f"./*/*/iso-*-shot*"))

        def get_iso_from_name(x):
            iso_pattern = re.compile("iso-([0-9]+)-shot-.")

            # print(int(iso_pattern.match(x.name).group(1)))
            return int(iso_pattern.match(x.name).group(1))

        img_paths = [x for x in img_paths if get_iso_from_name(x) in range(*isos)]

        labels = self.get_img_labels(img_paths)

        # filter labels
        filt_inds = [
            ind
            for ind, x in enumerate(labels)
            if ((not bool(cams)) or (any(x["camera"] == c for c in cams)))
            and ((not bool(nshots)) or (any(x["nshot"] == n for n in nshots)))
            and ((not bool(scenes)) or (any(x["scene"] == s for s in scenes)))
        ]
        # return filtered paths
        img_paths = [img_paths[ind] for ind in filt_inds]

        return natsorted(img_paths)

    def get_target_path(self, img_path: Path) -> np.array:
        """A function generating target (clean image) given a (noisy) image path

        Args:
            img_path (Path): image path

        Returns:
            gt_img : target image
        """

        gt_path = img_path.parent / "averaged-iso-100.tiff"

        return tiff_read(gt_path)

    def get_img_labels(self, paths_list: list) -> dict:
        """
        Args:
        img_path (Path): image path

        Returns:
            
            labels (dict): dictionary of labels with keys "iso" (int), "nshot" (int), "camera" (str), and "scene" (str)
            
        """
        labels = []
        for img_path in paths_list:
            iso_pattern = re.compile("iso-([0-9]+)-shot-([0-9]+)-")
            iso = int(iso_pattern.match(img_path.parts[-1]).group(1))
            nshot = int(iso_pattern.match(img_path.parts[-1]).group(2))
            camera = img_path.parts[-2]
            scene = img_path.parts[-3]

            labels.append(
                {"iso": iso, "nshot": nshot, "camera": camera, "scene": scene}
            )

        return labels

    def __getitem__(self, index: int) -> tuple:
        #print(self.labels)

        x = tiff_read(self.img_paths[index // self.n_patches_per_im])

        y = self.get_target_path(self.img_paths[index // self.n_patches_per_im])

        label = self.labels[index // self.n_patches_per_im]
        if self.transform:
            x, y = self.transform((x, y))
        #print("in item")
        return x, y, label

    def __len__(self):

        return len(list(self.img_paths)) * self.n_patches_per_im


class DebayerPair(object):
    """
    Pytorch Debayer
    
    Args:

    """

    def __init__(self, scaling):
        self.scaling = scaling

    def __call__(self, sample):
        noisy, gt = sample[0], sample[1]

        deb = Debayer3x3()
        
        return deb(noisy), deb(gt)


class ToTensor(object):
    """
    Converts a pair of noisy,gt images into to float32 tensors
    
    """

    def __call__(self, sample):

        noisy, gt = sample[0], sample[1]

        noisy = torch.from_numpy(np.array(noisy).astype(np.float32))
        gt = torch.from_numpy(np.array(gt).astype(np.float32))

        return noisy, gt


class ToChannels(object):
    def __call__(self, sample):

        noisy, gt = sample[0], sample[1]

        return list(map(cfa_to_chnls, [noisy, gt]))


class FloatScale(object):
    """
    Scales to [-1,1]
    
    Args:

    """

    def __init__(self, int_dtype=np.uint16, sc_type="neg1to1"):
        self.sc_type = sc_type
        self.int_dtype = int_dtype

    def __call__(self, sample):

        noisy, gt = sample[0], sample[1]

        assert (
            noisy.dtype == self.int_dtype
        ), f"input image should have dtype {self.int_dtype}"

        # print(f"in scale: {img_as_float32(gt).min(),img_as_float32(gt).max()}")
        if self.sc_type == "neg1to1":
            return 2 * img_as_float32(noisy) - 1, 2 * img_as_float32(gt) - 1
        elif self.sc_type == "zeroto1":
            return img_as_float32(noisy), img_as_float32(gt)


class CenterCrop(object):
    """
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size // 2, output_size // 2)
        else:
            assert len(output_size) == 2
            self.output_size = tuple(i // 2 for i in output_size)

    def __call__(self, sample):

        noisy, gt = sample[0], sample[1]

        noisy, gt = list(map(cfa_to_chnls, [noisy, gt]))

        ccrop = transforms.CenterCrop(self.output_size)

        noisy = ccrop(noisy)
        gt = ccrop(gt)

        return list(map(chnls_to_cfa, [noisy, gt]))


class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size // 2, output_size // 2)
        else:
            assert len(output_size) == 2
            self.output_size = tuple(i // 2 for i in output_size)

    def __call__(self, sample):

        noisy, gt = sample[0], sample[1]

        noisy, gt = list(map(cfa_to_chnls, [noisy, gt]))
        i, j, h, w = transforms.RandomCrop.get_params(
            noisy, output_size=self.output_size
        )
        noisy = tf.crop(noisy, i, j, h, w)
        gt = tf.crop(gt, i, j, h, w)

        return list(map(chnls_to_cfa, [noisy, gt]))


class ToScale(object):
    """
    Switch between [-1,1] or [0,1]
    
    Args:

    """

    def __init__(self, sc_type="neg1to1"):
        self.sc_type = sc_type

    def __call__(self, sample):

        noisy, gt = sample[0], sample[1]

        if self.sc_type == "neg1to1":
            return 2 * noisy - 1, 2 *gt - 1
        elif self.sc_type == "zeroto1":
            return 0.5 * noisy + 0.5, 0.5 * gt + 0.5

class CoordCrop:
    """
    Crop the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """


    def __init__(self, output_size, i, j):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size // 2, output_size // 2)
        else:
            assert len(output_size) == 2
            self.output_size = tuple(i // 2 for i in output_size)
        self.i = i
        self.j = j

    def __call__(self, sample):

        i, j, h, w = transforms.RandomCrop.get_params(
            sample, output_size=self.output_size
        )
        cropped = tf.crop(sample, self.i, self.j, h, w)

        return cropped



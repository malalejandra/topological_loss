import torch
from pathlib import Path
import sys

sys.path.append("..")
from utils import read_json
import model as model_module


def get_model_config(model_path):
    """
    
    """
    model_path = Path(model_path)
    if "pix2pix" in str(model_path).lower():
        config_p = model_path.parent.parent / "config.json"
        # print("conf_p",config_p)
    else:
        config_p = model_path.parent / "config.json"
    return read_json(config_p)


def get_dl_params(model_path, seed=12, **kwargs):

    torch.manual_seed(seed)
    np.random.seed(seed)

    conf = get_model_config(model_path)

    dl_params = conf["dataloader"]["args"]
    print("params:", dl_params)
    for key, value in kwargs.items():
        dl_params.update({key: value})

    return dl_params


def get_nn_from_config(conf):

    for ch_par in ["channels", "in_channels", "out_channels"]:
        if ch_par in conf["arch"]["args"]:
            if conf["dataloader"]["args"]["debayer"]:
                conf["arch"]["args"][ch_par] = 3
            else:
                conf["arch"]["args"][ch_par] = 4

    model_dict = conf["arch"]

    # print(model_dict)

    if "pix2pix" not in conf["name"]:
        m = getattr(model_module, model_dict["type"])
    else:
        m = model_module.Pix2Pix()
    # print(type(m))

    if "pix2pix" not in conf["name"]:
        model = m(**model_dict["args"])
    else:
        model = model.netG
    #   print(type(model))

    return model


import sys,os

#os.environ['LD_LIBRARY_PATH'] +=  f"{os.pathsep + os.environ['CONDA_PREFIX']+'/lib'}"

#print(os.environ['LD_LIBRARY_PATH'])
import random
import argparse
import collections
import torch
import numpy as np
import dataloader.dataloaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer,TrainerGAN
from utils import prepare_device
#from torch.cuda.amp import GradScaler

# https://towardsdatascience.com/i-am-so-done-with-cuda-out-of-memory-c62f42947dca

#scaler = GradScaler()


# from pathlib import Path

from functools import partial






def main(config):
    # fix random seeds for reproducibility
    SEED = config["trainer"]["seed"]

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    
    name = config["name"]
    logger = config.get_logger("train")
    pth = config["dataloader"]["args"]["data_dir"]

    for ch_par in ["channels", "in_channels", "out_channels"]:
        if ch_par in config["arch"]["args"]:
            if config["dataloader"]["args"]["debayer"]:
                config["arch"]["args"][ch_par] = 3
            else:
                config["arch"]["args"][ch_par] = 4

    # setup dataloader instances
    dataloader = config.init_obj("dataloader", module_data)
    # print(f"len: {len(dataloader)}")
    valid_dataloader = dataloader.split_validation()

    # build model architecture, then print to console

    is_gan = (
        "pix2pix" in config["arch"]["type"].lower()
    )  # prepare for (multi-device) GPU training

    device, device_ids = prepare_device(config["n_gpu"])
    if is_gan:
        config["arch"]["args"]["device_ids"] = device_ids

    else:
        pass
    # print(f"mod config d_ids: {config['arch']['args']['device_ids']}")
    model = config.init_obj("arch", module_arch)
    if not is_gan:
        model = torch.jit.script(model)


    #    device, device_ids = prepare_device(config['n_gpu'])
    if is_gan:
        model.criterionL1 = config.init_ftn("loss", module_loss)
        netG = config["arch"]["args"]["netG"]
    else:
        netG = ""
    logger.info(model)
    device, device_ids = prepare_device(config["n_gpu"])
    output_noise = config["arch"]["output_noise"]
  
    if not is_gan:
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # get function handles of loss and metrics

        criterion = config.init_ftn("loss", module_loss)
        

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
        lr_scheduler = config.init_obj(
            "lr_scheduler", torch.optim.lr_scheduler, optimizer
        )
    else:
        criterion = None
        optimizer = None
        lr_scheduler = None
      
        
   # print("CRITERION:",criterion)
    # metric params are acquired from dataloader params
    metrics = [
        partial(
            getattr(module_metric, met),
            #scaling=config["data()er"]["args"]["scaling"],
            sc_type=config["dataloader"]["args"]["sc_type"],
        )
        for met in config["metrics"]
    ]
    #print("METRS:", metrics)

    for m in metrics:
        m.__name__ = m.func.__name__

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler

    img_log_step = config["trainer"]["img_log_step"]
    autocast = config["trainer"]["autocast"]
    
    if is_gan:
        trainer = TrainerGAN(
            model,
            criterion,
            metrics,
            optimizer,
            config=config,
            device=device,
            dataloader=dataloader,
            valid_dataloader=valid_dataloader,
            lr_scheduler=lr_scheduler,
            img_log_step=img_log_step,
            output_noise=output_noise,
        )
    else:
        trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        dataloader=dataloader,
        valid_dataloader=valid_dataloader,
        lr_scheduler=lr_scheduler,
        img_log_step=img_log_step,
        output_noise=output_noise,
    )
        
    
    
   
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="dataloader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

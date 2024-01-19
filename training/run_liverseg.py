"""
The training process of liver tumor segmentation network, 
including model selection and initialization, data preparation, training loop, model saving, etc
"""

import os 
import numpy as np
import random
import argparse
import cv2 
import yaml
from bisect import bisect_right

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from torch.nn.functional import interpolate, adaptive_max_pool3d, avg_pool3d
from torch.cuda.amp import autocast as autocast

from liver_seg.model.liverseg import LiverSegNet
from data.liver_seg_2 import npyDataSet

torch.manual_seed(7) # cpu
torch.cuda.manual_seed(7) #gpu
np.random.seed(7) #numpy
random.seed(7) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn



def run_main(args):
    max_iters = 240000

    # Initialize LiverSegNet model
    unet = LiverSegNet(num_seg_classes=args["num_class"], inchannel=args["inchannel"])

    # Load model weights
    model_pth = unet.state_dict()
    if args["resume"]>-1:
        print('load from ', os.path.join(args["save_folder"],str(args["resume"]).zfill(5)+"_stage_one.pth"))
        pth = torch.load(os.path.join(args["save_folder"],str(args["resume"]).zfill(5)+"_stage_one.pth"))
        model_pth = pth["model"]
    elif args["pretrain"] is not None: 
        pth = torch.load(args["pretrain"])
        pth = pth["model"]
        for key in pth:
            if pth[key].shape==model_pth[key].shape:
                model_pth[key] = pth[key]
            else: 
                print(key, pth[key].shape, model_pth[key].shape)
    if model_pth is not None:
        unet.load_state_dict(model_pth)
    unet.cuda()

    # Initialize dataset
    sdf = dataset_dict[args["dataset"]](args["json_path"], args["num_image"])
    _data_loader = DataLoader(sdf, batch_size=args["batch_size"], shuffle=True, num_workers=16, pin_memory=True)

    solver = torch.optim.AdamW(unet.parameters(), eps=1e-8, betas=(0.9, 0.99),lr=5e-4, weight_decay=5e-2)
    lr_schedule = WarmUpLR(solver, max_iters, warmup_iters=5000)
    num_iter = 0
    if args["resume"]>-1:
        lr_schedule.load_state_dict(pth["opti"])
        num_iter = args["resume"]

    unet.train()
    
    flag = True
    while flag:
        data_loader = PreFetch(_data_loader)
        image, sdf = data_loader.next()
        while image is not None:
            solver.zero_grad()

            # Forward pass
            outputs = unet.forward(image.contiguous())
            del image
            
            """
            # Compute loss
            Loss = DiceLoss + CeLoss * weights
            """

            loss.backward()
            sdf = _sdf
            solver.step()
            output[output>=0.5] = 1
            output[output<1] = 0
            out_reslt = {}
            
            """
            # Compute metrics and print results
            """
            del output
            image, sdf = data_loader.next()
            lr_schedule.step()

            # Save model at certain iterations
            if num_iter % 2500 == 0 and num_iter>0:
                save_dict = {}
                save_dict["model"] = unet.state_dict()
                save_dict["opti"] = lr_schedule.state_dict()
                torch.save(save_dict, args["save_folder"] + "/" + str(num_iter).zfill(5) + "_stage_one" + ".pth")
                print(args["save_folder"] + "/" + str(num_iter).zfill(5) + "_stage_two_stage_loss" + ".pth")
            if num_iter>=max_iters:
                flag = False
                break
            num_iter += 1


def arg():
    parase = argparse.ArgumentParser()
    parase.add_argument("--config", type=str, required=True)
    return parase.parse_args()

    

if __name__ == "__main__":
    args = arg()
    args = yaml.load(open(args.config, "r"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu_id"])
    if not os.path.exists(args["save_folder"]):
        os.makedirs(args["save_folder"])
    run_main(args)

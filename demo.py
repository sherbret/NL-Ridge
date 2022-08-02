#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : NL-Ridge
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2022, v1.0.

import torch
from torchvision.io import read_image, write_png
from nlridge import NLRidge
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, dest="sigma", help="Standard deviation of the noise (noise level). Should be between 0 and 50.", default=15)
parser.add_argument("--in", type=str, dest="path_in", help="Path to the image to denoise (PNG or JPEG).", default="./test_images/barbara.png")
parser.add_argument("--out", type=str, dest="path_out", help="Path to save the denoised image.", default="./denoised.png")
parser.add_argument("--add_noise", action='store_true', help="Add artificial Gaussian noise to the image.")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reading
img = read_image(args.path_in)[None, :, :, :].float().to(device)
img_noisy = img + args.sigma * torch.randn_like(img) if args.add_noise else img

# Choice of the parameters 
if args.sigma <= 15:
    model = NLRidge(7, 7, 18, 55, 37, 4) 
elif args.sigma <= 35:
    model = NLRidge(9, 9, 18, 90, 37, 4) 
else:
    model = NLRidge(11, 9, 20, 120, 37, 4)

# Denoising
t = time.time()
den = model(img_noisy, args.sigma)
print("Time elapsed:", round(time.time() - t, 3), "seconds")
den = den.clip(0, 255)

# Performance in PSNR
if args.add_noise:
    psnr = 10*torch.log10(255**2 / torch.mean((den - img)**2))
    print("PSNR:", round(float(psnr), 2), "dB")

# Writing
write_png(den[0, :, :, :].byte(), args.path_out)

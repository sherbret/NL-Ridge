#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : NL-Ridge
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2022, v1.0.

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
import argparse
from nlridge import NLRidge
import time

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, dest="sigma", help="Standard deviation of the noise (noise level). Should be between 1 and 50.", default=15)
parser.add_argument("--in", type=str, dest="img_to_denoise", help="Path to the image to denoise.", default="./datasets/Set12/08.png")
parser.add_argument("--out", type=str, dest="img_to_save", help="Path to save the denoised image.", default="./denoised.png")
parser.add_argument("--add_noise", action='store_true', help="Add artificial Gaussian noise to the image.")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

im = Image.open(args.img_to_denoise)
im = ImageOps.grayscale(im)
img = np.array(im) 

if args.add_noise:
	img_noisy = img + args.sigma * np.random.randn(*img.shape)
else:
	img_noisy = img

img_noisy_torch = torch.from_numpy(img_noisy).view(1, 1, *img_noisy.shape).to(device).float()

# Model
if args.sigma <= 15:
	model = NLRidge(7, 7, 18, 55, 37, 4) 
elif args.sigma <= 35:
	model = NLRidge(9, 9, 18, 90, 37, 4) 
else:
	model = NLRidge(11, 9, 20, 120, 37, 4) 
model.to(device)

# Denoising
t = time.time()
img_denoised_torch = model(img_noisy_torch, args.sigma)
print("Time elapsed:", time.time() - t)
img_denoised = img_denoised_torch.view(*img_noisy.shape).cpu().numpy()
img_denoised = np.clip(img_denoised, 0, 255)

# Performance in PSNR
if args.add_noise:
	print("PSNR:", round(10*np.log10(255**2 / np.mean((img_denoised - img)**2)), 2), "dB")

# Saving
im = Image.fromarray(img_denoised.astype(np.uint8))
im.save(args.img_to_save)





















#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : NL-Ridge
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2022, v1.0.

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def align_corners(x, s, value=0):
    N, C, H, W = x.size()
    if s == 1 or (H % s == 1 and W % s == 1):
        return x
    
    i_pad = (s - (H % s) + 1) % s
    j_pad = (s - (W % s) + 1) % s
    x_pad = nn.functional.pad(x, (0, j_pad, 0, i_pad), mode='constant', value=value)
    
    x_pad[:, :, -1:, :W:s] = x[:, :, -1:, ::s]
    x_pad[:, :, :H:s, -1:] = x[:, :, ::s, -1:]
    x_pad[:, :, -1: , -1:] = x[:, :, -1: , -1:]
    
    if i_pad > 0:
        x_pad[:, :, H-1:H, :W:s] = value
    if j_pad > 0:
        x_pad[:, :, :H:s, W-1:W] = value
    if i_pad > 0 and j_pad > 0:
        x_pad[:, :, H-1:H, W-1:W] = value
    return x_pad

class NLRidge(nn.Module):
    def __init__(self, p1=7, p2=7, m1=18, m2=55, w=37, s=4):
        super(NLRidge, self).__init__()
        self.p1 = p1
        self.p2 = p2
        self.m1 = m1
        self.m2 = m2
        self.window_size = w
        self.step = s
        
    def blockMatching(self, input_x, m, p):
        N, C, H, W = input_x.size() 
        w = self.window_size
        s = self.step 
        
        r = w//2
        x_patches = nn.functional.unfold(input_x, p).view(N, C*p**2, H-p+1, W-p+1)
        x_patches = align_corners(x_patches, s, value=float('inf'))
        x_pad = nn.functional.pad(x_patches, (r, r, r, r), mode='constant', value=float('inf')) 
        x_center = x_patches[:, :, ::s, ::s]
        x_dist = torch.empty(N, w**2, x_center.size(2), x_center.size(3), dtype=input_x.dtype, device=device)

        _, _, H_ext, W_ext = x_patches.size()
        for i in range(w):
            for j in range(w): 
                x_dist[:, i*w+j, :, :] = torch.sum((x_pad[:, :, i:i+H_ext:s, j:j+W_ext:s]-x_center)**2, dim=1)
                
        x_dist[:, r*w+r, :, :] = -float('inf') # to be sure that the central patch will be chosen     
        topk = torch.topk(x_dist, m, dim=1, largest=False, sorted=True)
        indices = topk.indices

        # (ind_rows, ind_cols) is a 2d-representation of indices
        ind_row = torch.div(indices, w, rounding_mode='floor') - w//2 
        ind_col = torch.fmod(indices, w) - w//2
        
        # (row_arange, col_arange) indicates, for each pixel, the number of its row and column
        # example: if we have an image 2x2 -> [[(0, 0), (0,1)], [(1, 0), (1,1)]]
        pix_row = torch.arange(H+w-p, device=device).view(1, 1, H+w-p, 1).repeat(N, m, 1, W+w-p)[:, :, r:H-p+1+r, r:W-p+1+r]
        pix_col = torch.arange(W+w-p, device=device).view(1, 1, 1, W+w-p).repeat(N, m, H+w-p, 1)[:, :, r:H-p+1+r, r:W-p+1+r]
        pix_row = align_corners(pix_row, s)[:, :, ::s, ::s]
        pix_col = align_corners(pix_col, s)[:, :, ::s, ::s]
        
        # (indices_row, indices_col) indicates, for each pixel, the pixel it is pointing to
        indices_row = ind_row + pix_row
        indices_col = ind_col + pix_col

        # back to (H-p+1) x (W-p+1) space
        indices_row = indices_row - r
        indices_col = indices_col - r
        indices_row[indices_row>H-p] = H-p
        indices_col[indices_col>W-p] = W-p

        # from 2d to 1d representation of indices 
        indices = indices_row * (W-p+1) + indices_col
        
        indices = indices.view(N, m, -1)
        indices = indices.transpose(1, 2)
        indices = indices.reshape(N, -1)
        return indices
    
    def aggregation(self, X_hat, weights, indices, input_y, p):
        N, C, H, W = input_y.size() 

        # Replace patches at their own place
        X_hat = X_hat * weights
        X_hat = X_hat.permute(0, 3, 1, 2).reshape(N, C*p**2, -1)
        weights = weights.view(N, 1, -1).repeat(1, C*p**2, 1)
        X_sum = torch.zeros(N, C*p**2, (H-p+1) * (W-p+1), dtype=X_hat.dtype, device=device)
        divisor = torch.zeros(N, C*p**2, (H-p+1) * (W-p+1), dtype=X_hat.dtype, device=device)
        
        for i in range(N):
            X_sum[i, :, :].index_add_(1, indices[i, :], X_hat[i, :, :])
            divisor[i, :, :].index_add_(1, indices[i, :], weights[i, :, :])
 
        # Overlap patches
        xx = nn.functional.fold(X_sum, output_size=(H, W), kernel_size=p)
        divisor = nn.functional.fold(divisor, output_size=(H, W), kernel_size=p)
        return xx / divisor
    
    def group_patches(self, input_y, indices, m, n, p):
        unfold_y = nn.functional.unfold(input_y, p)
        N = unfold_y.size(0)
        Y = torch.gather(unfold_y, dim=2, index=indices.view(N, 1, -1).repeat(1, n, 1))
        Y = Y.transpose(1, 2)
        Y = Y.reshape(N, -1 , m * n)
        Y = Y.view(N, -1 , m, n)
        return Y
         
    def step1(self, input_y, sigma):
        N, C, H, W = input_y.size() 
        p, m = self.p1, self.m1
        
        # Block Matching
        y_block = torch.mean(input_y, dim=1, keepdim=True) # for color
        indices = self.blockMatching(y_block, m, p)
        Y = self.group_patches(input_y, indices, m, C*p**2, p)
        
        # Treat patches
        X_hat, weights = self.denoise1(Y, sigma)
        return self.aggregation(X_hat, weights, indices, input_y, p)
        
    
    def step2(self, input_y, input_x, sigma):
        N, C, H, W = input_y.size() 
        p, m = self.p2, self.m2
        
        # Block Matching
        x_block = torch.mean(input_x, dim=1, keepdim=True) # for color
        indices = self.blockMatching(x_block, m, p)
        Y = self.group_patches(input_y, indices, m, C*p**2, p)
        X = self.group_patches(input_x, indices, m, C*p**2, p)
        
        # Treat patches
        X_hat, weights = self.denoise2(Y, X, sigma)
        return self.aggregation(X_hat, weights, indices, input_y, p)
    
    def denoise1(self, Y, sigma):
        N, B, m, n = Y.size()
        YtY = Y @ Y.transpose(2, 3)
        Im = torch.eye(m, device=device).repeat(N, B, 1, 1)        
        theta = torch.linalg.solve(YtY, YtY - n * sigma**2 * Im).transpose(2, 3) 
        X_hat = theta @ Y  
        weights = 1 / torch.sum(theta**2, dim=3, keepdim=True)
        return X_hat, weights
    
    def denoise2(self, Y, X, sigma):
        N, B, m, n = Y.size()
        XtX = X @ X.transpose(2, 3)
        Im = torch.eye(m, device=device).repeat(N, B, 1, 1)
        theta = torch.linalg.solve(XtX + n * sigma**2 * Im, XtX).transpose(2, 3)        
        X_hat = theta @ Y 
        weights = 1 / torch.sum(theta**2, dim=3, keepdim=True)
        return X_hat, weights
        
    def forward(self, input_y, sigma):
        den1 = self.step1(input_y, sigma)
        den2 = self.step2(input_y, den1, sigma)
        return den2

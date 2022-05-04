#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : NL-Ridge
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2022, v1.0.

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def subsample(x, s):
    if s == 1:
        return x
    N, C, H, W = x.size()
    x_sub = x[:, :, ::s, ::s]
    if H % s == 1 and W % s == 1:
        return x_sub
    else:
        N, C, Hs, Ws = x_sub.size()
        if H % s != 1 and W % s != 1:
            x_sub_f = torch.empty(N, C, Hs+1, Ws+1, dtype=x.dtype, device=device)
            last_row = x[:, :, -1:, ::s]
            last_col = x[:, :, ::s, -1:]
            last_row_col = x[:, :, -1: , -1:]
            x_sub_f[:, :, :Hs, :Ws] = x_sub
            x_sub_f[:, :, Hs:, :Ws] = last_row
            x_sub_f[:, :, :Hs, Ws:] = last_col
            x_sub_f[:, :, -1:, -1:] = last_row_col
        elif H % s != 1:
            x_sub_f = torch.empty(N, C, Hs+1, Ws, dtype=x.dtype, device=device)
            last_row = x[:, :, -1:, ::s]
            x_sub_f[:, :, :Hs, :] = x_sub
            x_sub_f[:, :, Hs:, :] = last_row
        else:
            x_sub_f = torch.empty(N, C, Hs, Ws+1, dtype=x.dtype, device=device)
            last_col = x[:, :, ::s, -1:]
            x_sub_f[:, :, :, :Ws] = x_sub
            x_sub_f[:, :, :Hs, Ws:] = last_col
        return x_sub_f

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
        x_center = nn.functional.unfold(input_x, p)
        x_center = x_center.view(N, C*p**2, H-p+1, W-p+1)
        x_pad = nn.functional.pad(x_center, (r, r, r, r), mode='constant', value=float('inf')) # of size (N, p^2, H-p+1+w-1, W-p+1+w-1) = (N, p^2, H+w-p, W+w-p)
        x_center = subsample(x_center, s)
        x_dist = torch.empty(N, w**2, x_center.size(2), x_center.size(3), dtype=input_x.dtype, device=device)

        for i in range(w):
            for j in range(w): 
                x_ij = x_pad[:, :, i:i+H-p+1, j:j+W-p+1]
                x_ij = subsample(x_ij, s)
                x_dist[:, i*w+j, :, :] = torch.sum((x_ij-x_center)**2, dim=1)
                
        x_dist[:, r*w+r, :, :] = -float('inf') # to be sure that the central patch will be chosen     
        topk = torch.topk(x_dist, m, dim=1, largest=False, sorted=True)
        indices = topk.indices

        # (ind_rows, ind_cols) is a 2d-representation of indices
        # example: from numbers in [0, w**2[ to [-w//2, w//2]^2
        ind_rows = torch.div(indices, w, rounding_mode='floor') - w//2 
        ind_cols = torch.fmod(indices, w) - w//2
        
        # (row_arange, col_arange) indicates, for each pixel, the number of its row and column
        # example: if we have an image 2x2 -> [[(0, 0), (0,1)], [(1, 0), (1,1)]]
        row_arange = torch.arange(H+w-p, device=device).view(1, 1, H+w-p, 1).repeat(N, m, 1, W+w-p)[:, :, r:H-p+1+r, r:W-p+1+r]
        col_arange = torch.arange(W+w-p, device=device).view(1, 1, 1, W+w-p).repeat(N, m, H+w-p, 1)[:, :, r:H-p+1+r, r:W-p+1+r]
        row_arange = subsample(row_arange, s)
        col_arange = subsample(col_arange, s)
        
        # (indices_row, indices_col) indicates, for each pixel, the pixel it is pointing to
        indices_row = ind_rows+row_arange
        indices_col = ind_cols+col_arange

        # Normalization
        indices_row = indices_row - r
        indices_col = indices_col - r

        # indices gives the 1d representation of the number of the patch -> from [0, H+w-p[ x [0, W+w-p[ to [0, (H+w-p)x(W+w-p)[ 
        # we work in an image (H-p+1, W-p+1)
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
        num = nn.functional.fold(X_sum, output_size=(H, W), kernel_size=p)
        divisor = nn.functional.fold(divisor, output_size=(H, W), kernel_size=p)
        return num / divisor
    
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
        p, m= self.p1, self.m1
        
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

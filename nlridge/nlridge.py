#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : NL-Ridge
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2022, v1.0.

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def reflect_unfold(x, r, p):
    reflect_x = nn.functional.pad(x, (r, r, r, r), mode='reflect')
    return nn.functional.unfold(reflect_x, p)

def stack_neigh(x, p, inc):
    """ replace each pixel by the patch of size patch_size x patch_size of 
    its neighboorhood size -> from 1xHxW to p^2 x (H+2*inc) x (W+2*inc) """
    N, C, H, W = x.size() 
    x = reflect_unfold(x, p//2 + inc, p)
    return x.view(N, p**2, H+2*inc, W+2*inc)

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
        
        x = stack_neigh(input_x, p, w//2) # of size N, p^2, H+w-1, W+w-1
        
        x_center = x[:, :, w//2:H+w//2, w//2:W+w//2] # of size N, p^2, H, W
        x_center = x_center[:, :, ::s, ::s]
        x_dist = torch.empty(N, w**2, H, W, device=device)
        x_dist = x_dist[:, :, ::s, ::s]
        for i in range(w):
            for j in range(w): 
                x_other = x[:, :, i:i+H, j:j+W]
                x_other = x_other[:, :, ::s, ::s]
                x_dist[:, i*w+j, :, :] = torch.sum((x_other-x_center)**2, dim=1)
                
        x_dist[:, (w//2)*w+(w//2), :, :] = -1 # to be sure that the central patch will be chosen     
        topk = torch.topk(x_dist, m, dim=1, largest=False, sorted=True)
        indices = topk.indices

        # (ind_rows, ind_cols) is a 2d-representation of indices
        # example: from numbers in [0, w**2[ to [-w//2, w//2]^2
        ind_rows = torch.div(indices, w, rounding_mode='floor') - w//2 
        ind_cols = torch.fmod(indices, w) - w//2
        
        # (row_arange, col_arange) indicates, for each pixel, the number of its row and column
        # example: if we have an image 2x2 -> [[(0, 0), (0,1)], [(1, 0), (1,1)]]
        row_arange = torch.arange(H+w-1, device=device).view(1, 1, H+w-1, 1).repeat(N, m, 1, W+w-1)[:, :, w//2:H+w//2, w//2:W+w//2]
        col_arange = torch.arange(W+w-1, device=device).view(1, 1, 1, W+w-1).repeat(N, m, H+w-1, 1)[:, :, w//2:H+w//2, w//2:W+w//2]
        row_arange = row_arange[:, :, ::s, ::s]
        col_arange = col_arange[:, :, ::s, ::s]
        
        # (indices_row, indices_col) indicates, for each pixel, the pixel it is pointing to
        indices_row = ind_rows+row_arange
        indices_col = ind_cols+col_arange
        
        # indices gives the 1d representation of the number of the patch -> from [0, H+w-1[ x [0, W+w-1[ to [0, (H+w-1)x(W+w-1)[ 
        # we work in an image of size (H+w-1)x(W+w-1)
        indices = indices_row * (W+w-1) + indices_col
        
        indices = indices.view(N, m, -1)
        indices = indices.transpose(1, 2)
        indices = indices.reshape(N, -1)
        return indices
    
    def aggregation(self, Y, weights, indices, input_y, unfold_y, p):
        N, C, H, W = input_y.size() 
        w = self.window_size
        
        # Replace patches at their own place
        Y = Y * weights
        Y = Y.permute(0, 3, 1, 2).reshape(N, C*p**2, -1)
        weights = weights.view(N, 1, -1).repeat(1, C*p**2, 1)
        X_sum = torch.zeros_like(unfold_y)
        divisor = torch.zeros_like(unfold_y)
        
        for i in range(N):
            X_sum[i, :, :].index_add_(1, indices[i, :], Y[i, :, :])
            divisor[i, :, :].index_add_(1, indices[i, :], weights[i, :, :])
 
        # Overlap patches
        xx = nn.functional.fold(X_sum, output_size=(H+w-1+p-1, W+w-1+p-1), kernel_size=p)
        divisor = nn.functional.fold(divisor, output_size=(H+w-1+p-1, W+w-1+p-1), kernel_size=p)
        den_enlarged = xx / divisor
        r = w//2 + p//2
        den = den_enlarged[: , :, r:-r, r:-r]
        return den
    
    def group_patches(self, unfold_y, indices, m, n):
        N = unfold_y.size(0)
        Y = torch.gather(unfold_y, dim=2, index=indices.view(N, 1, -1).repeat(1, n, 1))
        Y = Y.transpose(1, 2)
        Y = Y.reshape(N, -1 , m * n)
        Y = Y.view(N, -1 , m, n)
        return Y
         
    def step1(self, input_y, sigma):
        N, C, H, W = input_y.size() 
        p, m, w = self.p1, self.m1, self.window_size
        
        # Block Matching
        y_block = torch.mean(input_y, dim=1, keepdim=True) # for color
        indices = self.blockMatching(y_block, m, p)
        unfold_y = reflect_unfold(input_y, p//2 + w//2, p)
        Y = self.group_patches(unfold_y, indices, m, C*p**2)
        
        # Treat patches
        X_hat, weights = self.denoise1(Y, sigma)
        return self.aggregation(X_hat, weights, indices, input_y, unfold_y, p)
        
    
    def step2(self, input_y, input_x, sigma):
        N, C, H, W = input_y.size() 
        p, m, w = self.p2, self.m2, self.window_size
        
        # Block Matching
        x_block = torch.mean(input_x, dim=1, keepdim=True) # for color
        indices = self.blockMatching(x_block, m, p)
        unfold_y = reflect_unfold(input_y, p//2 + w//2, p)
        unfold_x = reflect_unfold(input_x, p//2 + w//2, p)
        Y = self.group_patches(unfold_y, indices, m, C*p**2)
        X = self.group_patches(unfold_x, indices, m, C*p**2)
        
        # Treat patches
        X_hat, weights = self.denoise2(Y, X, sigma)
        return self.aggregation(X_hat, weights, indices, input_y, unfold_y, p)
    
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

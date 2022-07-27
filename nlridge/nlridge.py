#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : NL-Ridge
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2022, v1.0.

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NLRidge(nn.Module):
    def __init__(self, p1=7, p2=7, m1=18, m2=55, w=37, s=4):
        super(NLRidge, self).__init__()
        self.p1 = p1
        self.p2 = p2
        self.m1 = m1
        self.m2 = m2
        self.window_size = w
        self.step = s
        
    @staticmethod    
    def block_matching(input_x, m, p, w, s):
        def align_corners(x, s, value=0):
            N, C, H, W = x.size()
            i, j = (s - (H % s) + 1) % s, (s - (W % s) + 1) % s
            x = F.pad(x, (0, j, 0, i), mode='constant', value=value)
            x[:, :, [H-1, H-1+i], :W-1:s] = x[:, :, [H-1+i, H-1], :W-1:s]
            x[:, :, :H-1:s, [W-1, W-1+j]] = x[:, :, :H-1:s, [W-1+j, W-1]]
            x[:, :, [H-1, H-1+i], [W-1, W-1+j]] = x[:, :, [H-1+i, H-1], [W-1+j, W-1]]
            return x
        
        v = w // 2
        N, C, H, W = input_x.size() 
        patches = F.unfold(input_x, p).view(N, C*p**2, H-p+1, W-p+1)
        patches = align_corners(patches, s, value=float('inf'))
        _, _, k, l = patches.size()
        ref_patches = patches[:, :, ::s, ::s].permute(0, 2, 3, 1).contiguous()
        pad_patches = F.pad(patches, (v, v, v, v), mode='constant', value=float('inf')).permute(0, 2, 3, 1).contiguous()
        dist = torch.empty(N, w**2, ref_patches.size(1), ref_patches.size(2), dtype=input_x.dtype, device=device)
        
        for i in range(w):
            for j in range(w): 
                dist[:, i*w+j, :, :] = F.pairwise_distance(pad_patches[:, i:k+i:s, j:l+j:s, :], ref_patches)
                
        dist[:, v*w+v, :, :] = -float('inf') # to be sure that the reference patch will be chosen     
        indices = torch.topk(dist, m, dim=1, largest=False, sorted=False).indices

        # (ind_row, ind_col) is a 2d-representation of indices
        ind_row = torch.div(indices, w, rounding_mode='floor') - v
        ind_col = torch.fmod(indices, w) - v
        
        # (ind_row_ref, ind_col_ref) indicates, for each reference patch, the indice of its row and column
        ind_row_ref = align_corners(torch.arange(H-p+1, device=device).view(1, 1, -1, 1), s)[:, :, ::s, :]
        ind_col_ref = align_corners(torch.arange(W-p+1, device=device).view(1, 1, 1, -1), s)[:, :, :, ::s]
        ind_row_ref = ind_row_ref.expand(N, m, -1, ind_col_ref.size(3))
        ind_col_ref = ind_col_ref.expand(N, m, ind_row_ref.size(2), -1)
        
        # (indices_row, indices_col) indicates, for each reference patch, the indices of its most similar patches 
        indices_row = (ind_row_ref + ind_row).clip(max=H-p)
        indices_col = (ind_col_ref + ind_col).clip(max=W-p)

        # from 2d to 1d representation of indices 
        indices = (indices_row * (W-p+1) + indices_col).view(N, m, -1).transpose(1, 2).reshape(N, -1)
        return indices
    
    @staticmethod 
    def group_patches(input_y, indices, m, n, p):
        N = input_y.size(0)
        unfold_y = F.unfold(input_y, p)
        Y = torch.gather(unfold_y, dim=2, index=indices.view(N, 1, -1).expand(-1, n, -1))
        Y = Y.transpose(1, 2)
        Y = Y.view(N, -1, m, n)
        return Y
    
    @staticmethod 
    def denoise1(Y, sigma):
        N, B, m, n = Y.size()
        YtY = Y @ Y.transpose(2, 3)
        Im = torch.eye(m, dtype=Y.dtype, device=device).expand(N, B, -1, -1)      
        theta = torch.cholesky_solve(YtY - n * sigma**2 * Im, torch.linalg.cholesky(YtY))
        X_hat = theta @ Y  
        weights = 1 / torch.sum(theta**2, dim=3, keepdim=True).clip(1/m, 1)
        return X_hat, weights
    
    @staticmethod 
    def denoise2(Y, X, sigma):
        N, B, m, n = Y.size()
        XtX = X @ X.transpose(2, 3)
        Im = torch.eye(m, dtype=Y.dtype, device=device).expand(N, B, -1, -1)
        theta = torch.cholesky_solve(XtX, torch.linalg.cholesky(XtX + n * sigma**2 * Im))
        X_hat = theta @ Y 
        weights = 1 / torch.sum(theta**2, dim=3, keepdim=True).clip(1/m, 1)
        return X_hat, weights
    
    @staticmethod 
    def aggregation(X_hat, weights, indices, size, p):
        N, C, H, W = size 
        X_hat = (X_hat * weights).permute(0, 3, 1, 2).view(N, C*p**2, -1)
        weights = weights.view(N, 1, -1).expand(-1, C*p**2, -1)
        X_sum = torch.zeros(N, C*p**2, (H-p+1) * (W-p+1), dtype=X_hat.dtype, device=device)
        weights_sum = torch.zeros_like(X_sum)
        
        for i in range(N):
            X_sum[i, :, :].index_add_(1, indices[i, :], X_hat[i, :, :])
            weights_sum[i, :, :].index_add_(1, indices[i, :], weights[i, :, :])
 
        return F.fold(X_sum, (H, W), p) / F.fold(weights_sum, (H, W), p)
         
    def step1(self, input_y, sigma):
        N, C, H, W = input_y.size() 
        m, p, w, s = self.m1, self.p1, self.window_size, self.step
        y_mean = torch.mean(input_y, dim=1, keepdim=True) # for color
        indices = self.block_matching(y_mean, m, p, w, s)
        Y = self.group_patches(input_y, indices, m, C*p**2, p)
        X_hat, weights = self.denoise1(Y, sigma)
        x_hat = self.aggregation(X_hat, weights, indices, input_y.size(), p)
        return x_hat
        
    def step2(self, input_y, input_x, sigma):
        N, C, H, W = input_y.size() 
        m, p, w, s = self.m2, self.p2, self.window_size, self.step
        x_mean = torch.mean(input_x, dim=1, keepdim=True) # for color
        indices = self.block_matching(x_mean, m, p, w, s)
        Y = self.group_patches(input_y, indices, m, C*p**2, p)
        X = self.group_patches(input_x, indices, m, C*p**2, p)
        X_hat, weights = self.denoise2(Y, X, sigma)
        x_hat = self.aggregation(X_hat, weights, indices, input_y.size(), p)
        return x_hat
        
    def forward(self, input_y, sigma):
        den1 = self.step1(input_y, sigma)
        den2 = self.step2(input_y, den1, sigma)
        return den2

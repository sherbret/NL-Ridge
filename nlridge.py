#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : NL-Ridge
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2022, v1.0.

import torch
import torch.nn as nn
import torch.nn.functional as F

class NLRidge(nn.Module):
    def __init__(self, p1=7, p2=7, k1=18, k2=55, w=37, s=4, affine=True):
        super(NLRidge, self).__init__()
        self.p1 = p1 # patch size for step 1
        self.p2 = p2 # patch size for step 2
        self.k1 = k1 # group size for step 1
        self.k2 = k2 # group size for step 2
        self.window = w # size of the window centered around reference patches within which similar patches are searched (odd number)
        self.step = s # moving step size from one reference patch to another
        self.affine = affine # ensure affine combinations of patches 

    @staticmethod    
    def block_matching(input_x, k, p, w, s):
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
        ref_patches = patches[:, :, ::s, ::s].permute(0, 2, 3, 1).contiguous()
        pad_patches = F.pad(patches, (v, v, v, v), mode='constant', value=float('inf')).permute(0, 2, 3, 1).contiguous()
        hr, wr, hp, wp = ref_patches.size(1), ref_patches.size(2), patches.size(2), patches.size(3)
        dist = torch.empty(N, w**2, hr, wr, dtype=input_x.dtype, device=input_x.device)
        
        for i in range(w):
            for j in range(w): 
                if i != v or j != v: 
                    dist[:, i*w+j, :, :] = F.pairwise_distance(pad_patches[:, i:i+hp:s, j:j+wp:s, :], ref_patches)
                
        dist[:, v*w+v, :, :] = -float('inf') # the similarity matrices include the reference patch     
        indices = torch.topk(dist, k, dim=1, largest=False, sorted=False).indices

        # (ind_row, ind_col) is a 2d-representation of indices
        ind_row = torch.div(indices, w, rounding_mode='floor') - v
        ind_col = torch.fmod(indices, w) - v
        
        # (ind_row_ref, ind_col_ref) indicates, for each reference patch, the indice of its row and column
        ind_row_ref = align_corners(torch.arange(H-p+1, device=input_x.device).view(1, 1, -1, 1), s)[:, :, ::s, :].expand(N, k, -1, wr)
        ind_col_ref = align_corners(torch.arange(W-p+1, device=input_x.device).view(1, 1, 1, -1), s)[:, :, :, ::s].expand(N, k, hr, -1)
        
        # (indices_row, indices_col) indicates, for each reference patch, the indices of its most similar patches 
        indices_row = (ind_row_ref + ind_row).clip(max=H-p)
        indices_col = (ind_col_ref + ind_col).clip(max=W-p)

        # from 2d to 1d representation of indices 
        indices = (indices_row * (W-p+1) + indices_col).view(N, k, -1).transpose(1, 2).reshape(N, -1)
        return indices
    
    @staticmethod 
    def gather_groups(input_y, indices, k, p):
        unfold_y = F.unfold(input_y, p)
        N, n, _ = unfold_y.size()
        Y = torch.gather(unfold_y, dim=2, index=indices.view(N, 1, -1).expand(-1, n, -1)).transpose(1, 2).view(N, -1, k, n)
        return Y
    
    def variance_groups(self, X, noise_type, sigma, a_pois, b_pois, indices, k, p):
        if noise_type=="gaussian-homoscedastic":
            V = sigma**2 * torch.ones(X.size(0), 1, k, p**2, dtype=X.dtype, device=X.device)
        elif noise_type=="gaussian-heteroscedastic":
            V = self.gather_groups(sigma**2, indices, k, p)
        elif noise_type=="poisson":
            V = X
        elif noise_type=="poisson-gaussian":
            V = a_pois * X + b_pois
        else:
            raise ValueError('noise_type must be either gaussian-homoscedastic, gaussian-heteroscedastic, poisson or poisson-gaussian.')
        return V
    
    def denoise1(self, Y, V):
        N, B, k, n = Y.size()
        YtY = Y @ Y.transpose(2, 3)
        D = torch.sum(V, dim=3)
        Ik = torch.eye(k, dtype=Y.dtype, device=Y.device).expand(N, B, -1, -1)  
        S = torch.cholesky_solve(Ik, torch.linalg.cholesky(YtY))
        if self.affine:
            S1 = torch.sum(S, dim=3, keepdim=True)
            S_sum = torch.sum(S1, dim=2, keepdim=True)
            theta = Ik - (S - S1 @ S1.transpose(2, 3) / S_sum) * D.unsqueeze(3)
        else:
            theta = Ik - S * D.unsqueeze(3)
        theta = theta.transpose(2,3)
        X_hat = theta @ Y 
        weights = 1 / torch.sum(theta**2, dim=3, keepdim=True).clip(1/k, 1)
        return X_hat, weights
    
    def denoise2(self, Y, X, V):
        N, B, k, n = Y.size()
        XtX = X @ X.transpose(2, 3)
        D = torch.sum(V, dim=3)
        Ik = torch.eye(k, dtype=Y.dtype, device=Y.device).expand(N, B, -1, -1)  
        S = torch.cholesky_solve(Ik, torch.linalg.cholesky(XtX + torch.diag_embed(D)))
        if self.affine:
            S1 = torch.sum(S, dim=3, keepdim=True)
            S_sum = torch.sum(S1, dim=2, keepdim=True)
            theta = Ik - (S - S1 @ S1.transpose(2, 3) / S_sum) * D.unsqueeze(3)
        else:
            theta = Ik - S * D.unsqueeze(3)
        theta = theta.transpose(2,3)
        X_hat = theta @ Y
        weights = 1 / torch.sum(theta**2, dim=3, keepdim=True).clip(1/k, 1)
        return X_hat, weights
    
    @staticmethod 
    def aggregate(X_hat, weights, indices, H, W, p):
        N, _, _, n = X_hat.size()
        X = (X_hat * weights).permute(0, 3, 1, 2).view(N, n, -1)
        weights = weights.view(N, 1, -1).expand(-1, n, -1)
        X_sum = torch.zeros(N, n, (H-p+1) * (W-p+1), dtype=X.dtype, device=X.device)
        weights_sum = torch.zeros_like(X_sum)
        
        for i in range(N):
            X_sum[i, :, :].index_add_(1, indices[i, :], X[i, :, :])
            weights_sum[i, :, :].index_add_(1, indices[i, :], weights[i, :, :])
 
        return F.fold(X_sum, (H, W), p) / F.fold(weights_sum, (H, W), p)
         
    def step1(self, input_y, noise_type, sigma, a_pois, b_pois):
        _, _, H, W = input_y.size() 
        k, p, w, s = self.k1, self.p1, self.window, self.step
        y_mean = torch.mean(input_y, dim=1, keepdim=True) # for color
        indices = self.block_matching(y_mean, k, p, w, s)
        Y = self.gather_groups(input_y, indices, k, p)
        V = self.variance_groups(Y, noise_type, sigma, a_pois, b_pois, indices, k, p)
        X_hat, weights = self.denoise1(Y, V)
        x_hat = self.aggregate(X_hat, weights, indices, H, W, p)
        return x_hat
        
    def step2(self, input_y, input_x, noise_type, sigma, a_pois, b_pois):
        _, _, H, W = input_y.size()
        k, p, w, s = self.k2, self.p2, self.window, self.step
        x_mean = torch.mean(input_x, dim=1, keepdim=True) # for color
        indices = self.block_matching(x_mean, k, p, w, s)
        Y = self.gather_groups(input_y, indices, k, p)
        X = self.gather_groups(input_x, indices, k, p)
        V = self.variance_groups(X, noise_type, sigma, a_pois, b_pois, indices, k, p)
        X_hat, weights = self.denoise2(Y, X, V)
        x_hat = self.aggregate(X_hat, weights, indices, H, W, p)
        return x_hat
        
    def forward(self, input_y, noise_type="gaussian-homoscedastic", sigma=25.0, a_pois=1.0, b_pois=0.0):
        den1 = self.step1(input_y, noise_type, sigma, a_pois, b_pois)
        den2 = self.step2(input_y, den1, noise_type, sigma, a_pois, b_pois)
        return den2

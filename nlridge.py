#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : NL-Ridge
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2022, v1.0.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.transforms.functional import rgb_to_grayscale

class NLRidge(nn.Module):
    def __init__(self):
        super().__init__()
        self.set_parameters()

    def set_parameters(self, sigma=25.0, a_pois=1.0, b_pois=0.0,\
                        noise_type='gaussian-homoscedastic', \
                        p1=7, p2=7, k1=18, k2=55, w=37, s=4,\
                        constraints='linear'):
        self.noise_type = noise_type # either gaussian-homoscedastic, gaussian-heteroscedastic, poisson or poisson-gaussian.
        self.sigma = sigma # sigma parameter for Gaussian noise
        self.a_pois = a_pois # a parameter for Poisson-Gaussian noise
        self.b_pois = b_pois # b parameter for Poisson-Gaussian noise
        self.p1 = p1 # patch size for step 1
        self.p2 = p2 # patch size for step 2
        self.k1 = k1 # group size for step 1
        self.k2 = k2 # group size for step 2
        self.window = w # size of the window centered around reference patches within which similar patches are searched (odd number)
        self.step = s # moving step size from one reference patch to another
        self.constraints = constraints # either 'linear', 'affine', 'conical' or 'convex'
         
    @staticmethod
    def block_matching(input_x, k, p, w, s):
        if input_x.device == torch.device("cuda:0"): 
            input_x = input_x.half()
            
            @staticmethod
    def block_matching(input_x, k, p, w, s):
        if input_x.device == torch.device("cuda:0"): 
            input_x = input_x.half()
            
        def block_matching_aux(input_x, input_x_pad, k, p, v, s):
            N, C, H, W = input_x.size() 
            assert C == 1
            w_large = 2*v + p
            Href, Wref = -((H - p + 1) // -s), -((W - p + 1) // -s) # ceiling division, represents the number of reference patches along each axis for unfold with stride=s
            ref_patches = F.unfold(input_x, p, stride=s)
            ref_patches = rearrange(ref_patches, 'n (p1 p2) l -> (n l) 1 p1 p2', p1=p)
            local_windows = F.unfold(input_x_pad, w_large, stride=s)
            local_windows = rearrange(local_windows, 'n (p1 p2) l -> 1 (n l) p1 p2', p1=w_large)
            scalar_product = F.conv2d(local_windows, ref_patches / p**2, groups=N*Href*Wref)
            norm_patches = F.avg_pool2d(local_windows**2, p, stride=1)
            distances = norm_patches - 2 * scalar_product # (up to a constant)
            distances[:, :, v, v] = -float('inf') # the reference patch is always taken
            distances = rearrange(distances, '1 (n h w) p1 p2 -> n h w (p1 p2)', n=N, h=Href, w=Wref)
            indices = torch.topk(distances, k, dim=3, largest=False, sorted=False).indices # float('nan') is considered to be the highest value for topk 
            return indices

        v = w // 2
        w_large = 2*v + p
        input_x_pad = F.pad(input_x, [v]*4, mode='constant', value=float('nan'))
        N, C, H, W = input_x.size() 
        Href, Wref = -((H - p + 1) // -s), -((W - p + 1) // -s) # ceiling division, represents the number of reference patches along each axis for unfold with stride=s
        ind_H_ref = torch.arange(0, H-p+1, step=s, device=input_x.device)      
        ind_W_ref = torch.arange(0, W-p+1, step=s, device=input_x.device)
        if (H - p + 1) % s != 1:
            ind_H_ref = torch.cat((ind_H_ref, torch.tensor([H - p], device=input_x.device)), dim=0)
        if (W - p + 1) % s != 1:
            ind_W_ref = torch.cat((ind_W_ref, torch.tensor([W - p], device=input_x.device)), dim=0)
            
        indices = torch.empty(N, ind_H_ref.size(0), ind_W_ref.size(0), k, dtype=ind_H_ref.dtype, device=ind_H_ref.device)
        indices[:, :Href, :Wref, :] = block_matching_aux(input_x, input_x_pad, k, p, v, s)
        if (H - p + 1) % s != 1:
            indices[:, Href:, :Wref, :] = block_matching_aux(input_x[:, :, -p:, :], input_x_pad[:, :, -w_large:, :], k, p, v, s)
        if (W - p + 1) % s != 1:
            indices[:, :Href, Wref:, :] = block_matching_aux(input_x[:, :, :, -p:], input_x_pad[:, :, :, -w_large:], k, p, v, s)
            if (H - p + 1) % s != 1:
                indices[:, Href:, Wref:, :] = block_matching_aux(input_x[:, :, -p:, -p:], input_x_pad[:, :, -w_large:, -w_large:], k, p, v, s)
                
        # (ind_row, ind_col) is a 2d-representation of indices
        ind_row = torch.div(indices, 2*v+1, rounding_mode='floor') - v
        ind_col = torch.fmod(indices, 2*v+1) - v
        
        # from 2d to 1d representation of indices 
        indices = (ind_row + rearrange(ind_H_ref, 'h -> 1 h 1 1')) * (W-p+1) + (ind_col + rearrange(ind_W_ref, 'w -> 1 1 w 1'))
        return rearrange(indices, 'n h w k -> n (h w k)', n=N)
    
    @staticmethod 
    def gather_groups(input_y, indices, k, p):
        unfold_Y = F.unfold(input_y, p)
        N, n, l = unfold_Y.shape
        Y = torch.gather(unfold_Y, dim=2, index=repeat(indices, 'N l -> N n l', n=n))
        return rearrange(Y, 'N n (l k) -> N l k n', k=k)
    
    @staticmethod 
    def aggregate(X_hat, weights, indices, H, W, p):
        N, _, _, n = X_hat.size()
        X = rearrange(X_hat * weights, 'n l k p2 -> n p2 (l k)')
        weights = repeat(weights, 'N l k 1 -> N n (l k)', n=n)
        X_sum = torch.zeros(N, n, (H-p+1) * (W-p+1), dtype=X.dtype, device=X.device)
        weights_sum = torch.zeros_like(X_sum)
        
        for i in range(N):
            X_sum[i, :, :].index_add_(1, indices[i, :], X[i, :, :])
            weights_sum[i, :, :].index_add_(1, indices[i, :], weights[i, :, :])
 
        return F.fold(X_sum, (H, W), p) / F.fold(weights_sum, (H, W), p)
    
    def variance_groups(self, X, indices, k, p):
        if self.noise_type=='gaussian-homoscedastic':
            N, _, k, p = X.size()
            V = self.sigma**2 * torch.ones(N, 1, k, p, dtype=X.dtype, device=X.device)
        elif self.noise_type=='gaussian-heteroscedastic':
            V = self.gather_groups(self.sigma**2, indices, k, p)
        elif self.noise_type=='poisson':
            V = X
        elif self.noise_type=='poisson-gaussian':
            V = self.a_pois * X + self.b_pois
        else:
            raise ValueError('noise_type must be either gaussian-homoscedastic, gaussian-heteroscedastic, poisson or poisson-gaussian.')
        return V
    
    def compute_theta(self, Q, D):
        N, B, k, _ = Q.size()
        if self.constraints == 'linear' or self.constraints == 'affine':
            Ik = torch.eye(k, dtype=Q.dtype, device=Q.device).expand(N, B, -1, -1)
            Qinv = torch.inverse(Q) 
            if self.constraints == 'linear':
                theta = Ik - Qinv * D.unsqueeze(-1)
            else:
                Qinv1 = torch.sum(Qinv, dim=3, keepdim=True)
                Qinv2 = torch.sum(Qinv1, dim=2, keepdim=True)
                theta = Ik - (Qinv - Qinv1 @ Qinv1.transpose(2, 3) / Qinv2) * D.unsqueeze(-1)
        elif self.constraints == 'conical' or self.constraints == 'convex':
            # Coordinate descent algorithm
            N, B, k, _ = Q.size()
            C = torch.diag_embed(D) - Q
            theta = torch.ones_like(Q) / k
            for _ in range(1000):
                for i in range(k): 
                    if self.constraints == 'conical':
                        alpha = -(Q[:, :, i:i+1, :] @ theta + C[:, :, i:i+1, :]) / Q[:, :, i:i+1, i:i+1]
                        alpha = alpha.clip(min=-theta[:, :, i:i+1, :])
                        theta[:, :, i:i+1, :] += alpha
                    elif self.constraints == 'convex':
                        j = (i + int(torch.randint(low=1, high=k, size=(1,)))) % k
                        alpha = -((Q[:, :, i:i+1, :] - Q[:, :, j:j+1, :]) @ theta + C[:, :, i:i+1, :] - C[:, :, j:j+1, :]) /\
                                (Q[:, :, i:i+1, i:i+1] + Q[:, :, j:j+1, j:j+1] - 2 * Q[:, :, i:i+1, j:j+1])
                        alpha = alpha.clip(min=-theta[:, :, i:i+1, :], max=theta[:, :, j:j+1, :])
                        theta[:, :, i:i+1, :] += alpha
                        theta[:, :, j:j+1, :] -= alpha
        else:
            raise ValueError('constraints must be either linear, affine, conical or convex.')
        return theta.transpose(2,3)
 
    def denoise1(self, Y, V):
        N, B, k, n = Y.size()
        D = torch.sum(V, dim=3)
        Q = Y @ Y.transpose(2, 3)
        theta = self.compute_theta(Q, D)
        X_hat = theta @ Y 
        weights = 1 / torch.sum(theta**2, dim=3, keepdim=True).clip(1/k, 1)
        return X_hat, weights
    
    def denoise2(self, Y, X, V):
        N, B, k, n = Y.size()
        D = torch.sum(V, dim=3)
        Q = X @ X.transpose(2, 3) + torch.diag_embed(D)
        theta = self.compute_theta(Q, D)
        X_hat = theta @ Y
        weights = 1 / torch.sum(theta**2, dim=3, keepdim=True).clip(1/k, 1)
        return X_hat, weights
         
    def step1(self, input_y):
        _, C, H, W = input_y.size() 
        k, p, w, s = self.k1, self.p1, self.window, self.step
        y_grayscale = rgb_to_grayscale(input_y) if C == 3 else input_y
        indices = self.block_matching(y_grayscale, k, p, w, s)
        Y = self.gather_groups(input_y, indices, k, p)
        V = self.variance_groups(Y, indices, k, p)
        X_hat, weights = self.denoise1(Y, V)
        x_hat = self.aggregate(X_hat, weights, indices, H, W, p)
        return x_hat
        
    def step2(self, input_y, input_x):
        _, C, H, W = input_y.size()
        k, p, w, s = self.k2, self.p2, self.window, self.step
        x_grayscale = rgb_to_grayscale(input_x) if C == 3 else input_x
        indices = self.block_matching(x_grayscale, k, p, w, s)
        Y = self.gather_groups(input_y, indices, k, p)
        X = self.gather_groups(input_x, indices, k, p)
        V = self.variance_groups(X, indices, k, p)
        X_hat, weights = self.denoise2(Y, X, V)
        x_hat = self.aggregate(X_hat, weights, indices, H, W, p)
        return x_hat
        
    def forward(self, input_y, sigma=25.0, a_pois=1.0, b_pois=0.0,\
                        noise_type='gaussian-homoscedastic',\
                        p1=7, p2=7, k1=18, k2=55, w=37, s=4,\
                        constraints='linear'):
        # Recommended parameters for additive white Gaussian noise:
            # 0 < σ ≤ 15: p1=7, p2=7, k1=18, k2=55
            # 15 < σ ≤ 35: p1=9, p2=9, k1=18, k2=90
            # 35 < σ ≤ 50: p1=11, p2=9, k1=20, k2=120
        self.set_parameters(sigma, a_pois, b_pois, noise_type, p1, p2, k1, k2, w, s, constraints)
        den1 = self.step1(input_y)
        den2 = self.step2(input_y, den1)
        return den2

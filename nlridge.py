#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : NL-Ridge
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2022, v1.0.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class NLRidge(nn.Module):
    """ A PyTorch module implementing NL-Ridge denoising algorithm. """

    def __init__(self):
        """ Initializes the NL-Ridge module. """
        super().__init__()
         
    @staticmethod
    def block_matching(input_x, k, p, w, s):
        """
        Finds similar patches within a specified window around each reference patch.

        Args:
            input_x (torch.FloatTensor): Input image tensor of shape (N, C, H, W).
            k (int): Number of most similar patches to find.
            p (int): Patch size.
            w (int): Search window size.
            s (int): Stride for moving between reference patches.

        Returns:
            torch.LongTensor: Indices of shape (N, Href, Wref, k) of similar patches for each reference patch.
        """
            
        def block_matching_aux(input_x_pad, k, p, v, s):
            """
            Auxiliary function to perform block matching in a padded input tensor.

            Args:
                input_x_pad (torch.FloatTensor): Padded input tensor of shape (N, C, H, W).
                k (int): Number of similar patches to find.
                p (int): Patch size.
                v (int): Half of the search window size.
                s (int): Stride for moving between reference patches.

            Returns:
                torch.LongTensor: Indices of shape (N, Href, Wref, k) of similar patches for each reference patch.
            """
            N, C, H, W = input_x_pad.size() 
            assert C == 1
            Href, Wref = -((H - (2*v+p) + 1) // -s), -((W - (2*v+p) + 1) // -s) # ceiling division, represents the number of reference patches along each axis for unfold with stride=s
            norm_patches = F.avg_pool2d(input_x_pad**2, p, stride=1)
            norm_patches = F.unfold(norm_patches, 2*v+1, stride=s)
            norm_patches = rearrange(norm_patches, 'n (p1 p2) l -> 1 (n l) p1 p2', p1=2*v+1)
            local_windows = F.unfold(input_x_pad, 2*v+p, stride=s) / p
            local_windows = rearrange(local_windows, 'n (p1 p2) l -> 1 (n l) p1 p2', p1=2*v+p)
            ref_patches = rearrange(local_windows[..., v:-v, v:-v], '1 b p1 p2 -> b 1 p1 p2')
            scalar_product = F.conv2d(local_windows, ref_patches, groups=N*Href*Wref)
            distances = norm_patches - 2 * scalar_product # (up to a constant)
            distances[:, :, v, v] = float('-inf') # the reference patch is always taken
            distances = rearrange(distances, '1 (n h w) p1 p2 -> n h w (p1 p2)', n=N, h=Href, w=Wref)
            indices = torch.topk(distances, k, dim=-1, largest=False, sorted=False).indices # float('nan') is considered to be the highest value for topk 
            return indices

        v = w // 2
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
        indices[:, :Href, :Wref, :] = block_matching_aux(input_x_pad, k, p, v, s)
        if (H - p + 1) % s != 1:
            indices[:, Href:, :Wref, :] = block_matching_aux(input_x_pad[:, :, -(2*v + p):, :], k, p, v, s)
        if (W - p + 1) % s != 1:
            indices[:, :Href, Wref:, :] = block_matching_aux(input_x_pad[:, :, :, -(2*v + p):], k, p, v, s)
            if (H - p + 1) % s != 1:
                indices[:, Href:, Wref:, :] = block_matching_aux(input_x_pad[:, :, -(2*v + p):, -(2*v + p):], k, p, v, s)
                
        # (ind_row, ind_col) is a 2d-representation of indices
        ind_row = torch.div(indices, 2*v+1, rounding_mode='floor') - v
        ind_col = torch.fmod(indices, 2*v+1) - v
        
        # from 2d to 1d representation of indices 
        indices = (ind_row + rearrange(ind_H_ref, 'h -> 1 h 1 1')) * (W-p+1) + (ind_col + rearrange(ind_W_ref, 'w -> 1 1 w 1'))
        return indices
    
    @staticmethod
    def gather_groups(input_y, indices, p):
        """
        Gathers groups of patches based on the indices from block-matching.

        Args:
            input_y (torch.FloatTensor): Input image tensor of shape (N, C, H, W).
            indices (torch.LongTensor): Indices of similar patches of shape (N, Href, Wref, k).
            k (int): Number of similar patches.
            p (int): Patch size.

        Returns:
            torch.FloatTensor: Grouped patches of shape (N, Href, Wref, k, p**2).
        """
        unfold_Y = F.unfold(input_y, p)
        _, n, _ = unfold_Y.shape
        _, Href, Wref, k = indices.shape
        Y = torch.gather(unfold_Y, dim=2, index=repeat(indices, 'N h w k -> N n (h w k)', n=n))
        return rearrange(Y, 'N n (h w k) -> N h w k n', k=k, h=Href, w=Wref)
    
    @staticmethod 
    def aggregate(X_hat, weights, indices, H, W, p):
        """
        Aggregates groups of patches back into the image grid.

        Args:
            X_hat (torch.FloatTensor): Grouped denoised patches of shape (N, Href, Wref, k, p**2).
            weights (torch.FloatTensor): Weights of each patch of shape (N, Href, Wref, k, 1).
            indices (torch.LongTensor): Indices of the patches in the original image of shape (N, Href, Wref, k).  
            H (int): Height of the original image.
            W (int): Width of the original image.
            p (int): Patch size.

        Returns:
            torch.FloatTensor: Reconstructed image tensor.
        """
        N, _, _, _, n = X_hat.size()
        X = rearrange(X_hat * weights, 'N h w k n -> (N h w k) n')
        weights = repeat(weights, 'N h w k 1 -> (N h w k) n', n=n)
        offset = (H-p+1) * (W-p+1) * torch.arange(N, device=X.device).view(-1, 1, 1, 1)
        indices = rearrange(indices + offset, 'N h w k -> (N h w k)')

        X_sum = torch.zeros(N * (H-p+1) * (W-p+1), n, dtype=X.dtype, device=X.device)
        weights_sum = torch.zeros_like(X_sum)

        X_sum.index_add_(0, indices, X)
        weights_sum.index_add_(0, indices, weights)
        X_sum = rearrange(X_sum, '(N hw) n -> N n hw', N=N)
        weights_sum = rearrange(weights_sum, '(N hw) n -> N n hw', N=N)

        return F.fold(X_sum, (H, W), p) / F.fold(weights_sum, (H, W), p)
    
    def variance_groups(self, X, indices, p):
        """
        Computes the variance per pixel.

        Args:
            X (torch.FloatTensor): Grouped patches of shape (N, Href, Wref, k, n).
            indices (torch.LongTensor): Indices of the patches in the original image of shape (N, Href, Wref, k).  
            p (int): Patch size.

        Returns:
            V (torch.FloatTensor): Variance per pixel (*, k, n).
        """
        if self.noise_type=='gaussian-homoscedastic':
            N, _, _, k, n = X.shape 
            if isinstance(self.sigma, torch.Tensor): self.sigma = self.sigma.view(N, 1, 1, 1, 1) 
            V = self.sigma**2 * torch.ones(1, 1, 1, k, n, dtype=X.dtype, device=X.device)
        elif self.noise_type=='gaussian-heteroscedastic':
            V = self.gather_groups(self.sigma**2, indices, p)
        elif self.noise_type=='poisson':
            V = X
        elif self.noise_type=='poisson-gaussian':
            V = self.a_pois * X + self.b_pois
        else:
            raise ValueError('noise_type must be either gaussian-homoscedastic, gaussian-heteroscedastic, poisson or poisson-gaussian.')
        return V
    
    def compute_theta(self, Q, D):
        """
        Computes the theta matrix based on the provided constraints.
    
        Args:
            Q (torch.FloatTensor): Q matrix, shape (N, Href, Wref, k, k).
            D (torch.FloatTensor): Diagonal matrix, shape (*, k, 1).
    
        Returns:
            torch.FloatTensor: Theta matrix, shape (N, Href, Wref, k, k).
        """
        k = Q.size(-2)
        if self.constraints == 'linear' or self.constraints == 'affine':
            Ik = torch.eye(k, dtype=Q.dtype, device=Q.device)
            Qinv = torch.inverse(Q) 
            if self.constraints == 'linear':
                theta = Ik - Qinv * D
            else:
                Qinv1 = torch.sum(Qinv, dim=-1, keepdim=True)
                Qinv2 = torch.sum(Qinv1, dim=-2, keepdim=True)
                theta = Ik - (Qinv - Qinv1 @ Qinv1.transpose(-2, -1) / Qinv2) * D
        elif self.constraints == 'conical' or self.constraints == 'convex':
            # Coordinate descent algorithm
            C = torch.diag_embed(D.squeeze(-1)) - Q
            theta = torch.ones_like(Q) / k
            for _ in range(1000):
                for i in range(k): 
                    if self.constraints == 'conical':
                        alpha = -(Q[..., i:i+1, :] @ theta + C[..., i:i+1, :]) / Q[..., i:i+1, i:i+1]
                        alpha = alpha.clip(min=-theta[..., i:i+1, :])
                        theta[..., i:i+1, :] += alpha
                    elif self.constraints == 'convex':
                        j = (i + int(torch.randint(low=1, high=k, size=(1,)))) % k
                        alpha = -((Q[..., i:i+1, :] - Q[..., j:j+1, :]) @ theta + C[..., i:i+1, :] - C[..., j:j+1, :]) /\
                                (Q[..., i:i+1, i:i+1] + Q[..., j:j+1, j:j+1] - 2 * Q[..., i:i+1, j:j+1])
                        alpha = alpha.clip(min=-theta[..., i:i+1, :], max=theta[..., j:j+1, :])
                        theta[..., i:i+1, :] += alpha
                        theta[..., j:j+1, :] -= alpha
        else:
            raise ValueError('constraints must be either linear, affine, conical or convex.')
        return theta.transpose(-2, -1)
 
    def denoise1(self, Y, V):
        """
        Denoises each group of similar patches (step 1).
    
        Args:
            Y (torch.FloatTensor): Grouped noisy patches tensor, shape (N, Href, Wref, k, n).
            V (torch.FloatTensor): Variance per pixel (*, k, n).
    
        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: 
                X_hat: Denoised patches, shape (N, Href, Wref, k, n).
                weights: Patch weights, shape (N, Href, Wref, k, 1).
        """
        _, _, _, k, n = Y.shape
        D = torch.sum(V, dim=-1)
        Q = Y @ Y.transpose(-2, -1)
        alpha = 0.25 * torch.mean(V, dim=(1, 2, 3, 4), keepdim=True) # alpha > 0 -> noisier risk which ensures positive definiteness
        theta = self.compute_theta(Q + n * alpha * torch.eye(k, dtype=Y.dtype, device=Y.device), D.unsqueeze(-1) + n * alpha)
        X_hat = theta @ Y
        weights = 1 / torch.sum(theta**2, dim=-1, keepdim=True).clip(1/k, 1)
        return X_hat, weights
    
    def denoise2(self, Y, X, V):
        """
        Denoises each group of similar patches (step 2).
    
        Args:
            Y (torch.FloatTensor): Grouped noisy patches, shape (N, Href, Wref, k, n).
            X (torch.FloatTensor): Grouped denoised patches (after step 1), shape (N, Href, Wref, k, n).
            V (torch.FloatTensor): Variance per pixel (*, k, n).
    
        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: 
                X_hat: Denoised patches, shape (N, Href, Wref, k, n).
                weights: Patch weights, shape (N, Href, Wref, k, 1).
        """
        k = Y.size(-2)
        D = torch.sum(V, dim=-1)
        Q = X @ X.transpose(-2, -1) + torch.diag_embed(D)
        theta = self.compute_theta(Q, D.unsqueeze(-1))
        X_hat = theta @ Y
        weights = 1 / torch.sum(theta**2, dim=-1, keepdim=True).clip(1/k, 1)
        return X_hat, weights
         
    def step1(self, input_y):
        """
        Performs the first denoising step on the input image.
    
        Args:
            input_y (torch.FloatTensor): Noisy input image, shape (N, C, H, W).
    
        Returns:
            torch.FloatTensor: First-stage denoised image, shape (N, C, H, W).
        """
        _, _, H, W = input_y.size() 
        k, p, w, s = self.k1, self.p1, self.w, self.s
        y_grayscale = torch.mean(input_y, dim=1, keepdim=True)
        indices = self.block_matching(y_grayscale, k, p, w, s)
        Y = self.gather_groups(input_y, indices, p)
        V = self.variance_groups(Y, indices, p)
        X_hat, weights = self.denoise1(Y, V)
        x_hat = self.aggregate(X_hat, weights, indices, H, W, p)
        return x_hat
        
    def step2(self, input_y, input_x):
        """
        Performs the second denoising step on the input image.
    
        Args:
            input_y (torch.FloatTensor): Noisy input image, shape (N, C, H, W).
            input_x (torch.FloatTensor): Denoised input image (after step 1), shape (N, C, H, W).
    
        Returns:
            torch.FloatTensor: Second-stage denoised image, shape (N, C, H, W).
        """
        _, _, H, W = input_y.size()
        k, p, w, s = self.k2, self.p2, self.w, self.s
        x_grayscale = torch.mean(input_x, dim=1, keepdim=True) # for color
        indices = self.block_matching(x_grayscale, k, p, w, s)
        Y = self.gather_groups(input_y, indices, p)
        X = self.gather_groups(input_x, indices, p)
        V = self.variance_groups(X, indices, p)
        X_hat, weights = self.denoise2(Y, X, V)
        x_hat = self.aggregate(X_hat, weights, indices, H, W, p)
        return x_hat
        
    def forward(self, y, sigma=25.0, a_pois=1.0, b_pois=0.0,\
                        noise_type='gaussian-homoscedastic',\
                        p1=7, p2=7, k1=18, k2=55, w=37, s=4,\
                        constraints='linear'):
        """
        Executes NL-Ridge denoising algorithm on the input noisy image.
    
        Args:
            y (torch.FloatTensor): Noisy input image, shape (N, C, H, W).
            sigma (float or torch.FloatTensor of shape (N, *)): Standard deviation of the Gaussian noise.
            a_pois (float): a parameter of the Poisson-Gaussian noise.
            b_pois (float): b parameter of the Poisson-Gaussian noise.
            noise_type (str): Type of noise ('gaussian-homoscedastic', 'gaussian-heteroscedastic', 'poisson', or 'poisson-gaussian').
            p1 (int): Patch size for the first step.
            p2 (int): Patch size for the second step.
            k1 (int): Number of similar patches for the first step.
            k2 (int): Number of similar patches for the second step.
            w (int): Size of the search window (odd number).
            s (int): Moving step size from one reference patch to another.
            constraints (str): Type of constraints ('linear', 'affine', 'conical', or 'convex').
        
            Recommended parameters for additive white Gaussian noise:
            - 0 < σ ≤ 15: p1=7, p2=7, k1=18, k2=55
            - 15 < σ ≤ 35: p1=9, p2=9, k1=18, k2=90
            - 35 < σ ≤ 50: p1=11, p2=9, k1=20, k2=120
    
        Returns:
            torch.FloatTensor: Final denoised image, shape (N, C, H, W).
        """
        # Set parameters
        for key, value in locals().items(): 
            if key != "self" and key != "y": setattr(self, key, value) 
        den1 = self.step1(y)
        den2 = self.step2(y, den1)
        return den2

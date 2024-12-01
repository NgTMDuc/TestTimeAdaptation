import torch
import torchvision.transforms as transforms
import numpy as np 
import argparse
import time
import os
from PIL import Image
import torch.nn as nn
"""
TODO:
- WCT: https://arxiv.org/pdf/1705.08086 (Done)
- AdaIN: https://arxiv.org/pdf/1703.06868v2 (Done with 2 version)
- Histogram matching (HM): https://arxiv.org/pdf/2203.07740 
- WCT^2: https://arxiv.org/pdf/1903.09760
- Fast patch-based style transfer: https://arxiv.org/pdf/1612.04337
- PhotoWCT
- Poisson Equation: https://arxiv.org/pdf/1709.09828
- Domain Aware Universal Style transfer: https://arxiv.org/pdf/2108.04441v2
- Z_star: https://arxiv.org/pdf/2311.16491

Survey:
- https://github.com/Roujack/awesome-photorealistic-style-transfer
- https://paperswithcode.com/paper/arbitrary-style-transfer-in-real-time-with
"""



class WCT(nn.Module):
    """
        White and coloring transforms
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    def whiten_and_color(self, cF, sF):
        cFSize = cF.size()
        c_mean = torch.mean(cF, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean

        contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0], device=cF.device).float()
        c_u, c_e, c_v = torch.svd(contentConv, some=False)

        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        sFSize = sF.size()
        s_mean = torch.mean(sF, 1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF, sF.t()).div(sFSize[1] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)

        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cF)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

    def transform(self, cF, sF):
        alpha = self.args.alpha  # Get alpha directly from the args

        cF = cF.float()
        sF = sF.float()

        B, C, W, H = cF.size()  # Batch size, Channels, Width, Height
        _, _, W1, H1 = sF.size()  # Style image's spatial dimensions

        # Reshape content and style features to [B, C, W*H] and [B, C, W1*H1]
        cFView = cF.view(B, C, -1)  # [B, C, W*H]
        sFView = sF.view(B, C, -1)  # [B, C, W1*H1]

        # Initialize the output tensor
        csF = torch.zeros_like(cF)  # Create a tensor to hold the transformed features

        # Process each image in the batch
        for i in range(B):
            # Extract the content and style features for the i-th image
            cF_i = cFView[i]  # Shape: [C, W*H]
            sF_i = sFView[i]  # Shape: [C, W1*H1]

            # Perform whitening and coloring transformation for each image
            targetFeature_i = self.whiten_and_color(cF_i, sF_i)

            # Reshape back to the original spatial size
            targetFeature_i = targetFeature_i.view(C, W, H)  # Shape: [C, W, H]

            # Mix the transformed content with the original content using alpha
            ccsF_i = alpha * targetFeature_i + (1.0 - alpha) * cF[i]

            # Store the transformed features
            csF[i] = ccsF_i

        return csF


class AdaIN(nn.Module):
    """
        Adaptive instance normalization (AdaIN) with batch processing support.
    """
    def __init__(self, args):
        super(AdaIN, self).__init__()
        self.args = args
    
    def calc_mean_std(self, feat, eps=1e-5):
        """
        Calculate the mean and standard deviation for each feature map.
        feat: [B, C, H, W] - Batch of feature maps
        eps: Small value for numerical stability
        """
        size = feat.size()
        assert len(size) == 4, "Size of the batch should be 4: (BxCxHxW)"
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps  # Var across spatial dims (HxW)
        feat_std = feat_var.sqrt().view(N, C, 1, 1)  # Std dev across spatial dims
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)  # Mean across spatial dims
        return feat_mean, feat_std

    def adaptive_instance_normalization(self, content_feat, style_feat):
        """
        Perform adaptive instance normalization: 
        Transfer the style from `style_feat` to the content of `content_feat`.
        content_feat: [B, C, H, W] - Batch of content features
        style_feat: [B, C, H, W] - Batch of style features
        """
        assert content_feat.size()[:2] == style_feat.size()[:2], "Batch size and channels must match"
        
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)
        
        # Normalize content features and apply style statistics
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def _calc_feat_flatten_mean_std(self, feat):
        """
        Flatten the feature map and calculate per-channel mean and std.
        feat: [B, C, H, W]
        """
        B, C, H, W = feat.size()
        feat_flatten = feat.view(B, C, -1)  # Flatten H and W dimensions
        mean = feat_flatten.mean(dim=-1, keepdim=True)
        std = feat_flatten.std(dim=-1, keepdim=True)
        return feat_flatten, mean, std

    def _mat_sqrt(self, x):
        """
        Compute the matrix square root of a positive semi-definite matrix.
        x: [C, C] - A square matrix (covariance matrix)
        """
        U, D, V = torch.svd(x)
        return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())

    def coral(self, source, target):
        """
        Perform CORAL (Correlation Alignment) to align the second-order statistics of 
        source feature map with target feature map.
        source: [B, C, H, W] - Source feature map (e.g., content features)
        target: [B, C, H, W] - Target feature map (e.g., style features)
        """
        # Flatten and calculate mean and std for both source and target
        source_f, source_f_mean, source_f_std = self._calc_feat_flatten_mean_std(source)
        source_f_norm = (source_f - source_f_mean.expand_as(source_f)) / source_f_std.expand_as(source_f)
        source_f_cov_eye = torch.mm(source_f_norm, source_f_norm.transpose(1, 2)) + torch.eye(source_f.size(1)).to(source.device)

        target_f, target_f_mean, target_f_std = self._calc_feat_flatten_mean_std(target)
        target_f_norm = (target_f - target_f_mean.expand_as(target_f)) / target_f_std.expand_as(target_f)
        target_f_cov_eye = torch.mm(target_f_norm, target_f_norm.transpose(1, 2)) + torch.eye(target_f.size(1)).to(target.device)

        # Transfer the normalized source features to match target's covariance
        source_f_norm_transfer = torch.mm(
            self._mat_sqrt(target_f_cov_eye),
            torch.mm(torch.inverse(self._mat_sqrt(source_f_cov_eye)), source_f_norm)
        )

        source_f_transfer = source_f_norm_transfer * target_f_std.expand_as(source_f_norm) + target_f_mean.expand_as(source_f_norm)

        return source_f_transfer.view_as(source)

    def transform(self, content_features, style_features):
        if self.args.coral:
            return self.coral(content_features, style_features)
        else:
            return self.adaptive_instance_normalization(content_features, style_features)

def build_transform(args):
    # print("Checking")
    # print(args)
    if args.transform == "AdaIN":
        return AdaIN(args)
    if args.transform == "WCT":
        return WCT(args)
    
    raise ValueError(f"Not implemented transform type {args.transform}")
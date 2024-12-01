import torch
import torchvision.transforms as transforms
import numpy as np 
import argparse
import time
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from skimage import exposure

"""
TODO:
- WCT: https://arxiv.org/pdf/1705.08086 (Done)
- AdaIN: https://arxiv.org/pdf/1703.06868v2 (Done)
- Histogram matching (HM): https://arxiv.org/pdf/2203.07740 (Done)
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


# Helper function to calculate mean and std
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)  # Expecting a 4D tensor (N, C, H, W)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

# AdaMean Class
class AdaMean(nn.Module):
    def __init__(self):
        super(AdaMean, self).__init__()

    def forward(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, _ = calc_mean_std(style_feat)
        content_mean, _ = calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(size))
        return normalized_feat + style_mean.expand(size)

# AdaStd Class
class AdaStd(nn.Module):
    def __init__(self):
        super(AdaStd, self).__init__()

    def forward(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        _, style_std = calc_mean_std(style_feat)
        _, content_std = calc_mean_std(content_feat)

        normalized_feat = (content_feat) / content_std.expand(size)
        return normalized_feat * style_std.expand(size)

# EFDM Class
class EFDM(nn.Module):
    def __init__(self):
        super(EFDM, self).__init__()

    def forward(self, content_feat, style_feat):
        assert (content_feat.size() == style_feat.size())
        B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
        
        value_content, index_content = torch.sort(content_feat.view(B, C, -1))  # Sort content features
        value_style, _ = torch.sort(style_feat.view(B, C, -1))  # Sort style features
        
        inverse_index = index_content.argsort(-1)  # Inverse index for reordering
        
        # Apply the feature distribution matching
        new_content = content_feat.view(B, C, -1) + (value_style.gather(-1, inverse_index) - content_feat.view(B, C, -1).detach())
        
        return new_content.view(B, C, W, H)

# HM (Histogram Matching) Class
class HM(nn.Module):
    def __init__(self):
        super(HM, self).__init__()

    def forward(self, content_feat, style_feat):
        assert (content_feat.size() == style_feat.size())
        B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
        
        # Flatten spatial dimensions for histogram matching
        x_view = content_feat.view(-1, W, H)

        # Convert to numpy for histogram matching
        # Assuming match_histograms is now from skimage.exposure
        image1_temp = exposure.match_histograms(
            np.array(x_view.detach().clone().cpu().float().transpose(0, 2)),
            np.array(style_feat.view(-1, W, H).detach().clone().cpu().float().transpose(0, 2))
        )

        image1_temp = torch.from_numpy(image1_temp).float().to(content_feat.device).transpose(0, 2).view(B, C, W, H)
        
        return content_feat + (image1_temp - content_feat).detach()

    
class WCT(nn.Module):
    def __init__(self,args, eps=1e-5):
        super(WCT, self).__init__()
        self.eps = eps
        self.args = args

    def forward(self, content, style):
        """
        Apply the Whitening and Coloring Transform (WCT) for style transfer.
        
        Arguments:
        content -- The content image feature map (B x C x H x W).
        style -- The style image feature map (B x C x H x W).
        
        Returns:
        output -- The stylized content feature map (B x C x H x W).
        """
        B, C, Hc, Wc = content.size()
        _, _, Hs, Ws = style.size()

        # Compute mean and covariance for content and style
        content_mean, content_cov = self.compute_mean_and_cov(content)
        style_mean, style_cov = self.compute_mean_and_cov(style)

        # Whitening content features
        whitened_content = self.whiten(content, content_mean, content_cov)

        # Coloring content features using the style statistics
        output = self.color(whitened_content, style_mean, style_cov)

        return output

    def compute_mean_and_cov(self, x):
        """
        Compute the mean and covariance matrix for the given feature map x.
        
        Arguments:
        x -- The input feature map (B x C x H x W).
        
        Returns:
        mean -- The mean of the feature map (B x C x 1 x 1).
        cov -- The covariance matrix of the feature map (B x C x C).
        """
        B, C, H, W = x.size()

        # Flatten spatial dimensions
        x_flat = x.view(B, C, -1)

        # Compute mean
        mean = x_flat.mean(dim=2, keepdim=True)

        # Compute covariance matrix
        centered_x = x_flat - mean
        cov = torch.bmm(centered_x, centered_x.transpose(1, 2)) / (H * W - 1)

        return mean, cov

    def whiten(self, x, mean, cov):
        """
        Perform whitening of the content feature map using the given mean and covariance matrix.
        
        Arguments:
        x -- The content feature map (B x C x H x W).
        mean -- The mean of the content feature map (B x C x 1 x 1).
        cov -- The covariance matrix of the content feature map (B x C x C).
        
        Returns:
        whitened_x -- The whitened content feature map (B x C x H x W).
        """
        B, C, H, W = x.size()

        # Flatten the spatial dimensions
        x_flat = x.view(B, C, -1)

        # Center the content feature map
        centered_x = x_flat - mean

        # Perform whitening using covariance matrix
        cov_inv_sqrt = torch.inverse(cov + self.eps * torch.eye(C, device=cov.device))
        whitened_x_flat = torch.bmm(cov_inv_sqrt, centered_x)

        # Reshape back to the original feature map size
        whitened_x = whitened_x_flat.view(B, C, H, W)

        return whitened_x

    def color(self, x, mean, cov):
        """
        Perform coloring of the whitened content feature map using the style's mean and covariance matrix.
        
        Arguments:
        x -- The whitened content feature map (B x C x H x W).
        mean -- The mean of the style feature map (B x C x 1 x 1).
        cov -- The covariance matrix of the style feature map (B x C x C).
        
        Returns:
        colored_x -- The stylized content feature map (B x C x H x W).
        """
        B, C, H, W = x.size()

        # Flatten the spatial dimensions of the content
        x_flat = x.view(B, C, -1)

        # Apply coloring: scale with style's covariance and add style's mean
        cov_sqrt = torch.sqrt(cov + self.eps * torch.eye(C, device=cov.device))
        colored_x_flat = torch.bmm(cov_sqrt, x_flat)

        # Add style's mean
        colored_x_flat = colored_x_flat + mean

        # Reshape back to the original feature map size
        colored_x = colored_x_flat.view(B, C, H, W)

        return colored_x


class AdaIN(nn.Module):
    def __init__(self, args, eps=1e-5, disabled=False):
        super(AdaIN, self).__init__()
        self.eps = eps
        self.disabled = disabled
        self.args = args
        
    def forward(self, content, style):
        """
            Perform Adaptive Instance Normalization.
            
            Arguments:
            content -- The content image feature map (B x C x H x W).
            style -- The style image feature map (B x C x H x W).
            
            Returns:
            output -- The stylized content feature map (B x C x H x W).
        """
        if self.disabled:
            return content

        # Get the dimensions of content and style
        B, C, Hc, Wc = content.size()
        _, _, Hs, Ws = style.size()

        # Compute the mean and standard deviation of the style image
        style_mean = style.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)  # B x C x 1 x 1
        style_std = style.view(B, C, -1).std(dim=2).view(B, C, 1, 1)  # B x C x 1 x 1

        # Compute the mean and standard deviation of the content image
        content_mean = content.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)  # B x C x 1 x 1
        content_std = content.view(B, C, -1).std(dim=2).view(B, C, 1, 1)  # B x C x 1 x 1

        # Normalize the content features and apply the style statistics
        normalized_content = (content - content_mean) / (content_std + self.eps)
        output = normalized_content * (style_std + self.eps) + style_mean

        return output

def build_transform(args):
    if args.transform == "AdaIN":
        return AdaIN(args)
    if args.transform == "WCT":
        return WCT(args)
    if args.transform == "HM":
        return HM()
    if args.transform == "EFDM":
        return EFDM()
    if args.transform == "AdaMean":
        return AdaMean()
    if args.transform == "AdaStd":
        return AdaStd()
    raise ValueError(f"Not implemented transform type {args.transform}")

# class AdaIN(nn.Module):
#     """
#         Adaptive instance normalization (AdaIN) with batch processing support.
#     """
#     def __init__(self, args):
#         super(AdaIN, self).__init__()
#         self.args = args
    
#     def calc_mean_std(self, feat, eps=1e-5):
#         """
#         Calculate the mean and standard deviation for each feature map.
#         feat: [B, C, H, W] - Batch of feature maps
#         eps: Small value for numerical stability
#         """
#         size = feat.size()
#         assert len(size) == 4, "Size of the batch should be 4: (BxCxHxW)"
#         N, C = size[:2]
#         feat_var = feat.view(N, C, -1).var(dim=2) + eps  # Var across spatial dims (HxW)
#         feat_std = feat_var.sqrt().view(N, C, 1, 1)  # Std dev across spatial dims
#         feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)  # Mean across spatial dims
#         return feat_mean, feat_std

#     def adaptive_instance_normalization(self, content_feat, style_feat):
#         """
#         Perform adaptive instance normalization: 
#         Transfer the style from `style_feat` to the content of `content_feat`.
#         content_feat: [B, C, H, W] - Batch of content features
#         style_feat: [B, C, H, W] - Batch of style features
#         """
#         assert content_feat.size()[:2] == style_feat.size()[:2], "Batch size and channels must match"
        
#         size = content_feat.size()
#         style_mean, style_std = self.calc_mean_std(style_feat)
#         content_mean, content_std = self.calc_mean_std(content_feat)
        
#         # Normalize content features and apply style statistics
#         normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
#         return normalized_feat * style_std.expand(size) + style_mean.expand(size)

#     def _calc_feat_flatten_mean_std(self, feat):
#         """
#         Flatten the feature map and calculate per-channel mean and std.
#         feat: [B, C, H, W]
#         """
#         B, C, H, W = feat.size()
#         feat_flatten = feat.view(B, C, -1)  # Flatten H and W dimensions
#         mean = feat_flatten.mean(dim=-1, keepdim=True)
#         std = feat_flatten.std(dim=-1, keepdim=True)
#         return feat_flatten, mean, std

#     def _mat_sqrt(self, x):
#         """
#         Compute the matrix square root of a positive semi-definite matrix.
#         x: [C, C] - A square matrix (covariance matrix)
#         """
#         U, D, V = torch.svd(x)
#         return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())

#     def coral(self, source, target):
#         """
#         Perform CORAL (Correlation Alignment) to align the second-order statistics of 
#         source feature map with target feature map.
#         source: [B, C, H, W] - Source feature map (e.g., content features)
#         target: [B, C, H, W] - Target feature map (e.g., style features)
#         """
#         # Flatten and calculate mean and std for both source and target
#         source_f, source_f_mean, source_f_std = self._calc_feat_flatten_mean_std(source)
#         source_f_norm = (source_f - source_f_mean.expand_as(source_f)) / source_f_std.expand_as(source_f)
#         source_f_cov_eye = torch.mm(source_f_norm, source_f_norm.transpose(1, 2)) + torch.eye(source_f.size(1)).to(source.device)

#         target_f, target_f_mean, target_f_std = self._calc_feat_flatten_mean_std(target)
#         target_f_norm = (target_f - target_f_mean.expand_as(target_f)) / target_f_std.expand_as(target_f)
#         target_f_cov_eye = torch.mm(target_f_norm, target_f_norm.transpose(1, 2)) + torch.eye(target_f.size(1)).to(target.device)

#         # Transfer the normalized source features to match target's covariance
#         source_f_norm_transfer = torch.mm(
#             self._mat_sqrt(target_f_cov_eye),
#             torch.mm(torch.inverse(self._mat_sqrt(source_f_cov_eye)), source_f_norm)
#         )

#         source_f_transfer = source_f_norm_transfer * target_f_std.expand_as(source_f_norm) + target_f_mean.expand_as(source_f_norm)

#         return source_f_transfer.view_as(source)

#     def forward(self, content_features, style_features):
#         if self.args.coral:
#             return self.coral(content_features, style_features)
#         else:
#             return self.adaptive_instance_normalization(content_features, style_features)
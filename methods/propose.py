import torch
from copy import deepcopy
import torch.nn as nn
import torch.jit
import torchvision
import math
import numpy as np 
import matplotlib.pyplot as plt
from einops import rearrange

class ProposeMethod(nn.Module):
    def __init__(self, args, model, optimizer, steps = 1, episodic = False, deyo_margin = 0.5 * math.log(1000), margin_e0=0.4*math.log(1000)):
        super().__init__()
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.deyo_margin = deyo_margin
        self.margin_e0 = margin_e0
    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.args, self.model, self.optimizer, self.deyo_margin, self.margin_e0)
        return outputs
    def reset(self):
        pass

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad() 
def forward_and_adapt(x, args, model, optimizer, deyo_margin, margin):
    outputs = model(x)
    
    optimizer.zero_grad()
    entropys = softmax_entropy(outputs)
    
    # if args.filter_ent:
    filter_ids_1 = torch.where((entropys < deyo_margin))
    # else:
        # filter_ids_1 = torch.where((entropys <= math.log(1000)))
    
    x_prime = x.detach().clone()
    
    if args.aug_type=='occ':
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
        x_prime[:, :, args.row_start:args.row_start+args.occlusion_size,args.column_start:args.column_start+args.occlusion_size] = occlusion_window
    elif args.aug_type=='patch':
        resize_t = torchvision.transforms.Resize(((x.shape[-1]//args.patch_len)*args.patch_len,(x.shape[-1]//args.patch_len)*args.patch_len))
        resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
        x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
        x_prime = resize_o(x_prime)
    elif args.aug_type=='pixel':
        x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
        x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
    with torch.no_grad():
        outputs_prime = model(x_prime)
    
    prob_outputs = outputs.softmax(1)
    prob_outputs_prime = outputs_prime.softmax(1)
    
    cls1 = prob_outputs.argmax(dim = 1)
    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)
    
    # if args.filter_plpd:
    filter_ids_2 = torch.where(plpd > args.plpd_threshold)
    # else:
        # filter_ids_2 = torch.where(plpd >= -2.0)
    
    if args.reweight_ent or args.reweight_plpd:
        coeff = (args.reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                 args.reweight_plpd * (1 / (torch.exp(-1. * plpd.clone().detach())))
                )            
        entropys = entropys.mul(coeff)
    
    combined_filters = torch.unique(torch.cat([filter_ids_1[0], filter_ids_2[0]]))
    
    entropys = entropys[combined_filters]
    loss = entropys.mean(0)
    # print(filter_ids_2)
    # print(filter_ids_1)
    # Ids that belong to filter 2 but not the filter 1
    other_filter_ids = torch.from_numpy(np.setdiff1d(filter_ids_2[0].cpu(), filter_ids_1[0].cpu()))
    length_combined_filters = len(combined_filters)
    if len(other_filter_ids) != 0:
        # print("Update")
        selected_ids = other_filter_ids[torch.randint(0, len(other_filter_ids), (length_combined_filters,))]
        pseudo_labels = outputs[combined_filters].softmax(1).argmax(dim = 1)
        logits = model.module.transform_style_forward(x[combined_filters], x[selected_ids])
        
        CeLoss = nn.CrossEntropyLoss()(logits, pseudo_labels)
        total_loss = loss + CeLoss
    else:
        total_loss = loss
    total_loss.backward()
    optimizer.step()
    
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with DeYO."""
    # train mode, because DeYO optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what DeYO updates
    model.requires_grad_(False)
    # configure norm for DeYO updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model
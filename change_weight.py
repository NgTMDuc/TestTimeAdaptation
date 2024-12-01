import os
import torch.nn as nn
import pickle
import json
import models.Res as Resnet 
import torch 

# path = "pretrained/waterbirds_pretrained_model.pickle"
path_save = "pretrained/waterbirds_pretrained_model.pth"
# with open(path, "rb") as f:
    # model = pickle.load(f)

# torch.save(model.state_dict(), path_save)
net = Resnet.__dict__["resnet50"]()
net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=2)
net.load_state_dict(torch.load(path_save))


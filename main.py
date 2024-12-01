from logging import debug
import os
import time
import math
from config import get_args
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")
args = get_args()
current_working_directory = os.getcwd()
print(current_working_directory) # should print the cwd

if args.dset=='ImageNet-C':
    args.data = os.path.join(args.data_root, 'ImageNet')
    args.data_corruption = os.path.join(args.data_root, args.dset)
elif args.dset=='Waterbirds':
    args.data_corruption = os.path.join(args.data_root, args.dset)
    for file in os.listdir(args.data_corruption):
        if file.endswith('h5py'):
            h5py_file = file
            break
    args.data_corruption_file = os.path.join(args.data_root, args.dset, h5py_file)
elif args.dset=='ColoredMNIST':
    args.data_corruption = os.path.join(args.data_root, args.dset)
biased = (args.exp_type=='spurious')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import json
import random
if args.wandb_log:
    import wandb
from datetime import datetime
from tqdm import tqdm
import numpy as np
from pycm import *
from utils.utils import get_logger
from dataset.selectedRotateImageFolder import prepare_test_data
from utils.cli_utils import *

import torch    

from methods import tent, eata, sam, sar, deyo, propose
import timm

import models.Res as Resnet

import pickle
from dataset.waterbirds_dataset import WaterbirdsDataset
from dataset.ColoredMNIST_dataset import ColoredMNIST
from dataset.cifar10_dataset import Cifar10Dataset

def validate(val_loader, model, args):
    batch_time  = AverageMeter("Time", ":6.3f")
    
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    if biased:
        LL_AM = AverageMeter('LL Acc', ':6.2f')
        LS_AM = AverageMeter('LS Acc', ':6.2f')
        SL_AM = AverageMeter('SL Acc', ':6.2f')
        SS_AM = AverageMeter('SS Acc', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, LL_AM, LS_AM, SL_AM, SS_AM],
            prefix='Test: ')
        
    model.eval()
    
    end = time.time()
    correct_count = [0, 0, 0, 0]
    total_count = [1e-6, 1e-6, 1e-6, 1e-6]
    
    for i, dl in enumerate(tqdm(val_loader)):
        images, targets = dl[0], dl[1]
        images = images.cuda()
        targets = targets.cuda()
        
        if biased:
            if args.dset == "Waterbirds":
                place = dl[2]['place'].cuda()
            
            else:
                place = dl[2].cuda()
            
            group = 2 * targets + place
        else:
            group = None
        
        output = model(images)

        if biased:
            TFtensor = (output.argmax(dim=1) == targets)
            for group_idx in range(4):
                correct_count[group_idx] += TFtensor[group==group_idx].sum().item()
                total_count[group_idx] += len(TFtensor[group==group_idx])
            
            acc1, acc5 = accuracy(output, targets, topk=(1, 1))
        else:
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        if (i + 1) % args.wandb_interval == 0:
            if biased:
                LL = correct_count[0]/total_count[0]*100
                LS = correct_count[1]/total_count[1]*100
                SL = correct_count[2]/total_count[2]*100
                SS = correct_count[3]/total_count[3]*100
                
                LL_AM.update(LL, images.size(0))
                LS_AM.update(LS, images.size(0))
                SL_AM.update(SL, images.size(0))
                SS_AM.update(SS, images.size(0))
                
            if args.wandb_log:
                wandb.log({f'{args.corruption}/LL': LL,
                            f'{args.corruption}/LS': LS,
                            f'{args.corruption}/SL': SL,
                            f'{args.corruption}/SS': SS,
                        })
            progress.display(i)
        
        batch_time.update(time.time() - end) 
        end = time.time()
    
    if biased:
        logger.info(f"- Detailed result under {args.corruption}. LL: {LL:.5f}, LS: {LS:.5f}, SL: {SL:.5f}, SS: {SS:.5f}")
        if args.wandb_log:
            wandb.log({'final_avg/LL': LL,
                       'final_avg/LS': LS,
                       'final_avg/SL': SL,
                       'final_avg/SS': SS,
                       'final_avg/AVG': (LL+LS+SL+SS)/4,
                       'final_avg/WORST': min(LL,LS,SL,SS)
                      })
        avg = (LL+LS+SL+SS)/4
        logger.info(f"Result under {args.corruption}. The adaptation accuracy of {args.method} is  average: {avg:.5f}")
        
        LLs.append(LL)
        LSs.append(LS)
        SLs.append(SL)
        SSs.append(SS)
        acc1s.append(avg)
        acc5s.append(min(LL,LS,SL,SS))
        
        logger.info(f"The LL accuracy are {LLs}")
        logger.info(f"The LS accuracy are {LSs}")
        logger.info(f"The SL accuracy are {SLs}")
        logger.info(f"The SS accuracy are {SSs}")
        
        logger.info(f"The average accuracy are {acc1s}")
        logger.info(f"The worst accuracy are {acc5s}")
    
    else:
        logger.info(f"Result under {args.corruption}. The adaptation accuracy of {args.method} is top1: {top1.avg:.5f} and top5: {top5.avg:.5f}")
        
        acc1s.append(top1.avg.item())
        acc5s.append(top5.avg.item())
        
        logger.info(f"acc1s are {acc1s}")
        logger.info(f"acc5s are {acc5s}")
    return top1.avg, top5.avg


if __name__ == "__main__":
    if args.dset=='ImageNet-C':
        args.num_class = 1000
    elif args.dset=='Waterbirds' or args.dset=='ColoredMNIST':
        args.num_class = 2
    elif args.dset == "CIFAR-10-C":
        args.num_class = 10 
    print('The number of classes:', args.num_class)

    if args.dset == 'Waterbirds':
        assert biased
        assert args.data_corruption_file.endswith('h5py')
        assert args.model == 'resnet50_bn_torch'
    
    elif args.dset == 'ColoredMNIST':
        assert biased
        assert args.model == 'resnet18_bn'
    
    elif args.dset == "CIFAR-10-C":
        assert args.model == "resnet18_bn"

    if biased:
        assert (args.dset == 'Waterbirds' or args.dset == 'ColoredMNIST')
        assert args.lr_mul == 5.0
        
    now = datetime.now()
    date_time = now.strftime("%m-%d-%H-%M-%S")
    
    total_top1 = AverageMeter('Acc@1', ':6.2f')
    total_top5 = AverageMeter('Acc@5', ':6.2f')
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    args.logger_name = f"{args.method}-{args.model}"
    
    if args.method == "propose":
        args.logger_name += f"-{args.transform}-{args.layer}-{str(args.coral)}"
    
    args.logger_name += ".txt"
    
    output_path = os.path.join(args.output, str(args.level))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Create folder {}".format(output_path))
    
    logger = get_logger(name="project", output_directory=output_path, log_name=args.logger_name, debug=False) 
    common_corruptions = [
                            'gaussian_noise',
                            'shot_noise',
                            'impulse_noise',
                            'defocus_blur', 
                            'glass_blur', 
                            'motion_blur', 
                            'zoom_blur', 
                            'snow', 
                            'frost', 
                            'fog', 
                            'brightness', 
                            'contrast', 
                            'elastic_transform', 
                            'pixelate', 
                            'jpeg_compression'
                            ]
    if biased:
        common_corruptions = ['spurious correlation']
        
    args.e_margin *= math.log(args.num_class)
    args.sar_margin_e0 *= math.log(args.num_class)
    args.deyo_margin *= math.log(args.num_class)
    args.deyo_margin_e0 *= math.log(args.num_class)
    
    ir = args.imbalance_ratio
    acc1s, acc5s = [], []
    LLs, LSs, SLs, SSs = [], [], [], []
    for corrupt_i, corrupt in enumerate(common_corruptions):
        
        args.corruption = corrupt
        bs = args.test_batch_size
        args.print_freq = 50000 // 20 // bs
        
        #Load dataset
        if args.method in ["propose", 'tent', 'eata', 'sar', 'deyo', 'no_adapt']:
            if args.dset == "ImageNet-C":
                val_dataset, val_loader = prepare_test_data(args)
                val_dataset.switch_mode(True, False)
            
            elif args.dset == "Waterbirds":
                transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Resize((224, 224)),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                    )
                val_dataset = WaterbirdsDataset(file=args.data_corruption_file, split='test', transform=transform)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, 
                                                             shuffle=args.if_shuffle, num_workers=args.workers,
                                                             pin_memory=True)
            elif args.dset == "ColoredMNIST":
                kwargs = {'num_workers': args.workers, 'pin_memory': True}
                val_dataset = ColoredMNIST(root=args.data_corruption, env='test',  # flip=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                                               ]))
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, 
                                                             shuffle=args.if_shuffle, **kwargs)
            elif args.dset == "CIFAR-10-C":
                pass
            else:
                assert False, NotImplementedError
        else:
            assert False, NotImplementedError
        
        #Load model
        if args.method in ["propose",'tent', 'eata', 'sar', 'deyo', 'no_adapt']:
            if args.model == "resnet50_bn_torch":
                if args.dset == "Waterbirds":
                    with open(os.path.join(args.pretrained_folder, args.wbmodel_name), 'rb') as f:
                        net = pickle.load(f)
                elif args.dset == "ImageNet-C":
                    net = Resnet.__dict__["resnet50"](pretrained = True)
                
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
                args.lr *= args.lr_mul
            
            elif args.model == "resnet18_bn":
                if args.dset == "ColoredMNIST":
                    # with open(os.path.join(args.pretrained_folder, args.cmmodel_name), 'rb') as f:
                        # net = pickle.load(f)
                    net = Resnet.__dict__["resnet18"]()
                    net.fc = torch.nn.Linear(in_features=net.fc.in_features, out_features=2)
                    net.load_state_dict(torch.load(os.path.join(args.pretrained_folder, args.cmmodel_name)))
                    # with open()
                elif args.dset == "CIFAR-10-C":
                    with open(os.path.join(args.pretrained_folder, args.cfmodel_name), 'rb') as f:
                        net = pickle.load(f)
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
                args.lr *= args.lr_mul
                
            else:
                assert False, NotImplementedError
        else:
            assert False, NotImplementedError
            
        print(f"Using {torch.cuda.device_count()} GPUs")
        net = nn.DataParallel(net)
        net = net.cuda()

        #TODO
        # logger.info(args)
        #Adaptation
        if args.method == "no_adapt":
            # acc1, acc5 = validate(val_loader, net, args)
            pass
        
        elif args.method == "tent":
            net = tent.configure_model(net)
            params, param_names = tent.collect_params(net)
            #TODO
            # logger.info(param_names)
            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            net = tent.Tent(net, optimizer)
            # acc1, acc5 = validate(val_loader, net, args)
        
        elif args.method == "eata":
            if args.eata_fishers:
                print("EATA!")
                
                if args.dset == "Waterbirds":
                    fisher_dataset = WaterbirdsDataset(file=args.data_corruption_file, split='train', transform=transform)
                    fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=args.test_batch_size,
                                                                                    shuffle=args.if_shuffle, num_workers=args.workers,
                                                                                    pin_memory=True)
                
                elif args.dset == "ColoredMNIST":
                    fisher_dataset = ColoredMNIST(root=args.data_corruption, env='all_train', flip=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                                                  ]))     
                    fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=args.test_batch_size,
                                                                shuffle=args.if_shuffle, **kwargs)
                elif args.dset == "CIFAR-10-C":
                    # import torchvision.transforms as transforms
                    kwargs = {'num_workers': args.workers, 'pin_memory': True}
                    fisher_dataset = Cifar10Dataset(root= args.data_corruption, 
                                           corruption_type=args.corruption, 
                                           severity=args.level,
                                           transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))
                    fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=args.test_batch_size,
                                                             shuffle=args.if_shuffle, **kwargs)
                else:
                    fisher_dataset, fisher_loader = prepare_test_data(args)
                
                fisher_dataset.set_dataset_size(args.fisher_size)
                fisher_dataset.switch_mode(True, False)
                
                net = eata.configure_model(net)
                params, param_names = eata.collect_params(net)
                ewc_optimizer = torch.optim.SGD(params, 0.001)
                fishers = {}
                train_loss_fn = nn.CrossEntropyLoss().cuda()
                for iter_, data in enumerate(fisher_loader, start=1):
                    images, targets = data[0], data[1]
                    if args.gpu is not None:
                        images = images.cuda(non_blocking=True)
                    if torch.cuda.is_available():
                        targets = targets.cuda(non_blocking=True)
                    outputs = net(images)
                    _, targets = outputs.max(1)
                    loss = train_loss_fn(outputs, targets)
                    loss.backward()
                    for name, param in net.named_parameters():
                        if param.grad is not None:
                            if iter_ > 1:
                                fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                            else:
                                fisher = param.grad.data.clone().detach() ** 2
                            if iter_ == len(fisher_loader):
                                fisher = fisher / iter_
                            fishers.update({name: [fisher, param.data.clone().detach()]})
                    ewc_optimizer.zero_grad()
                logger.info("compute fisher matrices finished")
                del ewc_optimizer
            else:
                net = eata.configure_model(net)
                params, param_names = eata.collect_params(net)
                print('ETA!')
                fishers = None
            
            args.corruption = corrupt
            #TODO
            # logger.info(param_names)
            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            net = eata.EATA(args, net, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)
            
            # acc1, acc5 = validate(val_loader, net, args)
        
        elif args.method == "sar":
            net = sar.configure_model(net)
            params, param_names = sar.collect_params(net)
            #TODO
            # logger.info(param_names)
            
            base_optimizer = torch.optim.SGD
            optimizer = sam.SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
            net = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)
            # acc1, acc5 = validate(val_loader, net, args)
        
        elif args.method == "deyo":
            net = deyo.configure_model(net)
            params, param_names = deyo.collect_params(net)
            #TODO
            # logger.info(param_names)

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            net = deyo.DeYO(net, args, optimizer, deyo_margin=args.deyo_margin, margin_e0=args.deyo_margin_e0)
            # acc1, acc5 = val_loader(val_loader, net, args)
        elif args.method == "propose":
            net = propose.configure_model(net)
            params, param_names = propose.collect_params(net)
            #TODO
            # logger.info(param_names)
            
            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            net = propose.ProposeMethod(args, net, optimizer, deyo_margin=args.deyo_margin, margin_e0=args.deyo_margin_e0)
        else:
            assert False, NotImplementedError
        acc1, acc5 = validate(val_loader, net, args)
        total_top1.update(acc1, 1)
        total_top5.update(acc5, 1)
    if not biased:
        logger.info(f"The average of top1 accuracy is {total_top1.avg}")
        logger.info(f"The average of top5 accuracy is {total_top5.avg}")
        if args.wandb_log:
            wandb.log({'final_avg/top1': total_top1.avg,
                       'final_avg/top5': total_top5.avg})

            wandb.finish()
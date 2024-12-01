import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import torchvision
from torchvision import datasets, transforms
import models.Res as Resnet
from tqdm import tqdm
# Set up argument parser
parser = argparse.ArgumentParser(description='CIFAR-10 Training and Evaluation')
parser.add_argument('--root_dir', default='../DATA', help='path to data')
parser.add_argument('--dset_dir', default='cifar10', help='name of dataset directory')
parser.add_argument('--gpu', default='1', type=str, help='gpu index for training.')
parser.add_argument('--seed', default=2024, type=int, help='seed for initializing training.')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size for training.')
parser.add_argument('--test_batch_size', default=1000, type=int, help='batch_size for test.')
parser.add_argument('--workers', default=2, type=int, help='num_workers for train loader.')
parser.add_argument('--if_shuffle', default=1, type=int, help='shuffle for training.')
parser.add_argument('--max_epochs', default=100, type=int, help='epochs for training.')
parser.add_argument('--interval', default=10, type=int, help='intervals for saving.')
args = parser.parse_args()

# Set GPU environment
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Set random seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False        

# Define the CIFAR-10 dataset and transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 datasets
train_dataset = datasets.CIFAR10(root=args.root_dir, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root=args.root_dir, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.if_shuffle, num_workers=args.workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

# Function to test the model
def test_model(model, device, test_loader, set_name="test set"):
    print("Eval model")
    model.eval()
    CELoss = torch.nn.CrossEntropyLoss()
    test_loss = 0
    correct_count = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    total_count = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += CELoss(output, target).sum().item()  # sum up batch loss
            TFtensor = (output.argmax(dim=1) == target)
            for label in range(10):
                correct_count[label] += TFtensor[target == label].sum().item()
                total_count[label] += len(TFtensor[target == label])

    test_loss /= len(test_loader.dataset)
    accs = correct_count / total_count * 100

    print(f'\nPerformance on {set_name}: Average loss: {test_loss:.4f}, Accuracy: {correct_count.sum().item()}/{total_count.sum().item()} ({correct_count.sum().item() / total_count.sum().item() * 100:.2f}%)\n')
    for i in range(10):
        print(f'Class {i} accuracy: {accs[i].item():.2f}% ({correct_count[i]}/{total_count[i]})')

    return correct_count, total_count

# Function to train the model
def erm_train(model, device, train_loader, optimizer, epoch, args):
    print("Train model")
    model.train()
    CELoss = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = CELoss(output, target)
        loss.backward()
        optimizer.step()
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))

# Function to train and test the model
def train_and_test_erm(model, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

    for epoch in range(args.max_epochs):
        erm_train(model, device, train_loader, optimizer, epoch, args)
        if epoch % args.interval == 0 or epoch == args.max_epochs - 1:
            test_model(model, device, train_loader, set_name='train set')
            test_model(model, device, test_loader)

    # Save the trained model
    # if not os.path.exists(os.path.join("/mnt/disk1/ducntm/DeYO/pretrained", 'CIFAR10_model_update.pickle')):
        # with open(os.path.join("/mnt/disk1/ducntm/DeYO/pretrained", 'CIFAR10_model_update.pickle'), 'wb') as f:
            # pickle.dump(model, f)
    # else:
        # print('Pretrained model already exists.')
    save_path = "pretrained/cifar_pretrained.pth"
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    assert args.root_dir is not None
    assert args.dset_dir is not None
    
    args.root_dir = os.path.join(args.root_dir, args.dset_dir)
    
    # Load a pretrained ResNet18 model and adapt it for CIFAR-10 (10 classes)
    # model = torchvision.models.resnet18(pretrained=True)
    model = Resnet.__dict__["resnet18"]()
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
    model.eval()

    train_and_test_erm(model, args)
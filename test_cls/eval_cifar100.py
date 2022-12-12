from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *

from models import networks

import foolbox as fb
import eagerpy as ep

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=8,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.0075,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


mean_generation = [0.5, 0.5, 0.5]
std_generation = [0.5, 0.5, 0.5]

def _pgd_blackbox(model, fmodel, netG,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):

    std_map1 = torch.ones_like(X[:, 0:1, :, :]).cuda() * std_generation[0]
    std_map2 = torch.ones_like(X[:, 0:1, :, :]).cuda() * std_generation[1]
    std_map3 = torch.ones_like(X[:, 0:1, :, :]).cuda() * std_generation[2]
    mean_map1 = torch.ones_like(X[:, 0:1, :, :]).cuda() * mean_generation[0]
    mean_map2 = torch.ones_like(X[:, 0:1, :, :]).cuda() * mean_generation[1]
    mean_map3 = torch.ones_like(X[:, 0:1, :, :]).cuda() * mean_generation[2]
    std_map = torch.cat([std_map1, std_map2, std_map3], dim=1)
    mean_map = torch.cat([mean_map1, mean_map2, mean_map3], dim=1)

    X_clone = X.clone()
    X_clone = X_clone.sub(mean_map).div(std_map)
    X_clone = netG.forward(X_clone)
    X_clone = X_clone.mul(std_map).add(mean_map)
    out = model(X_clone)
    err = (out.data.max(1)[1] != y.data).float().sum()

    #########################################
    X_pgd = X.clone().detach()
    attack = fb.attacks.LinfDeepFoolAttack()
    # attack = fb.attacks.L2CarliniWagnerAttack(steps=10, stepsize=0.0075)
    epsilons = [0.031]
    X_pgd = ep.astensor(X_pgd)
    y_pgd = ep.astensor(y)
    advs2, advs, success = attack(fmodel, X_pgd, y_pgd, epsilons=epsilons)

    X_pgd = advs[0].raw
    X_pgd = X_pgd.sub(mean_map).div(std_map)
    X_pgd = netG.forward(X_pgd)
    X_pgd = X_pgd.mul(std_map).add(mean_map)

    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def eval_adv_test_blackbox2(model_target, model_source, netG, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    fmodel = fb.PyTorchModel(model_source, bounds=(0, 1))

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, fmodel, netG, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def main():
    input_nc = 3
    output_nc = 3
    ngf = 64
    netG = 'global'
    n_downsample_global = 2
    n_blocks_global = 9
    n_local_enhancers = 1
    n_blocks_local = 3
    norm = 'instance'
    gpu_ids = [0]

    model_black = ResNet50_class100().to(device)
    model_path = '../model_cls_NoD/cifar100/resnet50/model.pt'
    model_black.load_state_dict(torch.load(model_path))

    netG = networks.define_G(input_nc, output_nc, ngf, netG,
                             n_downsample_global, n_blocks_global, n_local_enhancers,
                             n_blocks_local, norm, gpu_ids=gpu_ids)
    save_path = '../checkpoints/cifar100_model_wideresnet/latest_net_G.pth'
    save_path_list = [save_path]

    model = WideResNet(num_classes=100).to(device)
    model_path = '../model_cls_NoD/cifar100/wideresnet/model.pt'
    model.load_state_dict(torch.load(model_path))

    for mm in range(len(save_path_list)):
        netG.load_state_dict(torch.load(save_path_list[mm]))
        netG.eval()
        eval_adv_test_blackbox2(model, model_black, netG, device, test_loader)
    del model
    del model_black


if __name__ == '__main__':
    main()

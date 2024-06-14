import copy
import torch
import argparse
from tqdm import tqdm
from os import path, makedirs
import os
import numpy as np
import torchvision
from sklearn.cluster import KMeans
from modules import transform, resnet, network, contrastive_loss, sam
from utils import yaml_config_hook
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from opacus.validators import ModuleValidator
from evaluation import evaluation
from opacus.accountants.utils import get_noise_multiplier
import pickle
import PIL
import math
from collections import defaultdict
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances
from utils import SimSiam, clustering, centeral_clustering_by_cosine_similarity, centeral_get_global_centroids, Aggregator, centeral_pretrain, setup_seed, pretrain, asymmetric_loss, symmetric_loss, get_global_centroids, clustering_by_cosine_similarity 

class Cifar100Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, transform):
        # 初始化文件路径或文件名列表。
        # 初始化该类的一些基本参数。
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # 返回数据集的总大小
        return self.samples.shape[0]

    def __getitem__(self, index):
        return (self.transform(self.samples[index]), self.transform(self.samples[index]))

class Cifar100DatasetTest(torch.utils.data.Dataset):
    def __init__(self, samples, labels, transform):
        # 初始化文件路径或文件名列表。
        # 初始化该类的一些基本参数。
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # 返回数据集的总大小
        return self.samples.shape[0]

    def __getitem__(self, index):
        return (self.transform(self.samples[index]), self.labels[index])

class Cifar100DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, samples, labels, transform):
        # 初始化文件路径或文件名列表。
        # 初始化该类的一些基本参数。
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # 返回数据集的总大小
        return self.samples.shape[0]

    def __getitem__(self, index):
        return self.transform(self.samples[index])


def getModelUpdate(gradDict, netcopy, net):
    net_para = net.state_dict()
    for name in net_para:
        gradDict[name] = (netcopy[name].data - net_para[name].data).clone()


def createclientDataIndex(samplePath):
    clientDataIndex={}
    with open(samplePath, "rb") as f:
        Sample=pickle.load(f) 

    samplePerClient = len(Sample)//args.n_clients
    index=torch.tensor(list(range(len(Sample))))

    for c in range(args.n_clients):
        if c==args.n_clients-1:
            clientDataIndex[c] = torch.tensor(index[c*samplePerClient:])
        else:
            clientDataIndex[c] = torch.tensor(index[c*samplePerClient:(c+1)*samplePerClient])
    
    return clientDataIndex

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    device= torch.device("cuda:5")
    print(device)
    parser = argparse.ArgumentParser('CCFC')
    parser.add_argument('--data_root', default= './datasets',type=str, help='path to dataset directory')
    parser.add_argument('--exp_dir', default='./save/CCFC/Cifar100-fed', type=str, help='path to experiment directory')
    
    parser.add_argument('--trial', type=str, default='v2', help='trial id')
    parser.add_argument('--seed', type=int, default = 66, help='random seed')
    
    parser.add_argument('--proj_hidden_dim', default = 512, type=int, help='feature dimension')
    parser.add_argument('--num_proj_layers', type=int, default=2, help='number of projection layer')
    parser.add_argument('--latent_dim', default = 256, type=int, help='feature dimension')
    parser.add_argument('--pre_hidden_dim', default = 64, type=int, help='feature dimension')
    
    parser.add_argument('--k', type = int, default = 20, help='the number of clusters')
    parser.add_argument('--lbd', type=float, default = 0.01, help='trade-off hyper')
    parser.add_argument('--p', type=float, default = 0., help='non-iid level')
    parser.add_argument('--lr', type=float, default = 0.0005, help='learning rate')
    
    parser.add_argument('--batch_size', type=int, default = 300, help='batch_size')
    parser.add_argument('--num_workers', type=int, default = 6, help='num of workers to use')
    parser.add_argument('--mini_bs', type=int, default = 150, help='mini_bs')
    parser.add_argument('--image_size', type=int, default = 224, help='image_size')
    parser.add_argument('--test_image_size', type=int, default = 256, help='test_image_size')
    parser.add_argument('--n_clients', type=int, default = 200, help='n_clients')
    parser.add_argument('--resnet', type=str, default = "ResNet18", help='resnet')
    parser.add_argument('--global_lr', type=float, default = 1, help='global_lr')
    parser.add_argument('--sample_ratio', type=float, default = 0.05, help='global_lr')
    
    args = parser.parse_args()
    setup_seed(args.seed)
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    trial_dir = path.join(args.exp_dir, args.trial)
    if not path.exists(trial_dir):
        makedirs(trial_dir)

    with open("./datasets/cifar100/Sample", "rb") as f:
        Sample = pickle.load(f)
    with open("./datasets/cifar100/Label20", "rb") as f:
        Label = pickle.load(f)


    ground_truth_all = Label
    clientDataIndex = createclientDataIndex("./datasets/cifar100/Sample")
    n = len(Sample)
    s = 0.5
    mean=[0.5071, 0.4867, 0.4408]
    std=[0.2675, 0.2565, 0.2761]
    transformation = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomResizedCrop(size=args.image_size, interpolation=PIL.Image.BICUBIC, scale=(0.2, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.4 * s, 0.2 * s)], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ]
    transformation.append(torchvision.transforms.ToTensor())
    transformation.append(torchvision.transforms.Normalize(mean=mean, std=std))
    transformation = torchvision.transforms.Compose(transformation)
    dataset_aug = Cifar100Dataset(Sample, Label, transformation)
    # aug_dataloader = DataLoader(
    #         dataset_aug,
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         drop_last=True,
    #         num_workers=args.num_workers,
    #         pin_memory=True,
    #     )

    transformation_test = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(
                    (args.test_image_size, args.test_image_size), interpolation=PIL.Image.BICUBIC
                ),
            torchvision.transforms.CenterCrop(args.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
        ]
    transformation_test = torchvision.transforms.Compose(transformation_test)
    dataset_test = Cifar100DatasetTest(Sample, Label, transformation_test)
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.mini_bs,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    dataset_train = Cifar100DatasetTrain(Sample, Label, transformation_test)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    Nets = {}
    Nets, agg, pseudo_labels = pretrain(Nets, args, Sample, Label, clientDataIndex, test_loader, ground_truth_all, n, trial_dir, device)
    # Nets, pseudo_labels = centeral_pretrain(Nets, args, aug_dataloader, test_loader, ground_truth_all, n, trial_dir, device)
   
    # agg = Aggregator(device, args)
    # loadpath="save/CCFC/v2/model_pretrain_0_59.pt"
    # checkpoint = torch.load(loadpath, map_location=device)
    # agg.global_model.load_state_dict(checkpoint)
    # Nets[f'model'] = copy.deepcopy(agg.global_model)
    # Nets[f'model'].train()
    # Nets[f'optim'] = torch.optim.Adam(Nets[f'model'].parameters(), lr = args.lr)
    # Nets[f'optim'].zero_grad()

    # global_centroids = get_global_centroids(args, test_loader, agg.global_model, device)
    # pseudo_labels = clustering_by_cosine_similarity(args, test_loader, agg.global_model, global_centroids, ground_truth_all.numpy(), device)    
    # print("pseudo_labels",len(pseudo_labels))

    global_centroids = centeral_get_global_centroids(args, test_loader, agg.global_model, device)
    pseudo_labels = centeral_clustering_by_cosine_similarity(args, test_loader, agg.global_model, global_centroids, ground_truth_all.numpy(), device)    
    print("pseudo_labels",len(pseudo_labels))

    print(f'training on: {device}')
    for epoch in range(200):
        # agg.global_model.eval()
        global_w = agg.global_model.state_dict()
        train_loss=0
        for batch_idx, x in enumerate(train_loader):
            Nets[f'model'].load_state_dict(copy.deepcopy(global_w))
            Nets[f'model'].train()
            true_epoch=0
            lossPerUser=0
            modelUpdateDict={}
            for local_epoch in range(3):
                batch_num = int(args.batch_size/args.mini_bs)
                batch_index=np.array(range(args.batch_size))
                np.random.shuffle(batch_index)
                for j in range(batch_num):
                    true_epoch += 1 
                    batch_records = batch_index[j*args.mini_bs:(j+1)*args.mini_bs]
                    x_batch=x[batch_records].to(device)
                    z, p = Nets[f'model'](x_batch)
                    with torch.no_grad():
                        _, p_g = agg.global_model(x_batch)
                    
                    count = 0
                    loss_cluster = 0
                    labels = pseudo_labels[clientDataIndex[batch_idx][batch_records]]
                    for j in torch.unique(labels):
                        idx_j = labels == j
                        if  sum(idx_j) > 1:
                            count += 1
                            loss_cluster += symmetric_loss(p[idx_j], z[idx_j])
                    loss_cluster /= count
                    
                    loss_model = asymmetric_loss(p, p_g)
                    loss = loss_cluster + args.lbd * loss_model
                    lossPerUser+=loss.item()
                    loss.backward()
                    Nets[f'optim'].step()
                    Nets[f'optim'].zero_grad()
            print('Round: ', epoch, "User:", batch_idx, 'Train Loss: %.3f' % (lossPerUser/true_epoch))    
            getModelUpdate(modelUpdateDict, agg.global_model.state_dict(), Nets[f'model']) # modelcopy - model 
    
            train_loss+=lossPerUser
            agg.collect(modelUpdateDict)   
        
        print('Round: ', epoch, 'Train Loss: %.3f' % (train_loss/(true_epoch*args.n_clients)))    
        agg.update()

        # global_centroids = get_global_centroids(args, test_loader, agg.global_model, device)
        # pseudo_labels = clustering_by_cosine_similarity(args, test_loader, agg.global_model, global_centroids, ground_truth_all.numpy(), device) 
    
        pseudo_labels = clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)
        if (epoch+1)%10==0:
            save_path = path.join(trial_dir, f"model_train_{int(args.p / 0.25)}_{epoch}.pt")
            torch.save(agg.global_model.state_dict(), save_path)

        # if (epoch+1)%10==0:
        #     save_path = path.join(trial_dir, f"model_train_{int(args.p / 0.25)}_{epoch}.pt")
        #     torch.save( Nets[f'model'].state_dict(), save_path)
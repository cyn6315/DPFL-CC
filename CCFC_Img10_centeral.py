import copy
import torch
import argparse
from tqdm import tqdm
from os import path, makedirs
import os
import numpy as np
import torchvision
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.pipeline import Pipeline
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
from utils import plus_centeral_pretrain, centeral_clustering, covariance_loss, SimSiam, centeral_clustering_by_cosine_similarity, centeral_get_global_centroids, centeral_pretrain, setup_seed, pretrain, asymmetric_loss, symmetric_loss, get_global_centroids, clustering_by_cosine_similarity
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics.pairwise import cosine_similarity

class Img10(torch.utils.data.Dataset):
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

class Img10Test(torch.utils.data.Dataset):
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

class Img10Train(torch.utils.data.Dataset):
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
        return (self.transform(self.samples[index]), index)


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
    device= torch.device("cuda:2")
    print(device)
    parser = argparse.ArgumentParser('CCFC')
    parser.add_argument('--data_root', default= './datasets',type=str, help='path to dataset directory')
    parser.add_argument('--exp_dir', default='./save/CCFC/Img', type=str, help='path to experiment directory')
    
    parser.add_argument('--trial', type=str, default='v3', help='trial id')
    parser.add_argument('--seed', type=int, default = 66, help='random seed')
    
    parser.add_argument('--proj_hidden_dim', default = 512, type=int, help='feature dimension')
    parser.add_argument('--num_proj_layers', type=int, default=2, help='number of projection layer')
    parser.add_argument('--latent_dim', default = 256, type=int, help='feature dimension')
    parser.add_argument('--pre_hidden_dim', default = 64, type=int, help='feature dimension')
    
    parser.add_argument('--k', type = int, default = 10, help='the number of clusters')
    parser.add_argument('--lbd', type=float, default = 0.1, help='trade-off hyper')
    parser.add_argument('--p', type=float, default = 0., help='non-iid level')
    parser.add_argument('--lr', type=float, default = 0.0005, help='learning rate')
    
    parser.add_argument('--batch_size', type=int, default = 145, help='batch_size')
    parser.add_argument('--num_workers', type=int, default = 6, help='num of workers to use')
    parser.add_argument('--mini_bs', type=int, default = 145, help='mini_bs')
    parser.add_argument('--image_size', type=int, default = 224, help='image_size')
    parser.add_argument('--test_image_size', type=int, default = 256, help='test_image_size')
    parser.add_argument('--n_clients', type=int, default = 1, help='n_clients')
    parser.add_argument('--resnet', type=str, default = "ResNet18", help='resnet')
    parser.add_argument('--global_lr', type=float, default = 1, help='global_lr')
    parser.add_argument('--eta', type=float, default = 0, help='eta')
    
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

    with open("./datasets/Img/Sample", "rb") as f:
        Sample = pickle.load(f)
    with open("./datasets/Img/Label", "rb") as f:
        Label = pickle.load(f)
    ground_truth_all = Label
    n = len(Sample)

    s = 0.5
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    transformation = transform.Transforms(size=args.image_size, mean=mean, std=std, blur=False).train_transform
    dataset_aug = Img10(Sample, Label, transformation)
    aug_dataloader = DataLoader(
            dataset_aug,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    transformation_test = transform.Transforms(size=args.image_size,test_size=args.test_image_size, mean=mean, std=std, blur=False).test_transform
    dataset_test = Img10Test(Sample, Label, transformation_test)
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    dataset_train = Img10Train(Sample, Label, transformation_test)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    # clientDataIndex = createclientDataIndex("./datasets/cifar10/Sample")
    Nets = {}
    Nets, pseudo_labels = centeral_pretrain(Nets, args, aug_dataloader, test_loader, ground_truth_all, n, trial_dir, device)
    # Nets, pseudo_labels = plus_centeral_pretrain(Nets, args, aug_dataloader, test_loader, ground_truth_all, n, trial_dir, device)
   
    # Nets[f'model'] = SimSiam(args).to(device)
    # # loadpath="save/CCFC/v1/model_pretrain_0_9.pt"
    # # checkpoint = torch.load(loadpath, map_location=device)
    # # Nets[f'model'].load_state_dict(checkpoint, strict=False)
    # for name, param in Nets[f'model'].named_parameters():
    #     param.requires_grad_(False)
    # Nets[f'model'].train()
    # Nets[f'optim'] = torch.optim.Adam(Nets[f'model'].parameters(), lr = args.lr)
    # Nets[f'optim'].zero_grad()
    # global_centroids = centeral_get_global_centroids(args, test_loader, Nets[f'model'], device)
    # pseudo_labels = centeral_clustering_by_cosine_similarity(args, test_loader, Nets[f'model'], global_centroids, ground_truth_all.numpy(), device)    
    # print("pseudo_labels",len(pseudo_labels))

    # reshaped_train=Sample.reshape(60000,3072)
    # kmeans = KMeans(n_clusters=10,max_iter=200)
    # global_centroids = kmeans.fit(reshaped_train).cluster_centers_
    # pred = cosine_similarity(reshaped_train, global_centroids).argmax(1)
    # nmi, ari, f, acc = evaluation.evaluate(ground_truth_all.numpy(), pred)
    # print('Global '+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))


    print(f'training on: {device}')
    for epoch in range(200):
        train_loss=0
        true_epoch=0
        Nets[f'model'].train()
        for batch_idx, (x, idx) in enumerate(train_loader):
            modelUpdateDict={}
            true_epoch += 1 
            z, p = Nets[f'model'](x.to(device)) 
            count = 0
            loss_cluster = 0
            labels = pseudo_labels[idx]
            for j in torch.unique(labels):
                idx_j = labels == j
                if  sum(idx_j) > 1:
                    count += 1
                    loss_cluster += symmetric_loss(p[idx_j], z[idx_j])
            loss_cluster /= count
            loss_cov = covariance_loss(z)/(args.latent_dim*args.latent_dim) 
            loss = loss_cluster + args.eta*loss_cov
            train_loss+=loss.item()
            loss.backward()
            Nets[f'optim'].step()
            Nets[f'optim'].zero_grad()
  
        print('Round: ', epoch, 'Train Loss: %.3f' % (train_loss/true_epoch))    
        pseudo_labels = centeral_clustering(args, test_loader, Nets[f'model'], ground_truth_all.numpy(), device)
        if (epoch+1)%10==0:
            save_path = path.join(trial_dir, f"model_train_{int(args.p / 0.25)}_{epoch}.pt")
            torch.save( Nets[f'model'].state_dict(), save_path)
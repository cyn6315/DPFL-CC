import copy
import torch
import random
import argparse
import numpy as np
from os import path
from torch import nn
from tqdm import tqdm
from PIL import ImageFilter
from torchvision import transforms
from torchvision.datasets import MNIST
import torchvision
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from modules import resnet
from sklearn.cluster import KMeans
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from opacus.validators import ModuleValidator
import math
from evaluation import evaluation
import pickle
import PIL
from tqdm import tqdm 
import math
from collections import defaultdict
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from torch.nn.functional import normalize


class Cifar10Dataset(torch.utils.data.Dataset):
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

class Aggregator:
    def __init__(self, device, args):
        self.global_model = SimSiam(args).to(device)
        self.device=device
        self.count = 0
        self.modelUpdate={}

    def update(self):
        print("count",self.count)
        if self.count == 0:
            return
        global_w = self.global_model.state_dict()
        for name in global_w:
            if name in self.modelUpdate:       
                global_w[name] = global_w[name]-torch.div(self.modelUpdate[name], self.count)     
        self.global_model.load_state_dict(copy.deepcopy(global_w)) #update global_model    
        self.count = 0
        self.modelUpdate.clear()

    def collect(self, model_grad):
        self.count+=1
        for name in model_grad:
            if name not in self.modelUpdate:
                self.modelUpdate[name] = copy.deepcopy(model_grad[name].data)
            else:
                self.modelUpdate[name] += copy.deepcopy(model_grad[name].data)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
            # nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        out_dim = in_dim

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
            # nn.ReLU()
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x

class SimSiam(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = resnet.get_resnet(args.resnet, 4)
        self.projector = projection_MLP(512, args.proj_hidden_dim, args.latent_dim, args.num_proj_layers)
        self.predictor = prediction_MLP(args.latent_dim, args.pre_hidden_dim)
           
    def forward(self, x):
        z = self.projector(self.backbone(x).view(-1, 512))
        p = self.predictor(z)    
        
        return z, p
    
    def forward_rep(self, x):
        h = self.backbone(x).view(-1, 512)
        return h
    
def asymmetric_loss(p, z): #sample-level
    z = z.detach()  # stop gradient
    return - F.cosine_similarity(p, z, dim=-1).mean()

def symmetric_loss(p, z): #cluster-level
    z = z.detach()  # stop gradient
    #ipdb.set_trace()
    z_norm, p_norm = F.normalize(z), F.normalize(p)
    return - torch.mm(z_norm, p_norm.T).mean()

def covariance_loss(z): #covariance-loss
    copy_z = z.detach()
    z = z - torch.mean(copy_z, dim=0)
    cov_matrix = torch.empty(0)
    for row in z:
        matrix = torch.matmul(row.unsqueeze(1), row.unsqueeze(0))
        if cov_matrix.numel() == 0:
            cov_matrix=matrix
        else:
            cov_matrix+=matrix
    
    cov_matrix = cov_matrix/len(z)
    average_cov_f2_norm_squared = torch.sum(torch.pow(cov_matrix, 2))
    return average_cov_f2_norm_squared

def get_centroids(latent_z, nClusters):
    kmeans = KMeans(n_clusters = nClusters).fit(latent_z)
    return kmeans.cluster_centers_
    

def get_centroids_and_labels(latent_z, nClusters):
    kmeans = KMeans(n_clusters = nClusters).fit(latent_z)
    return kmeans.cluster_centers_ , kmeans.labels_


def get_global_centroids(args, test_loader, model, device ):
    local_latent_z_ls = [] 
    local_centroids_ls = []
    model.eval()
    for step, (x, y) in enumerate(test_loader):
        latent_z = []
        with torch.no_grad():
            r=math.ceil(args.batch_size/args.mini_bs)
            for i in range(r):
                if i==r-1:
                    z = F.normalize(model(x[i*args.mini_bs:].to(device))[0])
                    latent_z.append(z)
                else:
                    z = F.normalize(model(x[i*args.mini_bs:(i+1)*args.mini_bs].to(device))[0])
                    latent_z.append(z)
        
        latent_z = torch.cat(latent_z).cpu().numpy()
        
        local_centroids = get_centroids(latent_z, args.k)
        local_centroids_ls.append(local_centroids)   

    local_centroids_all = np.concatenate(local_centroids_ls) ##check here 
    global_centroids = get_centroids(local_centroids_all, args.k)
    return global_centroids


def clustering_by_cosine_similarity(args, test_dataloader, global_model, global_centroids, ground_truth, device):
    global_model.eval()
    latent_z = []
    with torch.no_grad(): 
        for step, (x, _) in enumerate(test_dataloader):
            r=math.ceil(args.batch_size/args.mini_bs)
            for i in range(r):
                if i==r-1:
                    z = F.normalize(global_model(x[i*args.mini_bs:].to(device))[0]).cpu().detach()
                    latent_z.append(z)
                else:
                    z = F.normalize(global_model(x[i*args.mini_bs:(i+1)*args.mini_bs].to(device))[0]).cpu().detach()
                    latent_z.append(z)

    latent_z_all = torch.cat(latent_z, 0).cpu().numpy()
    pred = cosine_similarity(latent_z_all, global_centroids).argmax(1)
    nmi, ari, f, acc = evaluation.evaluate(ground_truth, pred)
    print('Global '+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    return torch.from_numpy(pred)

def clustering(args, test_loader, model, ground_truth, device ):
    local_latent_z_ls = [] 
    local_centroids_ls = []
    model.eval()
    for step, (x, y) in enumerate(test_loader):
        latent_z = []
        with torch.no_grad():
            r=math.ceil(args.batch_size/args.mini_bs)
            for i in range(r):
                if i==r-1:
                    z = F.normalize(model(x[i*args.mini_bs:].to(device))[0]).cpu()
                    latent_z.append(z)
                else:
                    z = F.normalize(model(x[i*args.mini_bs:(i+1)*args.mini_bs].to(device))[0]).cpu()
                    latent_z.append(z)
        
        latent_z = torch.cat(latent_z).numpy()
        local_latent_z_ls.append(latent_z)
        local_centroids = get_centroids(latent_z, args.k)
        local_centroids_ls.append(local_centroids)   

    local_centroids_all = np.concatenate(local_centroids_ls) ##check here 
    global_centroids = get_centroids(local_centroids_all, args.k)

    latent_z_all = np.concatenate(local_latent_z_ls) ##check here 
    pred = cosine_similarity(latent_z_all, global_centroids).argmax(1)
    nmi, ari, f, acc = evaluation.evaluate(ground_truth, pred)
    print('Global '+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    return torch.from_numpy(pred)

def noiid_clustering(args, test_loader, model, ground_truth, device ):
    local_latent_z_ls = [] 
    local_centroids_ls = []
    local_preds_ls = {}
    model.eval()
    for step, (x, y) in enumerate(test_loader):
        latent_z = []
        # print(step, torch.unique(y))
        with torch.no_grad():
            r=math.ceil(args.batch_size/args.mini_bs)
            for i in range(r):
                if i==r-1:
                    z = F.normalize(model(x[i*args.mini_bs:].to(device))[0]).cpu()
                    latent_z.append(z)
                else:
                    z = F.normalize(model(x[i*args.mini_bs:(i+1)*args.mini_bs].to(device))[0]).cpu()
                    latent_z.append(z)
        latent_z = torch.cat(latent_z).numpy()
        local_centroids, local_preds = get_centroids_and_labels(latent_z, args.classes_per_user)
        local_preds_ls[step] = local_preds
        local_centroids_ls.append(local_centroids)   

    local_centroids_all = np.concatenate(local_centroids_ls) ##check here 
    global_centroids, global_preds = get_centroids_and_labels(local_centroids_all, args.k)

    preClusters=[]
    for c in range(args.n_clients):
        globalCluster = global_preds[c*args.classes_per_user: (c+1)*args.classes_per_user]
        localCluster=local_preds_ls[c]
        preCluster = np.zeros(len(localCluster), dtype=int)
        for i in range(args.classes_per_user):
            index=np.where(localCluster==i)[0]
            preCluster[index] = globalCluster[i]
        preCluster=list(preCluster)
        preClusters.extend(preCluster)
    preClusters=np.array(preClusters)
    nmi, ari, f, acc = evaluation.evaluate(ground_truth, preClusters)
    print('Global '+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    return torch.from_numpy(preClusters)

def centeral_get_global_centroids(args, test_loader, model, device ):
    latent_z = []
    model.eval()
    for step, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            z = F.normalize(model(x.to(device))[0]).cpu()
            latent_z.append(z)
        
    latent_z = torch.cat(latent_z).numpy()
    global_centroids = get_centroids(latent_z, args.k)
    return global_centroids


def centeral_clustering_by_cosine_similarity(args, test_dataloader, global_model, global_centroids, ground_truth, device):
    global_model.eval()
    latent_z = []
    with torch.no_grad(): 
        for step, (x, _) in enumerate(test_dataloader):
            z = F.normalize(global_model(x.to(device))[0]).cpu().detach()
            latent_z.append(z)

    latent_z_all = torch.cat(latent_z, 0).cpu().numpy()
    pred = cosine_similarity(latent_z_all, global_centroids).argmax(1)
    nmi, ari, f, acc = evaluation.evaluate(ground_truth, pred)
    print('Global '+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    return torch.from_numpy(pred)

def centeral_clustering(args, test_loader, model, ground_truth, device ):
    latent_z = []
    model.eval()
    for step, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            r=math.ceil(args.batch_size/args.mini_bs)
            for i in range(r):
                if i==r-1:
                    z = F.normalize(model(x[i*args.mini_bs:].to(device))[0]).cpu()
                    latent_z.append(z)
                else:
                    z = F.normalize(model(x[i*args.mini_bs:(i+1)*args.mini_bs].to(device))[0]).cpu()
                    latent_z.append(z)
    latent_z = torch.cat(latent_z).numpy()
    global_centroids = get_centroids(latent_z, args.k)
    pred = cosine_similarity(latent_z, global_centroids).argmax(1)
    nmi, ari, f, acc = evaluation.evaluate(ground_truth, pred)
    print('Global '+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    return torch.from_numpy(pred)

def getModelUpdate(gradDict, netcopy, net):
    net_para = net.state_dict()
    for name in net_para:
        gradDict[name] = (netcopy[name].data - net_para[name].data).clone()

def getModelUpdate_param(gradDict, netcopy, net):
    for name, param in net.named_parameters ():
        if param.requires_grad:
            gradDict[name] = (netcopy[name].data - param.data).clone()


def pretrain(Nets, args, Sample, Label, clientDataIndex, test_loader, ground_truth_all, n, trial_dir, device):
    s = 0.5
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]

    # mean=[0.5071, 0.4867, 0.4408]
    # std=[0.2675, 0.2565, 0.2761]
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

    agg = Aggregator(device, args)

    loadpath="save/CCFC/v4/model_pretrain_0_1279.pt"
    # loadpath="save/CCFC++/Cifar100/v0/model_pretrain_0_59.pt"
    checkpoint = torch.load(loadpath, map_location=device)
    agg.global_model.load_state_dict(checkpoint, strict=False)

    Nets[f'model'] = SimSiam(args).to(device)
    # Nets[f'model'] = ModuleValidator.fix(Nets[f'model']).to(device)
    Nets[f'optim'] = torch.optim.Adam(Nets[f'model'].parameters(), lr = args.lr)
    Nets[f'optim'].zero_grad()

    pseudo_labels = clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)
    pseudo_labels = centeral_clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)
    return
    for name, param in Nets[f'model'].named_parameters():
        print(name, param.numel(), param.size())
    totalIndex=np.array(range(args.n_clients))
    sampleClients=int(args.n_clients*args.sample_ratio)
    for epoch in range(2000):
        # agg.global_model.eval()
        global_w = agg.global_model.state_dict() 
        train_loss=0
        np.random.shuffle(totalIndex)
        clients_sampled = list(totalIndex[0:sampleClients])
        clients_sampled.sort()
        indices = []
        for c in clients_sampled:
            indices.extend(clientDataIndex[c])
        dataset_aug = Cifar10Dataset(Sample[indices], Label[indices], transformation)
        aug_dataloader = DataLoader(
            dataset_aug,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,)
        for batch_idx, (x1, x2) in enumerate(aug_dataloader):
            Nets[f'model'].load_state_dict(copy.deepcopy(global_w))
            Nets[f'model'].train()
            lossPerUser=0
            modelUpdateDict={}
            gradDict={}
            true_epoch=0
            batch_num = int(args.batch_size/args.mini_bs)
            batch_index=np.array(range(args.batch_size))
            np.random.shuffle(batch_index)
            for j in range(batch_num):
                true_epoch += 1 
                batch_records = batch_index[j*args.mini_bs:(j+1)*args.mini_bs]
                x1_batch = x1[batch_records].to(device)
                x2_batch = x2[batch_records].to(device)  
                z1, p1 = Nets[f'model'](x1_batch)
                z2, p2 = Nets[f'model'](x2_batch)  
                with torch.no_grad():
                    _, p1_g = agg.global_model(x1_batch)
                    _, p2_g = agg.global_model(x2_batch)
                
                loss_sample =asymmetric_loss(p1, z2) + asymmetric_loss(p2, z1)
                loss_model =asymmetric_loss(p1, p1_g) + asymmetric_loss(p2, p2_g)
                loss = loss_sample + args.lbd * loss_model
                loss.backward()
                lossPerUser+=loss.item()
                Nets[f'optim'].step() 
                Nets[f'optim'].zero_grad()
            # print('Round: ', epoch, "User:", clients_sampled[batch_idx], 'Train Loss: %.3f' % (lossPerUser/true_epoch))
            getModelUpdate(modelUpdateDict, agg.global_model.state_dict(), Nets[f'model'])  
            train_loss+=lossPerUser/true_epoch
            agg.collect(modelUpdateDict)
        # Averaging the local models' parameters to get global model
        agg.update()
        print('Round: ', epoch, 'Train Loss: %.3f' % (train_loss/sampleClients))
        if (epoch+1)%10==0:
            print("centeral clustering:")
            pseudo_labels = centeral_clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)
        if (epoch+1)%40==0:
            pseudo_labels = clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)

        save_path = path.join(trial_dir, f"model_pretrain_{int(args.p / 0.25)}_{epoch}.pt")
        if (epoch+1)%40==0:
            torch.save(agg.global_model.state_dict(), save_path)
    
   
    Nets[f'model'] = copy.deepcopy(agg.global_model)
    Nets[f'optim'] = torch.optim.Adam(Nets[f'model'].parameters(), lr = args.lr) 
    
    pseudo_labels = clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)
    print("pseudo_labels",len(pseudo_labels))
    return Nets, agg, pseudo_labels
    

def centeral_pretrain(Nets, args, dataLoader, test_loader, ground_truth_all, n, trial_dir, device): 
    # loadpath="save/CCFC/v2/model_pretrain_0_9.pt"
    # checkpoint = torch.load(loadpath, map_location=device)

    Nets[f'model'] = SimSiam(args).to(device)
    # Nets[f'model'] = ModuleValidator.fix(Nets[f'model']).to(device)
    # Nets[f'model'].load_state_dict(checkpoint, strict=False)

    Nets[f'model'].train()
    Nets[f'optim'] = torch.optim.Adam(Nets[f'model'].parameters(), lr = args.lr)
    Nets[f'optim'].zero_grad()
    
    pseudo_labels = centeral_clustering(args, test_loader, Nets[f'model'], ground_truth_all.numpy(), device)    
    
    for name in Nets[f'model'].state_dict():
        print(name)
    for epoch in range(50):
        Nets[f'model'].train()
        train_loss=0
        true_epoch=0
        for batch_idx, (x1, x2) in enumerate(dataLoader):
            true_epoch += 1 
            x1 = x1.to(device)
            x2 = x2.to(device)  
            z1, p1 = Nets[f'model'](x1)
            z2, p2 = Nets[f'model'](x2)  
            loss = asymmetric_loss(p1, z2) + asymmetric_loss(p2, z1)
            loss.backward()
            train_loss+=loss.item()
            Nets[f'optim'].step() 
            Nets[f'optim'].zero_grad()
        print('Round: ', epoch, 'Train Loss: %.3f' % (train_loss/true_epoch))
        # global_centroids = centeral_get_global_centroids(args, test_loader, Nets[f'model'], device)
        # pseudo_labels = centeral_clustering_by_cosine_similarity(args, test_loader, Nets[f'model'], global_centroids, ground_truth_all.numpy(), device) 
        
        # if (epoch+1)%2==0:
        pseudo_labels = centeral_clustering(args, test_loader,  Nets[f'model'], ground_truth_all.numpy(), device )

        save_path = path.join(trial_dir, f"model_pretrain_{int(args.p / 0.25)}_{epoch}.pt")
        if (epoch+1)%10==0:
            torch.save(Nets[f'model'].state_dict(), save_path)
    
    Nets[f'optim'] = torch.optim.Adam(Nets[f'model'].parameters(), lr = args.lr) 
    pseudo_labels = centeral_clustering(args, test_loader,  Nets[f'model'], ground_truth_all.numpy(), device )
    print("pseudo_labels",len(pseudo_labels))
    return Nets, pseudo_labels
    
def plus_pretrain(Nets, args, Sample, Label, clientDataIndex, test_loader, ground_truth_all, n, trial_dir, device):
    s = 0.5
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]

    # mean=[0.5071, 0.4867, 0.4408]
    # std=[0.2675, 0.2565, 0.2761]
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

    agg = Aggregator(device, args)

    loadpath="save/CCFC++/Img/v0/model_pretrain_0_119.pt"
    print("loadpath", loadpath)
    checkpoint = torch.load(loadpath, map_location=device)
    agg.global_model.load_state_dict(checkpoint, strict=False)

    Nets[f'model'] = SimSiam(args).to(device)
    # Nets[f'model'] = ModuleValidator.fix(Nets[f'model']).to(device)
    Nets[f'optim'] = torch.optim.Adam(Nets[f'model'].parameters(), lr = args.lr)
    Nets[f'optim'].zero_grad()

    pseudo_labels = clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)
    pseudo_labels = centeral_clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)
    for name, param in Nets[f'model'].named_parameters():
        print(name, param.numel(), param.size())
    totalIndex=np.array(range(args.n_clients))
    sampleClients=int(args.n_clients*args.sample_ratio)
    for epoch in range(2000):
        # agg.global_model.eval()
        global_w = agg.global_model.state_dict() 
        train_loss=0
        np.random.shuffle(totalIndex)
        clients_sampled = list(totalIndex[0:sampleClients])
        clients_sampled.sort()
        indices = []
        for c in clients_sampled:
            indices.extend(clientDataIndex[c])
        dataset_aug = Cifar10Dataset(Sample[indices], Label[indices], transformation)
        aug_dataloader = DataLoader(
            dataset_aug,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,)
        for batch_idx, (x1, x2) in enumerate(aug_dataloader):
            Nets[f'model'].load_state_dict(copy.deepcopy(global_w))
            Nets[f'model'].train()
            lossPerUser=0
            modelUpdateDict={}
            gradDict={}
            true_epoch=0
            batch_num = int(args.batch_size/args.mini_bs)
            batch_index=np.array(range(args.batch_size))
            np.random.shuffle(batch_index)
            for j in range(batch_num):
                true_epoch += 1 
                batch_records = batch_index[j*args.mini_bs:(j+1)*args.mini_bs]
                x1_batch = x1[batch_records].to(device)
                x2_batch = x2[batch_records].to(device)  
                z1, p1 = Nets[f'model'](x1_batch)
                z2, p2 = Nets[f'model'](x2_batch)  
                with torch.no_grad():
                    _, p1_g = agg.global_model(x1_batch)
                    _, p2_g = agg.global_model(x2_batch)
                
                loss_sample =asymmetric_loss(p1, z2) + asymmetric_loss(p2, z1)
                loss_model =asymmetric_loss(p1, p1_g) + asymmetric_loss(p2, p2_g)
                loss_cov = (covariance_loss(z1) + covariance_loss(z2))/(args.latent_dim*args.latent_dim)
                loss = loss_sample + args.lbd * loss_model + args.eta*loss_cov
                loss.backward()
                lossPerUser+=loss.item()
                Nets[f'optim'].step() 
                Nets[f'optim'].zero_grad()
            # print('Round: ', epoch, "User:", clients_sampled[batch_idx], 'Train Loss: %.3f' % (lossPerUser/true_epoch))
            getModelUpdate(modelUpdateDict, agg.global_model.state_dict(), Nets[f'model'])  
            train_loss+=lossPerUser/true_epoch
            agg.collect(modelUpdateDict)
        # Averaging the local models' parameters to get global model
        agg.update()
        print('Round: ', epoch, 'Train Loss: %.3f' % (train_loss/sampleClients))
        if (epoch+1)%10==0:
            print("centeral clustering:")
            pseudo_labels = centeral_clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)
        if (epoch+1)%40==0:
            pseudo_labels = clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)

        save_path = path.join(trial_dir, f"model_pretrain_{int(args.p / 0.25)}_{epoch}.pt")
        if (epoch+1)%40==0:
            torch.save(agg.global_model.state_dict(), save_path)
    
   
    Nets[f'model'] = copy.deepcopy(agg.global_model)
    Nets[f'optim'] = torch.optim.Adam(Nets[f'model'].parameters(), lr = args.lr) 
    
    pseudo_labels = clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)
    print("pseudo_labels",len(pseudo_labels))
    return Nets, agg, pseudo_labels
    

def plus_centeral_pretrain(Nets, args, dataLoader, test_loader, ground_truth_all, n, trial_dir, device): 
    loadpath="save/CCFC++/Img/v0/model_pretrain_0_119.pt"
    checkpoint = torch.load(loadpath, map_location=device)
    print("loadpath",loadpath)
    Nets[f'model'] = SimSiam(args).to(device)
    Nets[f'model'].load_state_dict(checkpoint, strict=False)

    Nets[f'model'].train()
    Nets[f'optim'] = torch.optim.Adam(Nets[f'model'].parameters(), lr = args.lr)
    Nets[f'optim'].zero_grad()

    pseudo_labels = centeral_clustering(args, test_loader, Nets[f'model'], ground_truth_all.numpy(), device)    
    
    for name, param in Nets[f'model'].named_parameters():
        print(name, param.numel(), param.size())
    for epoch in range(500):
        Nets[f'model'].train()
        train_loss=0
        true_epoch=0
        for batch_idx, (x1, x2) in enumerate(dataLoader):
            true_epoch += 1 
            x1 = x1.to(device)
            x2 = x2.to(device)  
            z1, p1 = Nets[f'model'](x1)
            z2, p2 = Nets[f'model'](x2)  
            loss_cov = covariance_loss(z1)/(args.latent_dim*args.latent_dim)  + covariance_loss(z2)/(args.latent_dim*args.latent_dim)
            loss_sample = asymmetric_loss(p1, z2) + asymmetric_loss(p2, z1)
            loss = loss_sample + args.eta*loss_cov
            loss.backward()
            train_loss+=loss.item()
            Nets[f'optim'].step() 
            Nets[f'optim'].zero_grad()
        print('Round: ', epoch, 'Train Loss: %.3f' % (train_loss/true_epoch))

        if (epoch+1)%2==0:
            pseudo_labels = centeral_clustering(args, test_loader,  Nets[f'model'], ground_truth_all.numpy(), device )

        save_path = path.join(trial_dir, f"model_pretrain_{int(args.p / 0.25)}_{epoch}.pt")
        if (epoch+1)%10==0:
            torch.save(Nets[f'model'].state_dict(), save_path)
    
   
    Nets[f'optim'] = torch.optim.Adam(Nets[f'model'].parameters(), lr = args.lr) 
    
    pseudo_labels = centeral_clustering(args, test_loader,  Nets[f'model'], ground_truth_all.numpy(), device )
    print("pseudo_labels",len(pseudo_labels))
    return Nets, pseudo_labels

def setBNeval(model):
    model.projector.layer1[1].eval()
    model.projector.layer2[1].eval()
    model.projector.layer3[1].eval()

    model.predictor.layer1[1].eval()


def noiid_pretrain(Nets, args, Sample, Label, clientDataIndex, test_loader, ground_truth_all, n, trial_dir, device):
    s = 0.5
    # mean=[0.4914, 0.4822, 0.4465]
    # std=[0.2023, 0.1994, 0.2010]

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

    agg = Aggregator(device, args)

    # loadpath="save/CCFC/v4/model_pretrain_0_19.pt"
    # loadpath="save/CCFC/Cifar100/v0/model_pretrain_0_19.pt"

    # loadpath="save/CCFC++/Cifar10/v1/model_pretrain_0_19.pt"
    loadpath="save//CCFC++/Cifar100/v0/model_pretrain_0_19.pt"
    checkpoint = torch.load(loadpath, map_location=device)
    agg.global_model.load_state_dict(checkpoint, strict=False)
    print(loadpath)

    Nets[f'model'] = SimSiam(args).to(device)
    # Nets[f'model'] = ModuleValidator.fix(Nets[f'model']).to(device)

    # print("centeal clustering")
    # centeral_clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device )
    pseudo_labels = noiid_clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)

    for name in Nets[f'model'].state_dict():
        print(name)

    totalIndex=np.array(range(args.n_clients))
    sampleClients=int(args.n_clients*args.sample_ratio)
    for epoch in range(50):
        agg.global_model.eval()
        global_w = agg.global_model.state_dict() 
        train_loss=0
        np.random.shuffle(totalIndex)
        clients_sampled = list(totalIndex[0:sampleClients])
        clients_sampled.sort()
        indices = []
        for c in clients_sampled:
            indices.extend(clientDataIndex[c])
        dataset_aug = Cifar10Dataset(Sample[indices], Label[indices], transformation)
        aug_dataloader = DataLoader(
            dataset_aug,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,)
        for batch_idx, (x1, x2) in enumerate(aug_dataloader):
            # Nets[f'optim'] = torch.optim.SGD(Nets[f'model'].parameters(), lr = args.lr, momentum=0.9, weight_decay=5e-4)
            Nets[f'model'].load_state_dict(copy.deepcopy(global_w))
            Nets[f'model'].train()
            Nets[f'optim'] = torch.optim.Adam(Nets[f'model'].parameters(), lr = args.lr)
            Nets[f'optim'].zero_grad()

            lossPerUser=0
            modelUpdateDict={}
            true_epoch=0
            batch_num = int(args.batch_size/args.mini_bs)
            batch_index=np.array(range(args.batch_size))
            np.random.shuffle(batch_index)
            for j in range(batch_num):
                if j >5:
                    break
                true_epoch += 1 
                batch_records = batch_index[j*args.mini_bs:(j+1)*args.mini_bs]
                x1_batch = x1[batch_records].to(device)
                x2_batch = x2[batch_records].to(device)  
                z1, p1 = Nets[f'model'](x1_batch)
                z2, p2 = Nets[f'model'](x2_batch)  
                with torch.no_grad():
                    _, p1_g = agg.global_model(x1_batch)
                    _, p2_g = agg.global_model(x2_batch)
                
                loss_sample =asymmetric_loss(p1, z2) + asymmetric_loss(p2, z1)
                loss_model =asymmetric_loss(p1, p1_g) + asymmetric_loss(p2, p2_g)
                loss_cov = covariance_loss(z1)/(args.latent_dim*args.latent_dim)  + covariance_loss(z2)/(args.latent_dim*args.latent_dim)
                # print(loss_sample, loss_model, loss_cov)
                loss = loss_sample + args.lbd * loss_model + args.eta*loss_cov
                # loss = loss_sample + args.lbd * loss_model
                loss.backward()
                lossPerUser+=loss.item()
                Nets[f'optim'].step() 
                Nets[f'optim'].zero_grad()
            # print('Round: ', epoch, "User:", clients_sampled[batch_idx], 'Train Loss: %.3f' % (lossPerUser/true_epoch))
            getModelUpdate(modelUpdateDict, agg.global_model.state_dict(), Nets[f'model'])  
            train_loss+=lossPerUser/true_epoch
            agg.collect(modelUpdateDict)
        # Averaging the local models' parameters to get global model
        agg.update()
        print('Round: ', epoch, 'Train Loss: %.3f' % (train_loss/sampleClients))
        
        # if (epoch+1)%1==0:
        #     print("centeral clustering")
        #     pseudo_labels = centeral_clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)
        if (epoch+1)%1==0:
            pseudo_labels = noiid_clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)

        save_path = path.join(trial_dir, f"model_pretrain_{int(args.p / 0.25)}_{epoch}.pt")
        if (epoch+1)%2==0:
            torch.save(agg.global_model.state_dict(), save_path)
    
   
    Nets[f'model'] = copy.deepcopy(agg.global_model)
    Nets[f'optim'] = torch.optim.Adam(Nets[f'model'].parameters(), lr = args.lr) 
    
    pseudo_labels = noiid_clustering(args, test_loader, agg.global_model, ground_truth_all.numpy(), device)
    print("pseudo_labels",len(pseudo_labels))
    return Nets, agg, pseudo_labels
    
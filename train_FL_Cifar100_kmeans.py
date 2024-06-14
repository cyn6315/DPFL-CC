import os
import numpy as np
import torch
import torchvision
import argparse
from sklearn.cluster import KMeans
from utils import yaml_config_hook, awasthisheffet
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from opacus.validators import ModuleValidator
from evaluation import evaluation
from opacus.accountants.utils import get_noise_multiplier
import torch.nn.functional as F
from fastDP import PrivacyEngine
import pickle
import timm
from tqdm import tqdm 
from fcmeans import FCM
from skfuzzy.cluster import cmeans
from sklearn.decomposition import PCA
from collections import defaultdict
from modules import transform, resnet, network, contrastive_loss, sam


def save_model2(args, model, current_epoch):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)


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
        return (self.transform(self.samples[index]), self.labels[index])

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


def createIIDClientDataset():
    clientDataIndex={}
    with open("./datasets/cifar100/Sample", "rb") as f:
        Sample=pickle.load(f) 

    samplePerClient = len(Sample)//args.n_clients
    index=torch.tensor(list(range(len(Sample))))

    for c in range(args.n_clients):
        if c==args.n_clients-1:
            clientDataIndex[c] = torch.tensor(index[c*samplePerClient:])
        else:
            clientDataIndex[c] = torch.tensor(index[c*samplePerClient:(c+1)*samplePerClient])
    
    return clientDataIndex


class Aggregator:
    def __init__(self, device):
        self.device=device
        self.count = 0
        self.localCenters={}
        self.label_pred={}
        self.local_estimates=[]

    def collect(self, cluster_centers, label_pred, user):
        self.localCenters[user] = np.array(cluster_centers)
        self.local_estimates.append(cluster_centers)
        self.label_pred[user] = label_pred

    
    def globalClusterAvg(self):
        nmiList, ariList, fList, accList= [],  [],  [], []
        kmeans = KMeans(n_clusters=20, init=self.localCenters[0])
        local_estimates = np.concatenate(self.local_estimates, axis=0)
        kmeans.fit(local_estimates)
        label_pred = kmeans.labels_
        for c in range(args.n_clients):
            globalCluster = label_pred[c*20: (c+1)*20]
            localCluster=self.label_pred[c]
            preCluster = np.zeros(len(localCluster), dtype=int)
            for i in range(20):
                index=np.where(localCluster==i)[0]
                preCluster[index] = globalCluster[i]
            trueLabel = datasets[c].labels.numpy()
            trueLabel=list(trueLabel)
            preCluster=list(preCluster)
            nmi, ari, f, acc = evaluation.evaluateNoiid(trueLabel, preCluster)
            nmiList.append(nmi)
            ariList.append(ari)
            fList.append(f)
            accList.append(acc)
        print('Average NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'\
          .format(sum(nmiList)/len(nmiList), sum(ariList)/len(ariList), sum(fList)/len(fList), sum(accList)/len(accList)))
    

    def globalCluster(self):
        kmeans = KMeans(n_clusters=args.num_class)
        local_estimates = np.concatenate(self.local_estimates, axis=0)
        kmeans.fit(local_estimates)
        label_pred = kmeans.labels_
        trueLabels=[]
        preClusters=[]
        for c in range(args.n_clients):
            globalCluster = label_pred[c*args.classes_per_user: (c+1)*args.classes_per_user]
            localCluster=self.label_pred[c]
            preCluster = np.zeros(len(localCluster), dtype=int)
            for i in range(args.classes_per_user):
                index=np.where(localCluster==i)[0]
                preCluster[index] = globalCluster[i]
            trueLabel = datasets[c].labels.numpy()
            trueLabel=list(trueLabel)
            preCluster=list(preCluster)
            trueLabels.extend(trueLabel)
            preClusters.extend(preCluster)
        nmi, ari, f, acc = evaluation.evaluate(trueLabels, preClusters)
        print(' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
        

def clipGrad(cluster_centers):
    sigma = get_noise_multiplier(
                target_epsilon = 8,
                target_delta = 1e-3,
                sample_rate = 1,
                epochs = 1,
            )
    clip_bound=15
    cluster_centers=torch.tensor(cluster_centers)
    norm = torch.norm(cluster_centers, p=2)
    print("Norm: ",norm)
    cluster_centers = torch.div(cluster_centers, max(1, torch.div(norm, clip_bound)))
    noise = torch.normal(
                mean=0,
                std=sigma * clip_bound,
                size=cluster_centers.size(),
                device=cluster_centers.device,
                dtype=cluster_centers.dtype,
            )
    cluster_centers+=noise
    return np.array(cluster_centers)

def train(agg, datasets):
    for c in range(args.n_clients):
        dataLoader = DataLoader(
            datasets[c],
            batch_size=args.mini_bs,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        features=[]
        # labels=[]
        for batch_idx, (x, y) in enumerate(dataLoader):
            x = x.to(device)
            # features.extend(model(x).tolist())
            features.extend(model.forward_rep(x).tolist())

        features=np.array(features)
        print("begin", c)
        kmeans = KMeans(n_clusters=args.classes_per_user)
        kmeans.fit(features)
        cluster_centers = kmeans.cluster_centers_
        label_pred = kmeans.labels_

        # fcm = FCM(n_clusters=20, max_iter=1000)
        # fcm.fit(features)
        # cluster_centers = fcm.centers
        # fcm.trained=True
        # fcm.setCenters(cluster_centers)
        # label_pred = fcm.predict(features)

        # cluster_centers, kmeans = awasthisheffet(features, 10)
        # label_pred = kmeans.labels_

        agg.collect(cluster_centers, label_pred, c) #共享内存
        
    agg.globalCluster()

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
    device= torch.device("cuda:0")
    print(device)
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config_DP_FL_Cifar100_kmeans.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # createNoIIDClientDataset()
    # # createIIDTrainAndTestDataset()
    
    # prepare data
    class_num = 20
    datasets=[]

    with open("./datasets/cifar100/Sample", "rb") as f:
        Sample = pickle.load(f)
    with open("./datasets/cifar100/Label20", "rb") as f:
        Label = pickle.load(f).cpu()

    transformation = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=(args.image_size, args.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.5)
        ]
    transformation = torchvision.transforms.Compose(transformation)

    # with open("./datasets/cifar10/clientDataIndex", "rb") as f:
    #     clientDataIndex = pickle.load(f)
    
    clientDataIndex = createIIDClientDataset()

    agg = Aggregator(device)

    datasets=[Cifar100Dataset(Sample[clientDataIndex[c]], Label[clientDataIndex[c]], transformation) for c in range(args.n_clients)]

    # model = timm.create_model("resnet18", pretrained=True, num_classes=0)
    # model = model.to(device)
    
    res = resnet.get_resnet("ResNet18", 6)
    model = network.Network_perCluster(res, args.feature_dim, args.num_class, args.r_proj)
    model = ModuleValidator.fix(model)
    name='save/Img-10-pretrain-transform/checkpoint_532.tar'
    checkpoint = torch.load(name, map_location=torch.device('cpu'))['net']
    model.load_state_dict(checkpoint, strict=False)
    model=model.to(device)

    # train
    train(agg, datasets)



# if __name__ == "__main__": # no-iid!!!!!!!!!
#     import warnings
#     warnings.filterwarnings("ignore")
#     device= torch.device("cuda:3")
#     print(device)
#     parser = argparse.ArgumentParser()
#     config = yaml_config_hook("config/config_DP_FL_Cifar100_kmeans.yaml")
#     for k, v in config.items():
#         parser.add_argument(f"--{k}", default=v, type=type(v))
#     args = parser.parse_args()
#     print(args)
#     if not os.path.exists(args.model_path):
#         os.makedirs(args.model_path)

#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     np.random.seed(args.seed)
    
#     # createNoIIDClientDataset()
#     # createNoIIDTrainAndTestDataset_cifar100()
    
#     # prepare data
#     datasets=[]

#     with open("./datasets/cifar100/Sample_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients), "rb") as f:
#         Sample = pickle.load(f)
#     with open("./datasets/cifar100/Label_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients), "rb") as f:
#         Label = pickle.load(f)
#     print("len label:",len(Label))

#     # with open("./datasets/cifar100/Sample_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients), "rb") as f:
#     #     Sample = pickle.load(f)
#     # with open("./datasets/cifar100/Label_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients), "rb") as f:
#     #     Label = pickle.load(f)
#     # print(len(Label))

#     transformation = [
#             torchvision.transforms.ToPILImage(),
#             torchvision.transforms.Resize(size=(args.image_size, args.image_size)),
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(mean=0.5, std=0.5)
#         ]
#     transformation = torchvision.transforms.Compose(transformation)
    
#     clientDataIndex = createclientDataIndex("./datasets/cifar100/Sample_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients))
#     agg = Aggregator(device)

#     datasets=[Cifar100Dataset(Sample[clientDataIndex[c]], Label[clientDataIndex[c]], transformation) for c in range(args.n_clients)]
    
#     # model = timm.create_model("resnet18", pretrained=True, num_classes=0)
#     # model = model.to(device)

#     res = resnet.get_resnet("ResNet18", 6)
#     model = network.Network_perCluster(res, args.feature_dim, args.num_class, args.r_proj)
#     model = ModuleValidator.fix(model)
#     name='save/Img-10-pretrain-transform/checkpoint_532.tar'
#     checkpoint = torch.load(name, map_location=torch.device('cpu'))['net']
#     model.load_state_dict(checkpoint, strict=False)
#     model=model.to(device)

#     # train
#     train(agg, datasets)
#     # nomodel_train(agg, datasets)
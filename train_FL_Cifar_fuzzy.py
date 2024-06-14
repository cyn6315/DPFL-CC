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

class Cifar10Dataset_nomodel(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        # 初始化文件路径或文件名列表。
        # 初始化该类的一些基本参数。
        self.samples = samples
        self.labels = labels

    def __len__(self):
        # 返回数据集的总大小
        return self.samples.shape[0]

    def __getitem__(self, index):
        return (self.samples[index], self.labels[index])


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
        return (self.transform(self.samples[index]), self.labels[index])

class Cifar10DatasetTest(torch.utils.data.Dataset):
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
    
def createIIDTrainAndTestDataset():
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=transformation)
    testset = torchvision.datasets.CIFAR10(root=args.dataset_dir, train=False, download=True, transform=transformation)
    dataset = data.ConcatDataset([trainset, testset])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=4) 
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx==0:
            Sample=inputs.clone()
            Label=targets.clone()
        else:
            Sample=torch.cat([Sample, inputs.clone()], dim=0)
            Label=torch.cat([Label, targets.clone()], dim=0)
    
    with open("./datasets/cifar10/Sample", "wb") as f:
        pickle.dump(Sample, f)  
    with open("./datasets/cifar10/Label", "wb") as f:
        pickle.dump(Label, f) 
    


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



def createNoIIDClientDataset():
    clientDataIndex={}
    for c in range(args.n_clients):
        clientDataIndex[c]=[]

    with open("./datasets/cifar10/Label", "rb") as f:
        Label=pickle.load(f) 
    totalIndex=np.array(range(args.n_clients))
    np.random.shuffle(totalIndex)
    halfClients=args.n_clients//2
    count=0
    for label in range(10):
        sampleIndex=np.where(np.array(Label)==label)[0]
        if count % 2==0:
            clients=np.array(totalIndex[0:halfClients])
        else:
            clients=np.array(totalIndex[halfClients:])
            np.random.shuffle(totalIndex)
        samplePerClient = len(sampleIndex)//len(clients)
        for i in range(len(clients)):
            c=clients[i]
            if i==len(clients)-1:
                c_index=list(sampleIndex[i*samplePerClient:])
            else:
                c_index=list(sampleIndex[i*samplePerClient:(i+1)*samplePerClient])
            clientDataIndex[c].extend(c_index)
        count+=1
    for c in range(args.n_clients):
        clientDataIndex[c]=torch.tensor(clientDataIndex[c])
    
    with open("./datasets/cifar10/clientDataIndex", "wb") as f:
        pickle.dump(clientDataIndex, f) 


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
    
    def globalCluster(self):
        kmeans = KMeans(n_clusters=args.num_class)
        local_estimates = np.concatenate(self.local_estimates, axis=0)
        kmeans.fit(local_estimates)
        self.local_estimates=[]
        # label_pred = kmeans.labels_
        return kmeans.cluster_centers_
    
    def globalCluster2(self):
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
        return kmeans.cluster_centers_


def train(agg, datasets):
    model.eval()
    for i in tqdm(range(5)):
        for c in range(args.n_clients):
            dataLoader = DataLoader(
                datasets[c],
                batch_size=args.mini_bs,
                shuffle=False,
                drop_last=False,
                num_workers=args.workers,
            )
            features=[]
            for batch_idx, (x, y) in enumerate(dataLoader):
                x = x.to(device)
                # features.extend(model(x).tolist())
                features.extend(model.forward_rep(x).tolist())
            features=np.array(features)
            if i==0:
                fcm = FCM(n_clusters=args.classes_per_user)
                fcm.fit(features)
            else:
                fcm = FCM(n_clusters=args.classes_per_user)
                fcm.fit_with_init(features, init=global_centers)
            cluster_centers = fcm.centers
            label_pred = fcm.predict(features)
            # print(label_pred)
            agg.collect(cluster_centers, label_pred, c) #共享内存
            
        global_centers = agg.globalCluster()
        trueLabels=[]
        preClusters=[]
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
            for batch_idx, (x, y) in enumerate(dataLoader):
                x = x.to(device)
                # features.extend(model(x).tolist())
                features.extend(model.forward_rep(x).tolist())
            features=np.array(features)
            fcm = FCM(n_clusters=args.num_class)
            fcm.trained=True
            fcm.setCenters(global_centers)
            label_pred = fcm.predict(features)
            trueLabel = datasets[c].labels.numpy()
            trueLabel=list(trueLabel)
            label_pred = list(label_pred)
            trueLabels.extend(trueLabel)
            preClusters.extend(label_pred)
        nmi, ari, f, acc = evaluation.evaluate(trueLabels, preClusters)
        print("round",i)
        print(' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))

def nomodel_train(agg, datasets):
    global_centers=0
    for i in range(1):
        for c in range(args.n_clients):
            dataLoader = DataLoader(
                datasets[c],
                batch_size=args.mini_bs,
                shuffle=False,
                drop_last=False,
                num_workers=args.workers,
            )
            features=[]
            for batch_idx, (x, y) in enumerate(dataLoader):
                x=x.reshape((-1,3072))
                features.extend(x.tolist())
            features=np.array(features)
            print("begin",c)
            fcm = FCM(n_clusters=args.classes_per_user, max_iter=800)
            fcm.fit(features)
            label_pred = fcm.predict(features)
            cluster_centers = fcm.centers
            agg.collect(cluster_centers, label_pred, c) #共享内存
        
        global_centers = agg.globalCluster2()    

        # global_centers = agg.globalCluster()
        # trueLabels=[]
        # preClusters=[]
        # for c in range(args.n_clients):
        #     dataLoader = DataLoader(
        #         datasets[c],
        #         batch_size=args.mini_bs,
        #         shuffle=False,
        #         drop_last=False,
        #         num_workers=args.workers,
        #     )
        #     features=[]
        #     for batch_idx, (x, y) in enumerate(dataLoader):
        #         x=x.reshape((-1,3072))
        #         features.extend(x.tolist())
        #     features=np.array(features)
        #     fcm = FCM(n_clusters=10)
        #     fcm.trained=True
        #     fcm.setCenters(global_centers)
        #     label_pred = fcm.predict(features)
        #     trueLabel = datasets[c].labels.numpy()
        #     trueLabel=list(trueLabel)
        #     label_pred = list(label_pred)
        #     trueLabels.extend(trueLabel)
        #     preClusters.extend(label_pred)
        # nmi, ari, f, acc = evaluation.evaluate(trueLabels, preClusters)
        # print("round",i)
        # print(' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))

# if __name__ == "__main__": #iid
#     import warnings
#     warnings.filterwarnings("ignore")
#     device= torch.device("cuda:4")
#     print(device)
#     parser = argparse.ArgumentParser()
#     config = yaml_config_hook("config/config_DP_FL_Cifar10_fuzzy.yaml")
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
#     # # createIIDTrainAndTestDataset()
    
#     # prepare data
#     datasets=[]

#     with open("./datasets/cifar10/Sample", "rb") as f:
#         Sample = pickle.load(f)
#     with open("./datasets/cifar10/Label", "rb") as f:
#         Label = pickle.load(f)

#     # with open("./datasets/cifar100/Sample", "rb") as f:
#     #     Sample = pickle.load(f)
#     # with open("./datasets/cifar100/Label20", "rb") as f:
#     #     Label = pickle.load(f)

#     transformation = [
#             torchvision.transforms.ToPILImage(),
#             torchvision.transforms.Resize(size=(args.image_size, args.image_size)),
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(mean=0.5, std=0.5)
#         ]
#     transformation = torchvision.transforms.Compose(transformation)
    
#     clientDataIndex = createclientDataIndex("./datasets/cifar10/Sample")

#     agg = Aggregator(device)

#     # datasets=[Cifar10Dataset_nomodel(Sample[clientDataIndex[c]], Label[clientDataIndex[c]]) for c in range(args.n_clients)]
    
#     datasets=[Cifar10Dataset(Sample[clientDataIndex[c]], Label[clientDataIndex[c]], transformation) for c in range(args.n_clients)]

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


if __name__ == "__main__": #no-iid
    import warnings
    warnings.filterwarnings("ignore")
    device= torch.device("cuda:4")
    print(device)
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config_DP_FL_Cifar10_fuzzy.yaml")
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
    datasets=[]

    # with open("./datasets/cifar10/Sample_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients), "rb") as f:
    #     Sample = pickle.load(f)
    # with open("./datasets/cifar10/Label_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients), "rb") as f:
    #     Label = pickle.load(f)
    # print("len label:",len(Label))

    with open("./datasets/cifar100/Sample_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients), "rb") as f:
        Sample = pickle.load(f)
    with open("./datasets/cifar100/Label_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients), "rb") as f:
        Label = pickle.load(f)
    print("len label:",len(Label))

    transformation = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=(args.image_size, args.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.5)
        ]
    transformation = torchvision.transforms.Compose(transformation)

    
    clientDataIndex = createclientDataIndex("./datasets/cifar100/Sample_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients))

    agg = Aggregator(device)

    datasets=[Cifar10Dataset(Sample[clientDataIndex[c]], Label[clientDataIndex[c]], transformation) for c in range(args.n_clients)]

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

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

    
    def globalCluster(self):
        kmeans = KMeans(n_clusters=20, max_iter=1000, init=self.localCenters[0])
        local_estimates = np.concatenate(self.local_estimates, axis=0)
        kmeans.fit(local_estimates)
        self.local_estimates=[]
        # label_pred = kmeans.labels_
        return kmeans.cluster_centers_
    
    def globalCluster2(self):
        kmeans = KMeans(n_clusters=20, max_iter=1000, init=self.localCenters[0])
        local_estimates = np.concatenate(self.local_estimates, axis=0)
        kmeans.fit(local_estimates)
        label_pred = kmeans.labels_
        trueLabels=[]
        preClusters=[]
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
            trueLabels.extend(trueLabel)
            preClusters.extend(preCluster)
        nmi, ari, f, acc = evaluation.evaluate(trueLabels, preClusters)
        print(' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))


def train(agg, datasets):
    model.eval()
    for i in tqdm(range(1)):
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

    

def inference(loader, testmodel, device):
    testmodel.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = testmodel.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    # print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def testiid(round, testmodel):
    nmiList, ariList, fList, accList= [],  [],  [], []
    transformation = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=(args.image_size, args.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.5)
        ]
    transformation = torchvision.transforms.Compose(transformation)
    dataset = Cifar10DatasetTest(Sample, Label, transformation)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=300,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    print("### Creating features from model ###")
    X, Y = inference(data_loader, testmodel, device)
    for c in range(args.n_clients):
        client_X = np.array(X[np.array(clientDataIndex[c])])
        client_Y = np.array(Y[np.array(clientDataIndex[c])])
        nmi, ari, f, acc = evaluation.evaluateNoiid(client_Y, client_X)
        nmiList.append(nmi)
        ariList.append(ari)
        fList.append(f)
        accList.append(acc)
        # print('Rround '+ str(round)+ ' User '+str(c) + ' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    print('Average NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'\
          .format(sum(nmiList)/len(nmiList), sum(ariList)/len(ariList), sum(fList)/len(fList), sum(accList)/len(accList)))



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    device= torch.device("cuda:4")
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
    print(Label)
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

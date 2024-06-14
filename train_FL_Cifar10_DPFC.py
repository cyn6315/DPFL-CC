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
import math


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



class MFModel (torch.nn.Module):
    def __init__(self, n_features, n_samples):
        super (MFModel, self).__init__ ()
        self.W = torch.nn.Embedding(args.num_class, n_features)
        self.H = torch.nn.Embedding(n_samples, args.num_class)
        self.H.weight.data = torch.clamp(self.H.weight.data, min=0, max=1)

    def embeddings(self, H_ids):
        H_emb = self.H(H_ids)
        W_emd = self.W(torch.arange(0, 10).to(device)).t()
        return H_emb, W_emd

    def forward(self, X, H_ids):
        self.H.weight.data = torch.clamp(self.H.weight.data, min=0, max=1)
        H_ids = H_ids.to(device)
        n_samples = len(H_ids)
        H_emb, W_emd = self.embeddings(H_ids)
        loss = torch.sum((X - torch.mm(W_emd, H_emb.t()))** 2)/n_samples
        trace = (torch.mm(torch.mm(H_emb.t(), torch.ones(n_samples,n_samples).to(device)), H_emb).trace()-torch.sum(H_emb** 2))/n_samples
        loss+= args.thou/2*trace
        loss+= (args.miuh+args.miuw)/2*(torch.sum(H_emb** 2) + torch.sum(W_emd** 2))/n_samples
        return loss


class Aggregator:
    def __init__(self, device):
        self.mfmodel = MFModel(512, 60000).to(device)
        self.count = 0
        self.modelUpdate={}
        self.optimizer = torch.optim.SGD(self.mfmodel.parameters(), lr=args.global_lr, momentum=0.9)

    def update(self):
        print("count",self.count )
        if self.count == 0:
            return
        self.optimizer.zero_grad()
        normSum=0
        for name, param in self.mfmodel.named_parameters ():
            if name in self.modelUpdate:
                noise = torch.normal(
                    mean=0,
                    std=sigma * args.clip_bound * math.sqrt(400),
                    size=self.modelUpdate[name].size(),
                    device=self.modelUpdate[name].device,
                    dtype=self.modelUpdate[name].dtype,
                )
                updated = torch.div(self.modelUpdate[name], self.count)
                param.grad = updated + torch.div(noise, 400)
                if normSum == 0:
                    normSum = torch.sum(torch.pow(updated, exponent=2))
                else:
                    normSum += torch.sum(torch.pow(updated, exponent=2))
            else:
                param.grad = None
        self.optimizer.step()
        self.count = 0
        self.modelUpdate.clear()
        print("updated norm: ", torch.sqrt(normSum))

    def collect(self, model_grad):
        self.count+=1
        for name in model_grad:
            if name not in self.modelUpdate:
                self.modelUpdate[name] = model_grad[name].data.clone()
            else:
                self.modelUpdate[name] += model_grad[name].data.clone()

def setParaFromAgg(model, agg):
    for name, param in model.named_parameters ():
        if "W" in name:
            param.data = torch.tensor(agg[name].clone())


def getModelUpdate(gradDict, netcopy, net, localParams):
    flag=0
    for name, param in net.named_parameters ():
        if param.requires_grad and (localParams not in name):
            gradDict[name] = (netcopy[name].data - param.data).clone()
            if flag==0:
                normSum = torch.sum(torch.pow(gradDict[name], exponent=2))
                flag=1
            else:
                normSum += torch.sum(torch.pow(gradDict[name], exponent=2))
    return normSum

def clipGrad(gradDict, normSum): 
    norm=torch.sqrt(normSum)
    print("Norm: ",norm)
    scale=torch.div(norm, args.clip_bound)
    for name in gradDict:
        gradDict[name] = torch.div(gradDict[name], max(1, scale.item()))

def train(agg, datasets, round):
    mfmodel.train()
    model.eval()
    for c in range(args.n_clients):
        dataLoader = DataLoader(
            datasets[c],
            batch_size=args.mini_bs,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        lossPerUser=0
        setParaFromAgg(mfmodel, agg.mfmodel.state_dict())
        optimizer=torch.optim.SGD(mfmodel.parameters(),  lr=args.learning_rate, momentum=0.9)
        modelUpdateDict={}
        truestep=0
        for i in range(args.local_epoch):
            for batch_idx, (x, y) in enumerate(dataLoader):
                truestep+=1
                x = x.to(device)
                features=model(x)
                loss = mfmodel(features.t(), clientDataIndex[c][batch_idx*args.mini_bs:(batch_idx+1)*args.mini_bs])
                lossPerUser+=loss
                optimizer.zero_grad ()
                loss.backward()
                optimizer.step()
        normSum = getModelUpdate(modelUpdateDict, agg.mfmodel.state_dict (), mfmodel, "H")
        clipGrad(modelUpdateDict, normSum)
        agg.collect(modelUpdateDict) #共享内存
        print('Round: ', round, "User:", c, 'Train Loss: %.3f' % (lossPerUser/truestep))
        
    agg.mfmodel.train()
    agg.update()


def testiid(testmodel):
    feature_vector = torch.argmax(mfmodel.H.weight.data, dim=1).cpu()
    labels_vector = Label.numpy()
    print(feature_vector, labels_vector)
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)

    print("### Creating features from model ###")
    nmi, ari, f, acc = evaluation.evaluate(labels_vector, feature_vector)
    print('Global '+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))

    # nmiList, ariList, fList, accList= [],  [],  [], []
    # for c in range(args.n_clients):
    #     client_X = np.array(X[np.array(clientDataIndex[c])])
    #     client_Y = np.array(Y[np.array(clientDataIndex[c])])
    #     nmi, ari, f, acc = evaluation.evaluate(client_Y, client_X)
    #     nmiList.append(nmi)
    #     ariList.append(ari)
    #     fList.append(f)
    #     accList.append(acc)
    # print('Average NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'\
    #       .format(sum(nmiList)/len(nmiList), sum(ariList)/len(ariList), sum(fList)/len(fList), sum(accList)/len(accList)))

def decide_requires_grad(model,finetuneParams):
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if finetuneParams in name :
            param.requires_grad_(True)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    device= torch.device("cuda:2")
    print(device)
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config_DP_FL_Cifar10_DPFC.yaml")
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
    # createIIDTrainAndTestDataset()
    
    # prepare data
    datasets=[]

    # with open("./datasets/cifar10/Sample", "rb") as f:
    #     Sample = pickle.load(f)
    # with open("./datasets/cifar10/Label", "rb") as f:
    #     Label = pickle.load(f).cpu()

    with open("./datasets/cifar10/Sample_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients), "rb") as f:
        Sample = pickle.load(f)
    with open("./datasets/cifar10/Label_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients), "rb") as f:
        Label = pickle.load(f)
    print("len label:",len(Label))

    transformation = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=(args.image_size, args.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.5)
        ]
    transformation = torchvision.transforms.Compose(transformation)
    
    # clientDataIndex = createclientDataIndex("./datasets/cifar10/Sample")
    clientDataIndex = createclientDataIndex("./datasets/cifar10/Sample_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients))

    agg = Aggregator(device)

    datasets=[Cifar10Dataset(Sample[clientDataIndex[c]], Label[clientDataIndex[c]], transformation) for c in range(args.n_clients)]

    model = timm.create_model("resnet18", pretrained=True, num_classes=0)
    model = model.to(device)
    mfmodel = MFModel(512, 59200).to(device)

    sigma = get_noise_multiplier(
                target_epsilon = args.epsilon,
                target_delta = 1e-3,
                sample_rate = 1,
                epochs = args.epochs,
            )
    print("sigma:", sigma)

    for name, param in mfmodel.named_parameters():
        print(name, param.numel(), param.size())
    
    # train
    for i in range(args.epochs):
        train(agg, datasets, i)
        testiid(mfmodel)
        if i<4:
            args.global_lr=args.global_lr*0.7
        else:
            args.global_lr=1.5

import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook
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
import shutil
import copy


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
        return ([self.transform(self.samples[index]) for i in range(args.local_epoch)])

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
    


def createIIDClientDataset():
    clientDataIndex={}
    with open("./datasets/cifar10/Sample", "rb") as f:
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
    
    with open("./datasets/cifar10/clientDataIndex"+str(args.n_clients), "wb") as f:
        pickle.dump(clientDataIndex, f) 


class Aggregator:
    def __init__(self, device):
        res = resnet.get_resnet(args.resnet)
        model = network.Network(res, args.feature_dim, class_num)
        model = ModuleValidator.fix(model)
        # checkpoint = torch.load('save/Img/checkpoint_450.tar',map_location=torch.device('cpu'))['net']
        checkpoint = torch.load('save/Img-10-FL/checkpoint_500.tar',map_location=torch.device('cpu'))['net']
        model.load_state_dict(checkpoint)
        self.model = model.to(device)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.global_lr, momentum=0.3)
        self.device=device
        self.count = 0
        self.modelUpdate={}

    def update(self):
        print("count",self.count )
        if self.count == 0:
            return
        # self.optimizer.zero_grad()
        for name, param in self.model.named_parameters ():
            if name in self.modelUpdate:
                param.data = param.data + args.global_lr*(torch.div(self.modelUpdate[name], self.count)) 
                # param.grad = torch.div(self.modelUpdate[name], self.count)
        # self.optimizer.step()
        self.count = 0
        self.modelUpdate.clear()

    def collect(self, model_grad):
        self.count+=1
        for name in model_grad:
            if name not in self.modelUpdate:
                self.modelUpdate[name] = model_grad[name].data.clone()
            else:
                self.modelUpdate[name] += model_grad[name].data.clone()


def getModelUpdate(gradDict, netcopy, net):
    flag=0
    normSum=0
    for name, param in net.named_parameters ():
        if param.requires_grad:
            gradDict[name] = (param.data - netcopy[name].data).clone()
            if flag==0:
                normSum = torch.sum(torch.pow(gradDict[name], exponent=2))
                flag=1
            else:
                normSum += torch.sum(torch.pow(gradDict[name], exponent=2))
    return normSum


def setParaFromAgg(model, agg):
    for name, param in model.named_parameters ():
        param.data = agg[name].data.clone()


def clipGrad(gradDict, normSum): 
    norm=torch.sqrt(normSum)
    print("Norm: ",norm)
    scale=torch.div(norm, args.clip_bound)
    for name in gradDict:
        gradDict[name] = torch.div(gradDict[name], max(1, scale.item()))


def train(agg, round):
    agg.model.train()
    train_loss = 0
    setParaFromAgg(modelcopy, agg.model.state_dict())
    for batch_idx, (x_list) in enumerate(dataLoader):
        model.train()
        setParaFromAgg(model, agg.model.state_dict())

        # optCheckpoint = torch.load(os.path.join("save/clientsOpt2", "client_{}.tar".format(batch_idx)), map_location=device)
        # optimizer.load_state_dict(optCheckpoint['optimizer'])

        optimizer.load_state_dict(optimizerList[batch_idx])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        optimizer.zero_grad()
        c=batch_idx
        gradDict={}
        lossPerUser=0
        true_epoch=0
        for i in range(args.local_epoch):
            for j in range(i+1, args.local_epoch):
                true_epoch += 1 
                x_i = x_list[i].to(device)
                x_j = x_list[j].to(device)
                z_i, z_j, c_i, c_j = model(x_i, x_j)
                loss_instance = criterion_instance(z_i, z_j)
                loss_cluster = criterion_cluster(c_i, c_j)
                loss = loss_instance + loss_cluster
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lossPerUser += loss.item()
        normSum = getModelUpdate(gradDict, modelcopy.state_dict (), model) #model - modelcopy    
        print("Norm: ",torch.sqrt(normSum))
        # clipGrad(gradDict, normSum)
        train_loss+=lossPerUser
        agg.collect(gradDict) #共享内存

        # state = {'optimizer': optimizer.state_dict()}
        # torch.save(state, os.path.join("save/clientsOpt2", "client_{}.tar".format(batch_idx)))

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        optimizerList[batch_idx]=optimizer.state_dict()

        print('Round: ', round, "User:", c, 'Train Loss: %.3f' % (lossPerUser/true_epoch))
        if batch_idx==29:
            break
    agg.update()
    print('*********Round: ', round, 'Train Loss: %.3f' % (train_loss/((batch_idx+1)*true_epoch)))
    


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


def test(round, testmodel):
    train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=True,
            download=True,
            transform=transform.Transforms(size=args.image_size, mean=0.5, std=0.5).test_transform,
        )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_dir,
        train=False,
        download=True,
        transform=transform.Transforms(size=args.image_size, mean=0.5, std=0.5).test_transform,
    )
    dataset = data.ConcatDataset([train_dataset, test_dataset])
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    print("### Creating features from model ###")
    X, Y = inference(data_loader, testmodel, device)
    nmi, ari, f, acc = evaluation.evaluate(Y, X)
    print('Rround '+ str(round)+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    device= torch.device("cuda:2")
    print(device)
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config_DP_FL_Cifar_fix.yaml")
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
    class_num = 10
    datasets=[]

    with open("./datasets/cifar10/Sample", "rb") as f:
        Sample = pickle.load(f)
    with open("./datasets/cifar10/Label", "rb") as f:
        Label = pickle.load(f)
    s=0.5
    transformation = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomResizedCrop(size=args.image_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ]
    transformation.append(torchvision.transforms.ToTensor())
    transformation.append(torchvision.transforms.Normalize(mean=0.5, std=0.5))
    transformation = torchvision.transforms.Compose(transformation)
    dataset = Cifar10Dataset(Sample, Label, transformation)
    dataLoader = DataLoader(
            dataset,
            batch_size=args.mini_bs,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=True,
        )
    
    clientDataIndex = createIIDClientDataset()

    # # optimizer / loss
    
    criterion_instance = contrastive_loss.InstanceLoss(args.mini_bs, args.instance_temperature, device).to(
        device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, device).to(device)

    agg = Aggregator(device)
    model_fp = os.path.join("save/Cifar-10-FL-clientlevel", "checkpoint_{}.tar".format(399))
    checkpoint = torch.load(model_fp, map_location=device)
    agg.model.load_state_dict(checkpoint['net'])
    # testiid(0, agg.model)
    
    rescopy = resnet.get_resnet(args.resnet)
    modelcopy = network.Network(rescopy, args.feature_dim, class_num)
    modelcopy = ModuleValidator.fix(modelcopy)
    modelcopy = modelcopy.to(device)

    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model = ModuleValidator.fix(model)
    model = model.to(device)
    

    print('Number of total parameters: ', sum([p.numel() for p in model.parameters()]))
    print('Number of trainable p  arameters: ', sum([p.numel() for p in model.parameters() if p.requires_grad]))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizerList={}
    for i in range(args.n_clients):
        optimizerList[i]=optimizer.state_dict()

    # for i in range(args.n_clients):
    #     out = os.path.join("save/clientsOpt2", "client_{}.tar".format(i))
    #     state = {'optimizer': optimizer.state_dict()}
    #     torch.save(state, out)

    # train
    for epoch in range(args.start_epoch, args.epochs):
        train(agg, epoch)
        testiid(epoch, agg.model)
        if (epoch+1) %2 ==0:
            save_model2(args, agg.model, epoch)
            
    save_model2(args, agg.model, args.epochs)
    testiid(args.epochs, agg.model)

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
import PIL
from tqdm import tqdm 
import shutil
import copy
import loralib as lora
import math



def save_model2(args, model, current_epoch):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)


class Cifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, transform, transform_ori, transformation_strong):
        # 初始化文件路径或文件名列表。
        # 初始化该类的一些基本参数。
        self.samples = samples
        self.labels = labels
        self.transform = transform
        self.transform_ori = transform_ori
        self.transformation_strong = transformation_strong

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
        res = resnet.get_resnet(args.resnet, args.r_conv)
        model = network.Network(res, args.feature_dim, class_num, args.r_proj)
        model = ModuleValidator.fix(model)
        name='save/Img-10-pretrain-transform/checkpoint_532.tar'
        # name='save/Img-10-pretrain-transform-cluster/checkpoint_348.tar'
        checkpoint = torch.load(name,map_location=torch.device('cpu'))['net']
        print(name)
        model.load_state_dict(checkpoint, strict=False)
        self.model = model.to(device)
        self.device=device
        self.count = 0
        self.modelUpdate={}
        decide_requires_grad(self.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.global_lr, momentum=args.momentum)

    def update(self):
        print("count",self.count )
        if self.count == 0:
            return
        self.optimizer.zero_grad()
        normSum=0
        for name, param in self.model.named_parameters ():
            if name in self.modelUpdate:
                noise = torch.normal(
                    mean=0,
                    std=sigma * args.clip_bound,
                    size=self.modelUpdate[name].size(),
                    device=self.modelUpdate[name].device,
                    dtype=self.modelUpdate[name].dtype,
                )
                updated = torch.div(self.modelUpdate[name], self.count)
                # param.data = param.data + args.global_lr*(updated + torch.div(noise, 600))
                param.grad = updated + torch.div(noise, args.n_clients)
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


def getModelUpdate(gradDict, netcopy, net):
    flag=0
    for name, param in net.named_parameters ():
        if param.requires_grad:
            gradDict[name] = (netcopy[name].data - param.data).clone()
            if flag==0:
                normSum = torch.sum(torch.pow(gradDict[name], exponent=2))
                flag=1
            else:
                normSum += torch.sum(torch.pow(gradDict[name], exponent=2))
    return normSum

def calculateNormSuqre(gradDict):
    flag=0
    for name in gradDict:
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


def maskedParam(model, maskDict): 
    maskConv=["downsample","resnet.conv1.weight","resnet.layer1.0.conv1.weight","resnet.layer1.0.conv2.weight",\
                    "resnet.layer1.1.conv1.weight","resnet.layer1.1.conv2.weight", "resnet.layer2.0.conv1.weight"]
    for name, param in model.named_parameters ():
        for maskName in maskConv:
            if maskName in name:
                paramFlatten = torch.flatten(param.data)
                ratio=0.1
                if "resnet.conv1.weight" in name:
                    ratio = 0.8
                elif "resnet.layer2.0.downsample.0.weight" in name:
                    ratio = 0.7
                elif "resnet.layer3.0.downsample.0.weight" in name:
                    ratio = 0.6
                elif "resnet.layer4.0.downsample.0.weight" in name:
                    ratio = 0.6
                else:
                    ratio = 0.5
                saved=int(len(paramFlatten)*ratio)
                _, indices = torch.topk(paramFlatten, k=saved)        
                mask = torch.zeros_like(paramFlatten)
                mask[indices] = 1
                maskDict[name] = mask.view(param.data.size())
                break

def calculateMask(gradDict, modelUpdateDict):
    maskDict={}
    for name in gradDict:
        res = torch.abs(gradDict[name]*modelUpdateDict[name])
        flatten = torch.flatten(res)
        saved=int(len(flatten)*args.sparsity)
        _, indices = torch.topk(flatten, k=saved)        
        mask = torch.zeros_like(flatten)
        mask[indices] = 1
        maskDict[name] = mask.view(gradDict[name].size())
    return maskDict

def saveGrad(model, gradDict):
    for name, param in model.named_parameters ():
        if param.requires_grad:
            gradDict[name] = torch.tensor(param.grad)


def maskModelUpdate(modelUpdateDict, maskDict):
    for name in modelUpdateDict():
        if name in maskDict:
            modelUpdateDict[name] = torch.mul(modelUpdateDict[name], maskDict[name])


def train(agg, round):
    agg.model.train()
    train_loss = 0
    normlist=[]
    # args.resnet_lr=args.resnet_lr*0.98
    # args.downsample_lr=args.downsample_lr*0.98
    # args.project_lr=args.project_lr*0.98
    for batch_idx, (x_list) in enumerate(dataLoader):
        model.train()
        setParaFromAgg(model, agg.model.state_dict())
        # optimizer=optimizerList[batch_idx]
        res_params2 = [param for name, param in model.named_parameters() if 'resnet' in name and 'lora' not in name]
        res_params2_down = [param for name, param in model.named_parameters() if 'resnet' in name and 'lora' in name]
        pro_params2 = [param for name, param in model.named_parameters() if 'resnet' not in name]
        optimizer=torch.optim.SGD([
            {'params': res_params2, 'lr': args.resnet_lr},
            {'params': res_params2_down, 'lr': args.downsample_lr},
            {'params': pro_params2, 'lr': args.project_lr},
        ], momentum=0.9) 
        optimizer.zero_grad()
        c=batch_idx
        modelUpdateDict={}
        gradDict={}
        lossPerUser=0
        true_epoch=0
        contrasive_pair = []
        
        for i in range(args.local_epoch):
            for j in range(i+1, args.local_epoch):
                contrasive_pair.append([i,j])

        pair_index=np.array(range(len(contrasive_pair)))
        np.random.shuffle(pair_index)
        
        for i in range(len(contrasive_pair)):
            true_epoch += 1 
            pair = contrasive_pair[pair_index[i]]
            x_i = x_list[pair[0]].to(device)
            x_j = x_list[pair[1]].to(device)
            z_i, z_j, c_i, c_j = model(x_i, x_j)
            loss_instance = criterion_instance(z_i, z_j)
            loss_cluster = criterion_cluster(c_i, c_j)
            deltaNorm = getModelUpdate(modelUpdateDict, agg.model.state_dict (), model)
            loss_blur = args.miu*max(0, deltaNorm-math.pow(args.clip_bound, 2))
            loss = loss_instance + loss_cluster + loss_blur
            loss.backward()
            if i == len(contrasive_pair)-1:
                saveGrad(model, gradDict)
            optimizer.step()
            optimizer.zero_grad()
            lossPerUser += loss.item()
        
        getModelUpdate(modelUpdateDict, agg.model.state_dict (), model) # model - modelcopy 
        maskDict = calculateMask(gradDict, modelUpdateDict)  
        maskModelUpdate(modelUpdateDict, maskDict)
        normSum = calculateNormSuqre(modelUpdateDict)
        normlist.append(torch.sqrt(normSum).item())
        clipGrad(modelUpdateDict, normSum)
        train_loss+=lossPerUser
        agg.collect(modelUpdateDict) #共享内存
        print('Round: ', round, "User:", c, 'Train Loss: %.3f' % (lossPerUser/(true_epoch)))
        
        # if (round+1)%10==0 or (round+1)==args.epochs:
        #     state = {'optimizer': optimizer.state_dict()}
        #     torch.save(state, os.path.join("save/clientsOpt", "client_{}.tar".format(batch_idx)))

        # if batch_idx==29:
        #     break
       
    if args.clip_bound > 1.9:
        args.clip_bound=args.clip_bound - 0.4
    else:
        args.clip_bound = sum(normlist)/len(normlist) - 0.2
        if args.clip_bound > 2:
            args.clip_bound = 1.6
        elif args.clip_bound > 1.5:
            args.clip_bound = 1.5
    print("clip_bound", args.clip_bound)
    
    # agg.model.train()
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
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]
    transformation = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(
                    (args.test_image_size, args.test_image_size), interpolation=PIL.Image.BICUBIC
                ),
            torchvision.transforms.CenterCrop(args.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
        ]
    transformation = torchvision.transforms.Compose(transformation)
    dataset = Cifar10DatasetTest(Sample, Label, transformation)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    print("### Creating features from model ###")
    X, Y = inference(data_loader, testmodel, device)
    nmi, ari, f, acc = evaluation.evaluateNoiid(Y, X)
    print('Global '+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    for c in range(args.n_clients):
        client_X = np.array(X[np.array(clientDataIndex[c])])
        client_Y = np.array(Y[np.array(clientDataIndex[c])])
        nmi, ari, f, acc = evaluation.evaluateNoiid(client_Y, client_X)
        nmiList.append(nmi)
        ariList.append(ari)
        fList.append(f)
        accList.append(acc)
        # if c==0 or c==1 or c==2:
        #     print('Rround '+ str(round)+ ' User '+str(c) + ' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
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


def decide_requires_grad(model):
    finetuneParams=["bn", "bias", "lora", "downsample.1"]
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if 'resnet' in name :
            for finetuneName in finetuneParams:
                if finetuneName in name:
                    param.requires_grad_(True)
        elif "cluster_projector" in name or "instance_projector" in name or "lora" in name:
            param.requires_grad_(True)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    device= torch.device("cuda:1")
    print(device)
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config_DP_FL_Cifar_fix_adam.yaml")
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

    cpu_num = 6 # 这里设置成你想运行的CPU个数
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
    os.environ["MKL_NUM_THREADS"] = str(cpu_num) # noqa
    torch.set_num_threads(cpu_num )
    
    # createNoIIDClientDataset()
    # # createIIDTrainAndTestDataset()
    
    # prepare data
    class_num = 10
    datasets=[]

    with open("./datasets/cifar10/Sample", "rb") as f:
        Sample = pickle.load(f)
    with open("./datasets/cifar10/Label", "rb") as f:
        Label = pickle.load(f)

    s = 0.5
    mean=[0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]
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

    transformation_ori = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(
                    (args.test_image_size, args.test_image_size), interpolation=PIL.Image.BICUBIC
                ),
        torchvision.transforms.CenterCrop(args.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])

    transformation_strong = transform.Transforms(size=args.image_size, mean=mean, std=std).strong_transform
    dataset = Cifar10Dataset(Sample, Label, transformation, transformation_ori, transformation_strong)
    dataLoader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=True,
        )
    
    clientDataIndex = createIIDClientDataset()

    # # optimizer / loss
    
    criterion_instance = contrastive_loss.InstanceLoss(args.mini_bs, args.instance_temperature, device).to(device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, device).to(device)
    criterion_KL = contrastive_loss.ClusterKLLoss(args.mini_bs, args.instance_temperature, device).to(device)

    agg = Aggregator(device)
    # loadpath="save/Cifar-10-DPFL-ResNet18-adam"
    # model_fp = os.path.join(loadpath, "checkpoint_{}.tar".format(30))
    # checkpoint = torch.load(model_fp, map_location=device)
    # agg.model.load_state_dict(checkpoint['net'], strict=False)
    # print(loadpath)
    
    rescopy = resnet.get_resnet(args.resnet, args.r_conv)
    modelcopy = network.Network(rescopy, args.feature_dim, class_num, args.r_proj)
    modelcopy = ModuleValidator.fix(modelcopy)
    modelcopy = modelcopy.to(device)

    res = resnet.get_resnet(args.resnet, args.r_conv)
    model = network.Network(res, args.feature_dim, class_num, args.r_proj)
    model = ModuleValidator.fix(model)
    model = model.to(device)

    for name, param in model.named_parameters():
        print(name, param.numel(), param.size())

    sigma = get_noise_multiplier(
                target_epsilon = args.epsilon,
                target_delta = 1e-3,
                sample_rate = 1,
                epochs = args.epochs,
            )
    print("sigma:", sigma)
    

    decide_requires_grad(model)

    print('Number of total parameters: ', sum([p.numel() for p in model.parameters()]))
    print('Number of trainable p  arameters: ', sum([p.numel() for p in model.parameters() if p.requires_grad]))


    # train
    for epoch in range(args.start_epoch, args.epochs):
        # testiid(epoch, agg.model)
        if epoch>args.start_epoch:
            testiid(epoch, agg.model)
        train(agg, epoch)
        save_model2(args, agg.model, epoch)
            
    save_model2(args, agg.model, args.epochs)
    testiid(args.epochs, agg.model)

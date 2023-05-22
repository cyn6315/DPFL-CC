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


def createIIDTrainAndTestDataset():
    file = open("./datasets/ImageNet-10.txt",'r')  #打开文件
    file_data = file.readlines() #读取所有行
    for c in range(args.n_clients):
        if not os.path.exists("./datasets/Img_Client_260/client"+str(c)):
            os.mkdir("./datasets/Img_Client_260/client"+str(c))
        for row in file_data:
            row=row.strip()   
            if not os.path.exists("./datasets/Img_Client_260/client"+str(c)+"/"+row):
                os.mkdir("./datasets/Img_Client_260/client"+str(c)+"/"+row)
    classes=[]
    for row in file_data:
        row=row.strip() 
        classes.append(row)
    dataset = ImageFolder(root='./datasets/Img')
    totalIndex=np.array(range(len(dataset.targets)))
    np.random.shuffle(totalIndex)
    samplePerUser = len(totalIndex) // args.n_clients
    totalIndex=torch.from_numpy(totalIndex)
    for c in range(args.n_clients):
        clientClass={}
        for className in classes:
            clientClass[className]=0
        if c==args.n_clients-1:
            c_index=totalIndex[c*samplePerUser:]
        else:
            c_index=totalIndex[c*samplePerUser:(c+1)*samplePerUser]
        for i in range(len(c_index)):
            sourcePath=dataset.imgs[c_index[i]][0]
            path=sourcePath.split('/')
            newname="./datasets/Img_Client_260/client"+str(c)+"/"+str(path[3])+"/"+str(path[4].lower())
            clientClass[path[3]]+=1
            shutil.copyfile(sourcePath, newname)  
        for className in classes:
            if clientClass[className]==0:
                print("user ",c," ",className) 
                os.rmdir("./datasets/Img_Client_260/client"+str(c)+"/"+className)


def createNOIIDTrainAndTestDataset():
    file = open("./datasets/ImageNet-10.txt",'r')  #打开文件
    file_data = file.readlines() #读取所有行
    for c in range(args.n_clients):
        if not os.path.exists("./datasets/Img_Client_noiid200/client"+str(c)):
            os.mkdir("./datasets/Img_Client_noiid200/client"+str(c))
    totalIndex=np.array(range(args.n_clients))
    np.random.shuffle(totalIndex)
    halfClients=args.n_clients//2
    count=0
    clientClass={}
    for c in range(args.n_clients):
        clientClass[c]=0
    for row in file_data:
        row=row.strip() 
        path="./datasets/Img/"+row
        nameList=os.listdir(path)
        if count % 2==0:
            clients=np.array(totalIndex[0:halfClients])
        else:
            clients=np.array(totalIndex[halfClients:])
            np.random.shuffle(totalIndex)
        clientsNum=len(clients)
        samplePerUser=len(nameList)//clientsNum
        print("samplePerUser:",samplePerUser)
        for i in range(len(clients)):
            c=clients[i]
            if not os.path.exists("./datasets/Img_Client_noiid200/client"+str(c)+"/"+row):
                os.mkdir("./datasets/Img_Client_noiid200/client"+str(c)+"/"+row)
                clientClass[c]+=1
            if i==len(clients)-1:
                c_index=nameList[i*samplePerUser:]
            else:
                c_index=nameList[i*samplePerUser:(i+1)*samplePerUser]
            for n in range(len(c_index)):
                sourcePath="./datasets/Img/"+row+"/"+c_index[n]
                newname="./datasets/Img_Client_noiid200/client"+str(c)+"/"+row+"/"+c_index[n]
                shutil.copyfile(sourcePath, newname)  
        count+=1




class Aggregator:
    def __init__(self, device):
        res = resnet.get_resnet(args.resnet)
        model = network.Network(res, args.feature_dim, class_num)
        # model = ModuleValidator.fix(model)
        # checkpoint = torch.load('save/CIFAR-100/checkpoint_1000.tar',map_location=torch.device('cpu'))['net']
        # for name in model.state_dict():
        #     if "cluster_projector.2" not in name:
        #         model.state_dict()[name].data = checkpoint[name].data
        model.load_state_dict(torch.load('save/cifar-original/checkpoint_1000.tar',map_location=torch.device('cpu'))['net'])
        # model.load_state_dict(torch.load('save/CIFAR-10/checkpoint_1030.tar',map_location=torch.device('cpu'))['net'])
        self.model = model.to(device)
        self.device=device
        self.count = 0
        self.modelUpdate={}

    def update(self):
        print("count",self.count )
        # for name in self.modelUpdate:
        #     noise = torch.normal(
        #         mean=0,
        #         std=sigma * args.clip_bound,
        #         size=self.modelUpdate[name].size(),
        #         device=self.modelUpdate[name].device,
        #         dtype=self.modelUpdate[name].dtype,
        #     )
        #     self.modelUpdate[name] = self.modelUpdate[name] + noise
        if self.count == 0:
            return
        for name, param in self.model.named_parameters ():
            if name in self.modelUpdate:
                noise = torch.normal(
                    mean=0,
                    std=sigma * args.clip_bound,
                    size=self.modelUpdate[name].size(),
                    device=self.modelUpdate[name].device,
                    dtype=self.modelUpdate[name].dtype,
                )
                param.data = param.data + torch.div(self.modelUpdate[name], self.count) + torch.div(noise, args.n_clients)
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
    for name, param in net.named_parameters ():
        if param.requires_grad:
            gradDict[name] = (param.data - netcopy[name].data).clone()


def setParaFromAgg(model, agg):
    for name, param in model.named_parameters ():
        param.data = agg[name].data.clone()


def processGradByDP(gradDict):
    flag=0
    for name in gradDict:
        if flag==0:
            g = gradDict[name].view(-1).clone()
            flag=1
        else:
            g = torch.cat([g, gradDict[name].view(-1).clone()], dim=0)
    norm = torch.norm(g, p=2)
    print("Norm:", norm)
    clip_bound = args.clip_bound
    sigma = get_noise_multiplier(
            target_epsilon = args.epsilon,
            target_delta = 1e-3,
            sample_rate = 1,
            epochs = args.epochs,
        )
    print("sigma:",sigma,sigma*clip_bound)
    for name in gradDict:
        gradDict[name] = torch.div(gradDict[name], max(1, torch.div(norm, args.clip_bound)))
        noise = torch.normal(
                mean=0,
                std=sigma * clip_bound,
                size=gradDict[name].size(),
                device=gradDict[name].device,
                dtype=gradDict[name].dtype,
            )
        gradDict[name] = gradDict[name] + noise


def clipGrad(gradDict):
    flag=0
    for name in gradDict:
        if flag==0:
            g = gradDict[name].view(-1).clone()
            flag=1
        else:
            g = torch.cat([g, gradDict[name].view(-1).clone()], dim=0)
    norm = torch.norm(g, p=2)
    print("Norm:", norm)
    for name in gradDict:
        gradDict[name] = torch.div(gradDict[name], max(1, torch.div(norm, args.clip_bound)))


def train(agg, round, datasets):
    n_acc_steps=args.batch_size//args.mini_bs
    agg.model.train()
    train_loss = 0
    setParaFromAgg(modelcopy, agg.model.state_dict())
    totalIndex=np.array(range(args.n_clients))
    np.random.shuffle(totalIndex)
    sampleClients=int(args.n_clients*args.sample_ratio)
    for s in range(args.n_clients-5, args.n_clients-2):
        model.train()
        # c=totalIndex[s]
        c=s
        dataset = ImageFolder(root='./datasets/Img_Client_200/client'+str(c),\
                              transform=transform.Transforms(size=args.image_size, blur=True),)
        dataLoader = DataLoader(
            dataset,
            batch_size=args.mini_bs,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        lossPerUser=0
        setParaFromAgg(model, agg.model.state_dict())
        res_params2 = [param for name, param in model.named_parameters() if 'resnet' in name]
        pro_params2 = [param for name, param in model.named_parameters() if 'resnet' not in name]
        optimizer = torch.optim.Adam([
            {'params': res_params2, 'lr': args.resnet_lr},
            {'params': pro_params2, 'lr': args.project_lr}
        ])
        optimizer.zero_grad()
        gradDict={}
        for i in range(args.local_epoch):
            for batch_idx, ((x_i,x_j),_) in enumerate(dataLoader):
                x_i = x_i.to(device)
                x_j = x_j.to(device)
                z_i, z_j, c_i, c_j = model(x_i, x_j)
                loss_instance = criterion_instance(z_i, z_j)
                loss_cluster = criterion_cluster(c_i, c_j)
                loss = loss_instance + loss_cluster
                loss.backward()
                # if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(dataLoader)):
                if (batch_idx + 1) % n_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                lossPerUser += loss.item()
        print("test user",c)
        test(round, model)
        getModelUpdate(gradDict, modelcopy.state_dict (), model) #model - modelcopy
        clipGrad(gradDict)
        train_loss+=lossPerUser
        agg.collect(gradDict) #共享内存
        print('Round: ', round, "User:", c, 'Train Loss: %.3f' % (lossPerUser/(len(dataLoader)*args.local_epoch)))
    
    # cosineLR.step()
    keepParams=copy.deepcopy(agg.model.state_dict())
    agg.model.load_state_dict(copy.deepcopy(model.state_dict()))
    for name, param in agg.model.named_parameters():
        if param.requires_grad:
            param.data = keepParams[name]
    agg.update()
    print('*********Round: ', round, 'Train Loss: %.3f' % (train_loss/(args.n_clients*len(dataLoader)*args.local_epoch)))


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
        # if step % 20 == 0:
        #     print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    # print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def testNoiid(round, testmodel):
    nmiList, ariList, fList, accList= [],  [],  [], []
    for c in range(args.n_clients):
        dataset = ImageFolder(root='./datasets/Img_Client_noiid200/client'+str(c),\
                                transform=transform.Transforms(size=args.image_size).test_transform,)
        data_loader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
        )
        # print("### Creating features from model ###")
        X, Y = inference(data_loader, testmodel, device)
        nmi, ari, f, acc = evaluation.evaluateNoiid(Y, X)
        nmiList.append(nmi)
        ariList.append(ari)
        fList.append(f)
        accList.append(acc)
        print('Rround '+ str(round)+ ' User '+str(c) + ' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    print('Average NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'\
          .format(sum(nmiList)/len(nmiList), sum(ariList)/len(ariList), sum(fList)/len(fList), sum(accList)/len(accList)))

def testiid(round, testmodel):
    nmiList, ariList, fList, accList= [],  [],  [], []
    for c in tqdm(range(200)):
        dataset = ImageFolder(root='./datasets/Img_Client_200/client'+str(c),\
                                transform=transform.Transforms(size=args.image_size).test_transform,)
        data_loader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
        )
        # print("### Creating features from model ###")
        X, Y = inference(data_loader, testmodel, device)
        nmi, ari, f, acc = evaluation.evaluateNoiid(Y, X)
        nmiList.append(nmi)
        ariList.append(ari)
        fList.append(f)
        accList.append(acc)
        # print('Rround '+ str(round)+ ' User '+str(c) + ' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
    print('Average NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'\
          .format(sum(nmiList)/len(nmiList), sum(ariList)/len(ariList), sum(fList)/len(fList), sum(accList)/len(accList)))


def test(round, testmodel):
    dataset = torchvision.datasets.ImageFolder(
            root='datasets/Img',
            transform=transform.Transforms(size=args.image_size).test_transform,
        )
    data_loader = DataLoader(
        dataset,
        batch_size=200,
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
    config = yaml_config_hook("config/config_DP_FL_Img_fix.yaml")
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
    
    # createIIDTrainAndTestDataset()
    # createNOIIDTrainAndTestDataset()
   
    class_num = 10
    datasets=[]
    
    criterion_instance = contrastive_loss.InstanceLoss(args.mini_bs, args.instance_temperature, device).to(
        device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, device).to(device)

    agg = Aggregator(device)
    # model_fp = os.path.join("save/Img-10-FL-clientlevel", "checkpoint_{}.tar".format(6))
    # checkpoint = torch.load(model_fp, map_location=device)
    # agg.model.load_state_dict(checkpoint['net'])
    # testiid(0, agg.model)

    rescopy = resnet.get_resnet(args.resnet)
    modelcopy = network.Network(rescopy, args.feature_dim, class_num)
    # modelcopy = ModuleValidator.fix(modelcopy)
    modelcopy = modelcopy.to(device)

    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    # model = ModuleValidator.fix(model)
    model = model.to(device)
    res_params = [param for name, param in modelcopy.named_parameters() if 'resnet' in name]
    pro_params = [param for name, param in modelcopy.named_parameters() if 'resnet' not in name]

    optimizer = torch.optim.Adam([
        {'params': res_params, 'lr': args.resnet_lr},
        {'params': pro_params, 'lr': args.project_lr}
    ])
    # cosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0003)
    # print("next lr:",cosineLR.get_last_lr())

    sigma = get_noise_multiplier(
                target_epsilon = args.epsilon,
                target_delta = 1e-3,
                sample_rate = 1,
                epochs = args.epochs,
            )
    print("sigma:", sigma)
    
    finetuneParams=["bn"]
    print(finetuneParams)
    for name, param in model.named_parameters():
        print(name, param.numel(), param.size())
        param.requires_grad_(False)
        if 'resnet' in name :
            for finetuneName in finetuneParams:
                if finetuneName in name:
                    param.requires_grad_(True)
        elif "cluster_projector.2" in name or "instance_projector.2" in name:
            param.requires_grad_(True)
    

    print('Number of total parameters: ', sum([p.numel() for p in model.parameters()]))
    print('Number of trainable p  arameters: ', sum([p.numel() for p in model.parameters() if p.requires_grad]))
    
    if args.reload:
        model_fp = os.path.join("save/Img-10-FL-clientlevel", "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp, map_location=device)
        agg.model.load_state_dict(checkpoint['net'])
        model.load_state_dict(checkpoint['net'])
        args.start_epoch = checkpoint['epoch'] + 1

    # train
    for epoch in range(args.start_epoch, args.epochs):
        test(epoch, agg.model)
        train(agg, epoch, datasets)
        save_model2(args, agg.model, epoch)
            
    save_model2(args, agg.model, args.epochs)
    test(args.epochs, agg.model)

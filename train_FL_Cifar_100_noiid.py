import os
import numpy as np
import torch
import torchvision
import argparse
from sklearn.cluster import KMeans
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook
from torch.utils import data
from torch.utils.data import DataLoader
from opacus.validators import ModuleValidator
from evaluation import evaluation
from opacus.accountants.utils import get_noise_multiplier
import pickle
import PIL
import math
import copy
from collections import defaultdict

def save_model2(args, model, current_epoch):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)
    torch.save(clientsSavedParams, os.path.join(args.model_path, "clientsSavedParams_{}.tar".format(current_epoch)))
   


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
        return ([self.transform(self.samples[index]) for i in range(args.local_epoch)], self.labels[index])


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
    

def createIIDTrainAndTestDataset():
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=False, download=True, transform=transformation)
    trainset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=True, download=True, transform=transformation)
    dataset = data.ConcatDataset([trainset, testset])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=4) 
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx==0:
            Sample=inputs.clone()
            Label=targets.clone()
        else:
            Sample=torch.cat([Sample, inputs.clone()], dim=0)
            Label=torch.cat([Label, targets.clone()], dim=0)
    
    with open("./datasets/cifar100/Sample", "wb") as f:
        pickle.dump(Sample, f)  
    with open("./datasets/cifar100/Label", "wb") as f:
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


def createNoIIDTrainAndTestDataset():
    classes_per_user=args.classes_per_user
    num_users=args.n_clients
    num_classes = 20
    count_per_class = (classes_per_user * num_users) // num_classes
    class_dict = {}
    for i in range(num_classes):
        probs=np.array([1]*count_per_class)
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            max_class_counts = np.setdiff1d(max_class_counts, np.array(c))
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    
    with open("./datasets/cifar100/Sample", "rb") as f:
        Sample=pickle.load(f) 
    with open("./datasets/cifar100/Label20", "rb") as f:
        Label=pickle.load(f).cpu() 

    clientDataIndex = [[] for i in range(num_users)]
    data_class_idx = [[] for i in range(20)]
    num_samples = [0 for i in range(20)]
    for c in range(20):
        sampleIndex=np.where(np.array(Label)==c)[0]
        data_class_idx[c].extend(sampleIndex)
        num_samples[c] = len(sampleIndex)
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            clientDataIndex[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    for c in range(num_users):
        clientDataIndex[c]=torch.tensor(clientDataIndex[c])
        if c==0:
            sample_noiid=Sample[clientDataIndex[c]].clone()
            label_noiid=Label[clientDataIndex[c]].clone()
        else:
            sample_noiid=torch.cat([sample_noiid, Sample[clientDataIndex[c]].clone()], dim=0)
            label_noiid=torch.cat([label_noiid, Label[clientDataIndex[c]].clone()], dim=0)

    with open("./datasets/cifar100/Sample_noiid_class"+str(classes_per_user), "wb") as f:
        pickle.dump(sample_noiid, f)  
    with open("./datasets/cifar100/Label_noiid_class"+str(classes_per_user), "wb") as f:
        pickle.dump(label_noiid, f) 



def setParaFromChekpoint(model, checkpoint):
    for name, param in model.named_parameters ():
        if "cluster_projector2" in name and "cluster_projector2.2" not in name:
            name = name.replace('cluster_projector2', 'cluster_projector')
            param.data = checkpoint[name].to(device)


class Aggregator:
    def __init__(self, device):
        res = resnet.get_resnet(args.resnet, args.r_conv)
        model = network.Network_perCluster(res, args.feature_dim, args.num_class, args.r_proj)
        model = ModuleValidator.fix(model)
        # name="save/Cifar-10-DPFL-ResNet18-blur-lus-kl-threshold-lorabegin/checkpoint_51.tar"
        name='save/Img-10-pretrain-transform/checkpoint_532.tar'
        # name='save/Cifar10/checkpoint_1000.tar'
        checkpoint = torch.load(name,map_location=torch.device('cpu'))['net']
        print(name)
        model.load_state_dict(checkpoint, strict=False)
        self.model = model.to(device)
        # setParaFromChekpoint(model, checkpoint)
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
                # param.grad = updated + torch.div(noise, 400)
                param.grad = updated
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

def getModelUpdate(gradDict, netcopy, net, localParams):
    flag=0
    for name, param in net.named_parameters ():
        if param.requires_grad and (name not in localParams):
            gradDict[name] = (netcopy[name].data - param.data).clone()
            if flag==0:
                normSum = torch.sum(torch.pow(gradDict[name], exponent=2))
                flag=1
            else:
                normSum += torch.sum(torch.pow(gradDict[name], exponent=2))
    return normSum


def getModelUpdateNorm(netcopy, net, localParams):
    flag=0
    for name, param in net.named_parameters ():
        if param.requires_grad and (name not in localParams):
            if flag==0:
                normSum = torch.sum(torch.pow((netcopy[name] - param.data), exponent=2))
                flag=1
            else:
                normSum += torch.sum(torch.pow((netcopy[name] - param.data), exponent=2))
    return normSum


def setParaFromAgg(model, agg):
    for name, param in model.named_parameters ():
        param.data = torch.tensor(agg[name].clone())


def clipGrad(gradDict, normSum): 
    norm=torch.sqrt(normSum)
    print("Norm: ",norm)
    scale=torch.div(norm, args.clip_bound)
    for name in gradDict:
        gradDict[name] = torch.div(gradDict[name], max(1, scale.item()))


def calculateMask(gradDict, modelUpdateDict):
    maskDict={}
    for name in gradDict:
        if "bn" in name or "bias" in name or "downsample.1" in name:
            sparsity = args.bn_sparsity
        else:
            sparsity = args.linear_sparsity
        # res = torch.abs(gradDict[name]*modelUpdateDict[name])
        res = torch.abs(modelUpdateDict[name])
        flatten = torch.flatten(res)
        saved=int(len(flatten)*sparsity)
        _, indices = torch.topk(flatten, k=saved)        
        mask = torch.zeros_like(flatten)
        mask[indices] = 1
        maskDict[name] = mask.view(gradDict[name].size())
    return maskDict


def saveGrad(model, gradDict):
    for name, param in model.named_parameters ():
        if param.requires_grad:
            gradDict[name] = torch.tensor(param.grad.data.clone())


def maskModelUpdate(modelUpdateDict, maskDict):
    for name in modelUpdateDict:
        if name in maskDict:
            modelUpdateDict[name] = torch.mul(modelUpdateDict[name], maskDict[name])


def getTargetDis(cluster):
    weight = (cluster ** 2) / torch.sum(cluster, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def exceedThreshold(dis):
    dis_copy = torch.tensor(dis).cpu()
    top2, _ = torch.topk(dis_copy, k=2, dim=1, largest=True)
    dif = np.array(top2[:, 0] - top2[:, 1])
    index = torch.tensor(np.where(dif > args.kl_threshold)[0]).to(device)
    return index


def setParaFromSavedParams(model, savedParams):
    for name, param in model.named_parameters ():
        if name in savedParams.keys():
            param.data = savedParams[name].to(device)


def train(agg, round):
    agg.model.train()
    train_loss = 0
    normlist=[]
    totalIndex=np.array(range(args.n_clients))
    np.random.shuffle(totalIndex)
    sampleClients=int(args.n_clients*args.sample_ratio)
    clients_sampled = list(totalIndex[0:sampleClients])
    loss_function = torch.nn.KLDivLoss(size_average=False)
    for batch_idx, (x_list, y_label) in enumerate(dataLoader):
        if clients_sampled.count(batch_idx) == 0:
            continue
        c=batch_idx
        model.train()
        setParaFromAgg(model, agg.model.state_dict())
        setParaFromSavedParams(model, clientsSavedParams[c])
        res_params2_bn = [param for name, param in model.named_parameters() if 'resnet' in name and 'lora' not in name]
        res_params2_conv = [param for name, param in model.named_parameters() if 'resnet' in name and 'lora' in name]
        instance_projector_params2 = [param for name, param in model.named_parameters() if 'instance_projector' in name]
        cluster_projector_params2 = [param for name, param in model.named_parameters() if 'cluster_projector' in name]
        trans_params2 = [param for name, param in model.named_parameters() if 'trans' in name]
        optimizer=torch.optim.SGD([
            {'params': res_params2_bn, 'lr': args.resnet_lr},
            {'params': res_params2_conv, 'lr': args.downsample_lr},
            {'params': instance_projector_params2, 'lr': args.instance_project_lr},
            {'params': cluster_projector_params2, 'lr': args.cluster_project_lr},
            {'params': trans_params2, 'lr': args.trans_lr},
        ], momentum=0.9) 
       
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
        test_bubian={}
        for i in range(len(contrasive_pair)):
            pair = contrasive_pair[pair_index[i]]
            batch_num = int(args.batch_size/args.mini_bs)
            batch_index=np.array(range(args.batch_size))
            np.random.shuffle(batch_index)
            for j in range(batch_num):
                true_epoch += 1 
                batch_records = batch_index[j*args.mini_bs:(j+1)*args.mini_bs]
                x_i = x_list[pair[0]][batch_records].to(device)
                x_j = x_list[pair[1]][batch_records].to(device)       
                z_i, z_j, c_i, c_j = model(x_i, x_j)
                loss_instance = criterion_instance(z_i, z_j)
                loss_cluster = criterion_cluster(c_i, c_j)

                # c_i_target = getTargetDis(c_i)
                # index_i = exceedThreshold(c_i_target)
                # c_i_target=c_i_target[index_i]
                # c_i = c_i[index_i]

                # c_j_target = getTargetDis(c_j)
                # index_j = exceedThreshold(c_j_target)
                # c_j_target = c_j_target[index_j]
                # c_j = c_j[index_j]
                # loss_KL=0
                # if c_i.shape[0]>0:
                #     loss_KL+=loss_function(c_i.log(), c_i_target) / c_i.shape[0]
                # if c_j.shape[0]>0:
                #     loss_KL+=loss_function(c_j.log(), c_j_target) / c_j.shape[0]

                # with torch.no_grad():
                #     z_i_g = agg.model.forward_instance(x_i)
                #     z_j_g = agg.model.forward_instance(x_j)
  
                # loss_zhengze = (torch.sum(torch.pow((z_i - z_i_g), exponent=2))/len(z_i)+\
                #   torch.sum(torch.pow((z_j - z_j_g), exponent=2))/len(z_i))/2
                # loss_blur = getModelUpdateNorm(agg.model.state_dict (), model, clientsSavedParams[c])
                # print(loss_blur)
                # loss = loss_instance + loss_cluster + args.miu*loss_zhengze
                # loss = loss_instance + loss_cluster + args.miu*loss_blur
                loss = loss_instance + loss_cluster
                loss.backward()
                lossPerUser += loss.item()
                optimizer.step()
                optimizer.zero_grad()
        
        normSum = getModelUpdate(modelUpdateDict, agg.model.state_dict (), model, clientsSavedParams[c]) # modelcopy - model 
        normlist.append(torch.sqrt(normSum).item())
        clientsSavedParams[c] = saveClientsParams(model, localParams) 
        clipGrad(modelUpdateDict, normSum)
        train_loss+=lossPerUser
        agg.collect(modelUpdateDict) #共享内存
        print('Round: ', round, "User:", c, 'Train Loss: %.3f' % (lossPerUser/true_epoch))
      
        # if batch_idx>= 19:
        #     break
       
    if args.clip_bound > 1.9:
        args.clip_bound=args.clip_bound - 0.4
    else:
        args.clip_bound = sum(normlist)/len(normlist) - 0.2
        if args.clip_bound > 2:
            args.clip_bound = 1.6
        elif args.clip_bound > 1.5:
            args.clip_bound = 1.4

    print("clip_bound", args.clip_bound)
    
    agg.model.train()
    agg.update()
    print('*********Round: ', round, 'Train Loss: %.3f' % (train_loss/((batch_idx+1)*true_epoch)))
    


def inference(loader, testmodel, device):
    testmodel.eval()
    feature_vector = []
    labels_vector = []
    kmeans_vector = []
    for step, (x, y) in enumerate(loader):
        setParaFromSavedParams(testmodel, clientsSavedParams[step])
        x = x.to(device)
        with torch.no_grad():
            for i in range(int(args.batch_size/args.mini_bs)):
                c = testmodel.forward_cluster(x[i*args.mini_bs:(i+1)*args.mini_bs])
                c = c.detach()
                feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 100 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    kmeans_vector = np.array(kmeans_vector)
    return feature_vector, labels_vector


def calClusterCenters(feature_vector, predictlabel_vector, num_calss):
    center_list = []
    center_label = []
    for i in range(num_calss):
        cluster_i = feature_vector[np.where(np.array(predictlabel_vector)==i)[0]]
        if len(cluster_i)>0:
            center = np.sum(cluster_i, axis=0)/len(cluster_i)
            center_list.append(np.array(center))
            center_label.append(i)
    return np.array(center_list), center_label


def inference_kfed(loader, testmodel, device):
    testmodel.eval()
    client_local_predict={}
    labels_vector = []
    collect_local_centers=[]
    client_index_collect ={}
    client_index_now = 0
    client_cluster_label = {}
    for step, (x, y) in enumerate(loader):
        feature_vector = []
        predictlabel_vector = []
        setParaFromSavedParams(testmodel, clientsSavedParams[step])
        x = x.to(device)
        with torch.no_grad():
            for i in range(int(args.batch_size/args.mini_bs)):
                c = testmodel.forward_cluster(x[i*args.mini_bs:(i+1)*args.mini_bs])
                c = c.detach()
                predictlabel_vector.extend(c.cpu().detach().numpy())

                feature = testmodel.forward_instance(x[i*args.mini_bs:(i+1)*args.mini_bs])
                feature = feature.detach()
                feature_vector.extend(feature.cpu().detach().numpy())
            feature_vector = np.array(feature_vector)
            predictlabel_vector = np.array(predictlabel_vector)
            client_local_predict[step] = predictlabel_vector
            local_centers, cluster_label = calClusterCenters(feature_vector, predictlabel_vector, args.num_class)
            if True in np.isnan(local_centers):
                print("nan", local_centers)
            client_index_collect[step] = [client_index_now, client_index_now + len(local_centers)]
            client_index_now = client_index_now + len(local_centers)
            collect_local_centers.append(local_centers)
            client_cluster_label[step] = cluster_label
        labels_vector.extend(y.numpy())
    print("########### begin kmeans ##########")
    kmeans = KMeans(n_clusters=20)
    collect_local_centers = np.concatenate(collect_local_centers, axis=0)
    kmeans.fit(collect_local_centers)
    global_label_pred = kmeans.labels_
    global_results=[]
    for c in range(args.n_clients):
        client_global_label = global_label_pred[client_index_collect[c][0]: client_index_collect[c][1]]
        client_local_label = client_local_predict[c]
        preCluster = np.zeros(len(client_local_label), dtype=int)
        if len(client_cluster_label[c])!=client_index_collect[c][1]-client_index_collect[c][0]:
            print("false!!!")
        for i in range(len(client_cluster_label[c])):
            index=np.where(client_local_label==client_cluster_label[c][i])[0]
            preCluster[index] = client_global_label[i]
        preCluster=list(preCluster)
        global_results.extend(preCluster)

    labels_vector = np.array(labels_vector)
    global_results = np.array(global_results)
    return global_results, labels_vector


def testiid(testmodel):
    nmiList, ariList, fList, accList= [],  [],  [], []
    mean=[0.5071, 0.4867, 0.4408]
    std=[0.2675, 0.2565, 0.2761]
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
    dataset = Cifar100DatasetTest(Sample, Label, transformation)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    print("### Creating features from model ###")
    X, Y = inference(data_loader, testmodel, device)
    nmi, ari, f, acc = evaluation.evaluate(Y, X)
    print('Global '+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))

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


def testNoiid(testmodel):
    mean=[0.5071, 0.4867, 0.4408]
    std=[0.2675, 0.2565, 0.2761]
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
    dataset = Cifar100DatasetTest(Sample, Label, transformation)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    print("### Creating features from model ###")
    X, Y = inference_kfed(data_loader, testmodel, device)
    nmi, ari, f, acc = evaluation.evaluate(Y, X)
    print('Global '+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))



def decide_requires_grad(model):
    finetuneParams=["bn", "bias", "lora", "downsample.1"]
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if 'resnet' in name :
            for finetuneName in finetuneParams:
                if finetuneName in name:
                    param.requires_grad_(True)
        # elif "cluster_projector" in name or "instance_projector" in name:
        elif "cluster_projector" in name or "instance_projector" in name or "trans" in name:
            param.requires_grad_(True)


def saveClientsParams(model, keepLocal):
    savedParams = {}
    for name, param in model.named_parameters ():
        for localParam in keepLocal:
            if localParam in name:
                savedParams[name] = torch.tensor(param.data.clone()).cpu()
    return savedParams


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    device= torch.device("cuda:3")
    print(device)
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config_DP_FL_Cifar100_noiid.yaml")
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

    cpu_num = 5 # 这里设置成你想运行的CPU个数
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
    os.environ["MKL_NUM_THREADS"] = str(cpu_num) # noqa
    torch.set_num_threads(cpu_num )
    

    # createIIDTrainAndTestDataset()
    # createNoIIDTrainAndTestDataset()

    # prepare data
    datasets=[]

    with open("./datasets/cifar100/Sample_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients), "rb") as f:
        Sample = pickle.load(f)
    with open("./datasets/cifar100/Label_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients), "rb") as f:
        Label = pickle.load(f)
    print(len(Sample),len(Label))
    # with open("./datasets/cifar100/Sample", "rb") as f:
    #     Sample = pickle.load(f)
    # with open("./datasets/cifar100/Label20", "rb") as f:
    #     Label = pickle.load(f).cpu()
    
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

    dataset = Cifar100Dataset(Sample, Label, transformation)
    dataLoader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=True,
        )
    

    # clientDataIndex = createclientDataIndex("./datasets/cifar10/Sample_noiid_class"+str(args.classes_per_user))
    clientDataIndex = createclientDataIndex("./datasets/cifar100/Sample_noiid_class"+str(args.classes_per_user)+"_"+str(args.n_clients))
    print(torch.unique(Label[clientDataIndex[0]]))
    # # optimizer / loss
    
    criterion_instance = contrastive_loss.InstanceLoss(args.mini_bs, args.instance_temperature, device).to(device)
    criterion_cluster = contrastive_loss.ClusterLoss(args.num_class, args.cluster_temperature, device).to(device)

    agg = Aggregator(device)
    # loadpath="save/Cifar-100-DPFL-ResNet18-noiid-classes_per_user8-num_class8-prox"
    # model_fp = os.path.join(loadpath, "checkpoint_{}.tar".format(39))
    # checkpoint = torch.load(model_fp, map_location=device)
    # agg.model.load_state_dict(checkpoint['net'], strict=False)
    # print(loadpath)

    res = resnet.get_resnet(args.resnet, args.r_conv)
    model = network.Network_perCluster(res, args.feature_dim, args.num_class, args.r_proj)
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

    localParams=[]
    clientsSavedParams={}
    for c in range(args.n_clients):
        clientsSavedParams[c] = saveClientsParams(agg.model, localParams)
   
    # loadpath="save/Cifar-100-DPFL-ResNet18-noiid-classes_per_user8-bn"
    # model_fp = os.path.join(loadpath, "clientsSavedParams_{}.tar".format(21))
    # clientsSavedParams = torch.load(model_fp, map_location=device)

    # train
    for epoch in range(args.start_epoch, args.epochs):
        testNoiid(agg.model)
        # if epoch>args.start_epoch:
        #     testiid(agg.model)
        train(agg, epoch)
        save_model2(args, agg.model, epoch)
            
    save_model2(args, agg.model, args.epochs)
    testNoiid(agg.model)

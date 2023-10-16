import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from opacus.validators import ModuleValidator
from evaluation import evaluation
import PIL
import pickle
import torchvision.models as models
from opacus.accountants.utils import get_noise_multiplier

    
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
        return ((self.transform(self.samples[index]), self.transform(self.samples[index])), self.labels[index])


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

def train():
    model.train()
    loss_epoch = 0
    n_acc_steps=args.batch_size//args.mini_bs
    optimizer.zero_grad()
    for step, ((x_i, x_j), _) in enumerate(data_loader):
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        # if ((step + 1) % n_acc_steps == 0) or ((step + 1) == len(data_loader)):
        if (step + 1) % n_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch


def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        # if step % 20 == 0:
        #     print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


# def test(round):
#     train_dataset = torchvision.datasets.CIFAR10(
#             root=args.dataset_dir,
#             train=True,
#             download=True,
#             transform=transform.Transforms(size=args.image_size).test_transform,
#         )
#     test_dataset = torchvision.datasets.CIFAR10(
#             root=args.dataset_dir,
#             train=False,
#             download=True,
#             transform=transform.Transforms(size=args.image_size).test_transform,
#         )
#     dataset = data.ConcatDataset([train_dataset, test_dataset])
#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=200,
#         shuffle=False,
#         drop_last=False,
#         num_workers=args.workers,
#     )
#     print("### Creating features from model ###")
#     X, Y = inference(data_loader, model, device)
#     nmi, ari, f, acc = evaluation.evaluate(Y, X)
#     print('Rround '+ str(round)+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))

def testImg(round):
    dataset = torchvision.datasets.ImageFolder(
            root='datasets/Img',
            transform=transform.Transforms(size=args.image_size).test_transform
        )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=300,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    print("### Creating features from model ###")
    X, Y = inference(data_loader, model, device)
    nmi, ari, f, acc = evaluation.evaluate(Y, X)
    print('Rround '+ str(round)+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))


def testCifar(round):
    with open("./datasets/cifar10/Sample", "rb") as f:
        Sample = pickle.load(f)
    with open("./datasets/cifar10/Label", "rb") as f:
        Label = pickle.load(f)

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
    X, Y = inference(data_loader, model, device)
    nmi, ari, f, acc = evaluation.evaluate(Y, X)
    print('Rround '+ str(round)+' NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    device= torch.device("cuda:1")
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config_Cifar.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    print(device)
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

    # prepare data
    if args.dataset == "CIFAR-10":
        # train_dataset = torchvision.datasets.CIFAR10(
        #     root="./datasets",
        #     download=False,
        #     train=True,
        #     transform=transform.Transforms(size=args.image_size, s=0.5),
        # )
        # test_dataset = torchvision.datasets.CIFAR10(
        #     root="./datasets",
        #     download=False,
        #     train=False,
        #     transform=transform.Transforms(size=args.image_size, s=0.5),
        # )
        # dataset = data.ConcatDataset([train_dataset, test_dataset])
        # class_num = 10
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
        dataset = Cifar10Dataset(Sample, Label, transformation)
        class_num=10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    # elif args.dataset == "ImageNet-10":
    #     mean=[0.485, 0.456, 0.406]
    #     std=[0.229, 0.224, 0.225]
    #     dataset = torchvision.datasets.ImageFolder(
    #         root='datasets/Img',
    #         transform=transform.Transforms(size=args.image_size, mean=mean, std=std, s=0.5, blur=0.5),
    #     )
    #     class_num = 10
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/Img',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-dogs',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/tiny-imagenet-200/train',
            transform=transform.Transforms(s=0.5, size=args.image_size),
        )
        class_num = 200
    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.mini_bs,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    sigma = get_noise_multiplier(
                target_epsilon = args.epsilon,
                target_delta = 1e-3,
                sample_rate = 100/600,
                epochs = args.epochs,
            )
    print("sigma:", sigma)

    # initialize model
    # resnet18 = models.resnet18(pretrained=True)
    res = resnet.get_resnet(args.resnet, args.r_conv)
    # res.load_state_dict(resnet18.state_dict(), False)
    model = network.Network(res, args.feature_dim, class_num, args.r_proj, sigma)
    model = ModuleValidator.fix(model)
    name='save/Img-10-pretrain-transform/checkpoint_532.tar'
    checkpoint = torch.load(name,map_location=torch.device('cpu'))['net']
    model.load_state_dict(checkpoint, strict=False)
    model=model.to(device)

    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join("save/Img-10-pretrain-transform", "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp, map_location=device)
        model.load_state_dict(checkpoint['net'], False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    criterion_instance = contrastive_loss.InstanceLoss(args.mini_bs, args.instance_temperature, device).to(device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, device).to(device)
    # train
    for epoch in range(args.start_epoch, args.epochs):
        loss_epoch = train()
        testCifar(epoch)
        # if epoch % 4 == 0:
        #     save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
        # if epoch % 4 == 0:
        #     # testImg(epoch)
        #     testCifar(epoch)
    save_model(args, model, optimizer, args.epochs)

import torch.nn as nn
import torch
from torch.nn.functional import normalize
import loralib as lora


class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num, r_proj):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.r_proj = r_proj
        # self.instance_projector = nn.Sequential(
        #     lora.Linear(self.resnet.rep_dim, self.resnet.rep_dim, r=self.r_proj),
        #     nn.ReLU(),
        #     nn.Linear(self.resnet.rep_dim, self.feature_dim),
        # )
        # self.cluster_projector = nn.Sequential(
        #     lora.Linear(self.resnet.rep_dim, self.resnet.rep_dim, r=self.r_proj),
        #     nn.ReLU(),
        #     nn.Linear(self.resnet.rep_dim, self.cluster_num),
        #     nn.Softmax(dim=1)
        # )

        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )


    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)
        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
    

class Network_cluster(nn.Module):
    def __init__(self, resnet, feature_dim, class_num, r_proj):
        super(Network_cluster, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.r_proj = r_proj

        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )

        self.cluster_projector = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )


    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        instance_i=self.instance_projector(h_i)
        instance_j=self.instance_projector(h_j)
        z_i = normalize(instance_i, dim=1)
        z_j = normalize(instance_j, dim=1)

        c_i = self.cluster_projector(instance_i)
        c_j = self.cluster_projector(instance_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        instance=self.instance_projector(h)
        c = self.cluster_projector(instance)
        c = torch.argmax(c, dim=1)
        return c
    

class Network_Classifer(nn.Module):
    def __init__(self, resnet, feature_dim, class_num, rep_dim):
        super(Network_Classifer, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
    

class Network_Adapter(nn.Module):
    def __init__(self, resnet, feature_dim, class_num, rep_dim):
        super(Network_Adapter, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.adapter_instance_projector = nn.Sequential(
            nn.Linear(rep_dim, 96),
            nn.ReLU(),
            nn.Linear(96, rep_dim),
        )
        self.instance_projector = nn.Sequential(
            nn.Linear(rep_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.feature_dim),
        )
        self.adapter_cluster_projector = nn.Sequential(
            nn.Linear(rep_dim, 96),
            nn.ReLU(),
            nn.Linear(96, rep_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(rep_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)
        z_i = normalize(self.instance_projector(self.adapter_instance_projector(h_i)+h_i), dim=1)
        z_j = normalize(self.instance_projector(self.adapter_instance_projector(h_j)+h_j), dim=1)

        c_i = self.cluster_projector(self.adapter_cluster_projector(h_i)+h_i)
        c_j = self.cluster_projector(self.adapter_cluster_projector(h_j)+h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(self.adapter_cluster_projector(h)+h)
        c = torch.argmax(c, dim=1)
        return c

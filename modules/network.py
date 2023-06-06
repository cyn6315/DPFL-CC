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
        z_i = self.instance_projector(h_i)
        z_j = self.instance_projector(h_j)
        z_i = normalize(z_i, dim=1)
        z_j = normalize(z_j, dim=1)
        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
    
    def forward_instance(self, x):
        h = self.resnet(x)
        z = normalize(self.instance_projector(h), dim=1)
        return z
    
    def forward_cluster_rep(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
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
    
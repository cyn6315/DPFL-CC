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

        # self.trans_alpha = nn.Parameter(torch.ones(3, 224, 224))
        # self.trans_beta = nn.Parameter(torch.zeros(3, 224, 224))

        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector2 = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )
        

    def forward(self, x_i, x_j):
        # x_i=torch.einsum('...ijk, ...ijk->...ijk', x_i, self.trans_alpha) + self.trans_beta
        # x_j=torch.einsum('...ijk, ...ijk->...ijk', x_j, self.trans_alpha) + self.trans_beta
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)
        z_i = self.instance_projector(h_i)
        z_j = self.instance_projector(h_j)
        z_i = normalize(z_i, dim=1)
        z_j = normalize(z_j, dim=1)
        c_i = self.cluster_projector2(h_i)
        c_j = self.cluster_projector2(h_j)
        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector2(h)
        c = torch.argmax(c, dim=1)
        return c
    
    def forward_instance(self, x):
        h =self.resnet(x)
        # z = self.instance_projector(h)
        # z = normalize(z, dim=1)
        return h
    
    def forward_cluster_rep(self, x):
        h = self.resnet(x)
        c = self.cluster_projector2(h)
        return c
    
class Network_DEC(nn.Module):
    def __init__(self, resnet, feature_dim, class_num, r_proj):
        super(Network_DEC, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.r_proj = r_proj
        self.alpha = 1.0

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

    def getSoftLabel(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_projector[2].weight)**2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t() #soft assignment using t-distribution
        return t_dist
        
    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)
        z_i = normalize(self.cluster_projector[0](h_i), dim=1)
        z_j = normalize(self.cluster_projector[0](h_j), dim=1)
        c_i = self.getSoftLabel(z_i)
        c_j = self.getSoftLabel(z_j)
       
        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        z = self.cluster_projector[0](h)
        c = self.getSoftLabel(z)
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
    
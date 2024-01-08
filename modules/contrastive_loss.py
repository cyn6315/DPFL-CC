import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


class ClusterKLLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(ClusterKLLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.KL_loss = torch.nn.KLDivLoss(reduction='none')
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self.mask_correlated_clusters(batch_size)

    def mask_correlated_clusters(self, batch_size):
        mask = torch.ones((batch_size, batch_size))
        mask = mask.fill_diagonal_(0)
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        labels = torch.zeros(self.batch_size).to(self.device).long()
        for index in range(c_i.size(0)):
            p = c_i[index]        
            p = p.repeat(c_i.size(0), 1)
            klLoss = self.KL_loss(F.log_softmax(p, dim=-1), F.softmax(c_j, dim=-1)).sum(dim=1).reshape(1, -1)
            klLoss=-klLoss/self.temperature
            if index == 0:
                logits =torch.tensor(klLoss)
            else:
                logits = torch.cat((logits, klLoss), dim=0)
            labels[index] = torch.tensor(index, device=self.device)
            
        labels = labels.long()
        loss = self.criterion(logits, labels)
        loss /= self.batch_size

        return loss
    
    
def js_loss(x1, x2, xa, t=0.1, t2=0.1):
    """Relational loss objective function"""
    pred_sim1 = torch.mm(x1, xa.t())
    inputs1 = F.log_softmax(pred_sim1 / t, dim=1)
    pred_sim2 = torch.mm(x2, xa.t())
    inputs2 = F.log_softmax(pred_sim2 / t, dim=1)
    target_js = (F.softmax(pred_sim1 / t2, dim=1) + F.softmax(pred_sim2 / t2, dim=1)) / 2
    js_loss1 = F.kl_div(inputs1, target_js, reduction="batchmean")
    js_loss2 = F.kl_div(inputs2, target_js, reduction="batchmean")
    return (js_loss1 + js_loss2) / 2.0


def cluster_loss(c1, c2):
    inputs1 = torch.log(c1)
    inputs2 = torch.log(c2)
    target_js = (c1 + c2) / 2
    js_loss1 = F.kl_div(inputs1, target_js, reduction="batchmean")
    js_loss2 = F.kl_div(inputs2, target_js, reduction="batchmean")
    return (js_loss1 + js_loss2) / 2.0
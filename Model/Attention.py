import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.network import Classifier_1fc, DimReduction

class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred

class First_stream_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=5, droprate=0, device = 'cpu'):
        super(First_stream_Classifier, self).__init__()
        self.dimReduction = DimReduction(L*2, L, numLayer_Res = 0).to(device)    #第一流提取特征
        self.attention = Attention_Gated(L, D, K).to(device)
        self.classifier = Classifier_1fc(L, num_cls, droprate).to(device)

    def forward(self, x): ## x: N x L
        slide_sub_feat = []
        for subFeat_tensor in x:
            subFeat_tensor = subFeat_tensor.to('cuda')
            tmidFeat = self.dimReduction(subFeat_tensor)
#最大池化
#             tattFeat_tensor, _ = torch.max(tmidFeat,dim=0)
#             tattFeat_tensor = tattFeat_tensor.unsqueeze(0)  
#注意力池化
            tAA = self.attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
#平均池化
#             tattFeats = torch.mean(tmidFeat, dim=0, keepdim=True)
#             tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
       
    

            slide_sub_feat.append(tattFeat_tensor)

        slide_sub_feat = torch.cat(slide_sub_feat, dim=0)  ### numGroup x fs
        AA = self.attention(slide_sub_feat).squeeze(0)  ## K x N
        afeats = torch.einsum('ns,n->ns', slide_sub_feat, AA)  ### n x fs
        afeat = torch.sum(afeats, dim=0).unsqueeze(0)  ## 1 x fs
        pred = self.classifier(afeat) ## K x num_cls

        return pred, afeats, AA
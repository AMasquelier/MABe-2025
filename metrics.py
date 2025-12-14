from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, accuracy_score, f1_score, classification_report, confusion_matrix
import torch.nn as nn
import torch
import numpy as np


class CCE(nn.Module):
    def __init__(self, reduction='none', smoothing=0.1):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.smoothing = smoothing
    
    def forward(self, p, y, avail_lbl, reduction='mean'):
        B,L,N = p.shape

        lmse = (p.diff(dim=-1)**2).mean()

        #avail_lbl = avail_lbl.unsqueeze(1).repeat(1,N,
        p = p.transpose(1,2).reshape((B*N,L))
        y = y.transpose(1,2).reshape((B*N,L))

        n_classes = p.size(1)
        p = self.softmax(p)

        if self.smoothing>0:
            p = (1-self.smoothing) * p + self.smoothing / n_classes
        p = torch.clamp(p, 1e-8, 1 - 1e-8)

        ce = -y * torch.log(p)
        ce = ce.reshape(B,N,L) * avail_lbl.unsqueeze(1)

        if reduction=='mean': ce = ce.mean()
        return ce + lmse


class BCE(nn.Module):
    def __init__(self, reduction='none', gamma=(2,1), lam=0.5):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.gamma = gamma
        self.lam = lam
    
    def forward(self, p, y, avail_lbl, reduction='mean'):
        B,L,N = p.shape

        n_classes = p.size(1)
        p = self.sigmoid(p)
        p = torch.clamp(p, 1e-4, 1-1e-4)


        # dt = torch.log(p).diff(dim=-1).abs()
        # lmse = (torch.minimum(dt, torch.tensor(4.))**2).mean()

        # pb = p.sum(dim=1)
        # yb = y.sum(dim=1)
        
        p = p.transpose(1,2).reshape((B*N,L))
        y = y.transpose(1,2).reshape((B*N,L))

        #pb = torch.clamp(pb, 1e-4, 1-1e-4)

        #bce = - self.lam * y * (1-p)**self.gamma[0] * torch.log(p) - (1-self.lam) * (1-y) * p**self.gamma[1] * torch.log(1-p)
        #print(p.min(), p.max(), torch.log(p).min(), torch.log(1-p).min())
        bce = - self.lam * y * torch.log(p) - (1-self.lam) * (1-y) * torch.log(1-p)
        bce = bce.reshape(B,N,L) * avail_lbl.unsqueeze(1)

        #bbce = - self.lam * yb * torch.log(pb) - (1-self.lam) * (1-yb) * torch.log(1-pb)
        
        if reduction=='mean': bce = bce.sum() / (avail_lbl.sum() * N) 
        return bce #+ 0.15*lmse


        

def F1_score(targets, outputs, multi=False):
    idx = targets.sum(axis=1)==1
    idx_pred = outputs.max(axis=1)>.2

    print(outputs.max(axis=1).min(), outputs.max(axis=1).max())
    
    targets = targets.round() if multi else targets.argmax(axis=1)
    outputs = outputs.round() if multi else outputs.argmax(axis=1)
    outputs[~idx_pred] = -1

    print(classification_report(idx, idx_pred))
    print(classification_report(targets[idx], outputs[idx]))
    #print(confusion_matrix(targets, outputs))
    
    f1 = f1_score(targets[idx], outputs[idx], average='macro')
    return f1
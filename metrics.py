from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, accuracy_score, f1_score, classification_report, confusion_matrix
import torch.nn as nn
import torch
import numpy as np


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

        p = p.transpose(1,2).reshape((B*N,L))
        y = y.transpose(1,2).reshape((B*N,L))

        bce = - self.lam * y * torch.log(p) - (1-self.lam) * (1-y) * torch.log(1-p)
        bce = bce.reshape(B,N,L) * avail_lbl.unsqueeze(1)

        if reduction=='mean': bce = bce.sum() / (avail_lbl.sum() * N) 
        return bce


        

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
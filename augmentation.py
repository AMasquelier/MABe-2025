import torch
import torchvision
import torch.nn as nn
import numpy as np


class Masking(nn.Module):
    def __init__(self, N=16, lenght=64):
        super().__init__()
        self.N = N
        self.length = lenght
    
    def forward(self, x):
        B,L,C,_ = x.shape
        x = x.clone()

        for _ in range(self.N):
            start = np.random.randint(L)
            l = np.random.randint(self.length-1)+1
            x[:,start:start+l,np.random.randint(C)] *= 0

        return x



class Jitter(nn.Module):
    def __init__(self, mu=0.0, sigma=0.05):
        super().__init__()
        self.sigma = sigma
        self.mu = mu
    
    def forward(self, x):
        x = x + torch.randn_like(x) * self.sigma + self.mu
        return x



class RandomScale(nn.Module):
    def __init__(self, min_scale=0.75, max_scale=1.25):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def forward(self, x):
        B,L,C,_ = x.shape
        x = x.clone()

        x *= self.min_scale + np.random.rand() * (self.max_scale-self.min_scale)

        return x



class Mixup:
    def __init__(self, masking='none', mixup=True, N=1, theta=1):
        self.masking = masking
        self.mixup = mixup
        self.N = N
        self.alpha = 0.5
        self.theta = theta

    def __call__(self, x, y, probs=[]):
        x = x.clone()
        return self.batch_mixup(x,y,probs)
            
        
    def batch_mixup(self, x, y, probs=[]):
        batch_size = x.size(0)
        
        lam = np.random.beta(self.alpha,self.alpha)
        lam = max(lam, 1-lam)
        if len(probs)>0:
            idx = torch.multinomial(probs/probs.sum(), batch_size, replacement=True).to(x.device)
        else:
            idx = torch.randperm(batch_size).to(x.device)

        x = lam * x + (1 - lam) * x[idx]
        y = lam * y + (1 - lam) * y[idx]

        y[y>=self.theta] = 1
        
        return x, y
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import clear_output
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader

class MABeDataset(Dataset):
    PATH = '../../Datasets/MABe-mouse-behavior-detection/'
    config = {"split_size":.2, 'seq_len':1024, 'train_only':False, 'split_method':'split', 'fold':0}

    def process_condition(self, x):
        x = str(x)
        if 'lights off' in x: return 'lights off'
        if 'lights on' in x: return 'lights on'
        return x

    def process_age(self, x):
        x = str(x)
        try:
            if '-' in x: return sum([int(y) for y in x.split(' ')[0].split('-')])/2
            if '>' in x: return 40.0
            return int(x.split(' ')[0])
        except:
            return 15.0

    def process_labels(self, X):
        if isinstance(X, str): X = eval(X)
        if isinstance(X, list):
            y = []
            for x in X:
                y.append(x.split(',')[-1].replace("'", ""))
            return y
        return []


    def get_behaviors_labeled(self, x):
        if isinstance(x, str):
            B = {}
            X = eval(x)
            for x in X:
                m1,m2,b = x.split(',')
                m1,m2 = m1.replace("mouse", ''),m2.replace("mouse", '')
                if (m1,m2) in B: B[(m1,m2)].append(b.replace('"', '').replace("'", ''))
                else: B[(m1,m2)] = [b.replace('"', '').replace("'", '')]
    
            for p in B:
                vec = torch.zeros((len(self.LABELS)))
                for x in B[p]: vec[self.LABELS.index(x)] = 1
                vec[-1]=1
                B[p] = vec
            return B
        return {}
    
    
    def make_labels(self, x):
        out = np.zeros(len(self.LABELS))
        out[self.LABELS.index(x)]=1
        return out

    def __init__(self, is_train=True, config={}):     
        self.config.update(config)

        df = pd.read_csv(self.PATH+'train.csv')
        data = pd.read_csv(self.PATH+'train_data.csv').dropna()

        df['label'] = df.behaviors_labeled.apply(self.process_labels)
        self.LABELS = np.unique(df['label'].explode().dropna()).tolist()+['none']
        df['behaviors_labeled'] = df.behaviors_labeled.apply(self.get_behaviors_labeled)

        for m in range(4):
            df[f'mouse{m+1}_condition'] = df[f'mouse{m+1}_condition'].apply(self.process_condition)
            df[f'mouse{m+1}_age'] = df[f'mouse{m+1}_age'].apply(self.process_age)

        features = ['age', 'sex', 'strain']
        self.FEATURES = {f:set() for f in features}
        for f in features:
            for m in range(4):
                self.FEATURES[f].update(df[f'mouse{m+1}_{f}'].values)            
            self.FEATURES[f] = list(self.FEATURES[f])
            
        
        df = df.merge(data)
        df.index = df[['lab_id', 'video_id']].apply(lambda x: str(x[0])+' - '+str(x[1]), axis=1).values
        
        IDX = np.unique(df.index.values)
        np.random.shuffle(IDX)
        
        if self.config['train_only']: idx = IDX
        elif self.config['split_method']=='k-fold':
            skf = KFold(n_splits=int(1/self.config["split_size"]))
            FOLDS = list(skf.split(IDX))
            train_idx = IDX[FOLDS[self.config['fold']][0]].tolist()
            val_idx = IDX[FOLDS[self.config['fold']][1]].tolist()
            idx = train_idx if is_train else val_idx
        else:
            split_size = self.config['split_size']
            if is_train: idx = IDX[:int((1-split_size)*len(IDX))].tolist()
            else: idx = IDX[int((1-split_size)*len(IDX)):].tolist()
            np.random.shuffle(idx)
        

        DF = df.loc[idx]
        self.fps = DF['frames_per_second'].tolist()
        self.ppc = DF['pix_per_cm_approx'].tolist()
        self.arena_w = DF['arena_w'].tolist()
        self.arena_h = DF['arena_h'].tolist()
        self.arena_shape = DF['arena_shape'].tolist()
        self.path = [tuple(x) for x in DF[['lab_id', 'video_id', 'mouse_1', 'mouse_2']].astype(str).values]
        self.avail_lbl = DF['behaviors_labeled'].tolist()
        self.is_train = is_train

        print(len(self.path))
        self.TRACKS = {}
        with ThreadPoolExecutor(max_workers=24) as executor:
            for _ in executor.map(self.load_track, zip(self.path, self.fps, self.ppc)):
                clear_output(wait=True)
                print(len(self.TRACKS), '/', len(self.path))

        self.DF = DF
        

    def load_track(self, params):
        path, fps, ppc = params
        mice = [str(int(float(x))) for x in path[2:4]]
        X = np.load(self.PATH+f'train_processed/{'_'.join(path[:2])}_{'-'.join(mice)}.npy')
        y = np.load(self.PATH+f'train_processed/{'_'.join(path[:2])}_{'-'.join(mice)}_labels.npy')
        seq_len = self.config['seq_len']

        X = torch.tensor(X)
        y = torch.tensor(y)
        
        n_bp = 7

        x1 = X[:,:n_bp]
        
        if X.shape[1]<n_bp+2:
            s = 1
            x2 = X[:,:n_bp]
        else: 
            s = 0
            x2 = X[:,n_bp:n_bp*2]

        x = torch.concat([x1,x2], dim=1)

        self.TRACKS[path] = (x,y)
        

    def __len__(self):
        return len(self.path)
        

    def __getitem__(self, idx):
        path = self.path[idx]
        seq_len = self.config['seq_len']
        m1,m2 = path[-2],path[-1]

        
        context = []
        for f in self.FEATURES:
            for m in [m1,m2]:
                feat = torch.zeros(len(self.FEATURES[f]))
                if f=='age': context.append(torch.tensor(self.DF.iloc[idx][f'mouse{m}_{f}']).unsqueeze(0))
                else:
                    feat[self.FEATURES[f].index(self.DF.iloc[idx][f'mouse{m}_{f}'])] = 1
                    context.append(feat)
        context.append(torch.tensor(float(path[-3]==path[-2])).unsqueeze(0))
        context.append(torch.tensor(float(self.arena_shape[idx]=='circular')).unsqueeze(0))
        context.append(torch.tensor(self.arena_w[idx]).unsqueeze(0))
        context.append(torch.tensor(self.arena_h[idx]).unsqueeze(0))
        context = torch.cat(context)

        try:
            avail_lbl = self.avail_lbl[idx][(m1,m2 if m1!=m2 else 'self')]
        except:
            print(idx)
            print(self.avail_lbl[idx])
        
        x,y = self.TRACKS[path]

        if len(x)>seq_len and self.is_train:
            a = y[:len(x)-seq_len].numpy().astype(float).sum(axis=-1)
            if a.sum()==0: a+=1
            start = np.random.randint(len(x)-seq_len)#np.random.choice(np.arange(len(a)), p=a/np.sum(a))
        else:
            start = 0

        x = x[start:start+seq_len].transpose(0,1)
        y = y[start:start+seq_len, :-1].transpose(0,1)

        if y.size(1)!=seq_len: 
            x = torch.cat([x, torch.zeros((x.size(0),seq_len-x.size(1),2))], dim=1)
            y = torch.cat([y, torch.zeros((y.size(0),seq_len-y.size(1)))], dim=1)

        return (
            (x.float(), avail_lbl[:-1], context.float()),
            y.float()
        )


def create_dataloaders(config={'batch_size':32, 'num_workers':0}):
    train_dataset = MABeDataset(is_train=True, config=config)
    val_dataset = MABeDataset(is_train=False, config=config)

    torch.manual_seed(config['seed'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    return train_loader, val_loader
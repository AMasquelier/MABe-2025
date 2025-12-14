from models import *
from augmentation import *
from utils import set_seed
from metrics import BCE, F1_score
from dataset import MABeDataset, create_dataloaders

import os
import time
import mlflow
import hashlib
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import clear_output

CFG = {
    'model':ARModel,
    'num_workers':24,
    'seed':2,
    'batch_size':64,
    'losses':[
        BCE(),        
    ],
    'metrics': [
        F1_score
    ],
    'verbose':2,
    'train_only':True,
    'scheduler':True,
    "lr":1e-4,
}


class Trainer:
    
    def __init__(self, config={}, fold=0, epochs=16):
        self.config = CFG.copy()
        self.config.update(config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.exp_id = hashlib.sha256(str(time.time()).encode()).hexdigest()
        #self.experiment = mlflow.set_experiment(EXP_NAME)

        self.aug = torchvision.transforms.Compose([
            RandomScale(),
            Masking()
        ])

        self.mixup = Mixup()

        self.batch_size = self.config['batch_size']
        self.fold = fold

                
    def train_one_epoch(self, epoch=0):
        self.model.train()        
        Loss = 0
        n_steps = len(self.train_loader)
        batch_size = self.config['batch_size']

        
        if self.config['verbose']==2: pbar = tqdm(enumerate(self.train_loader), total=n_steps, desc="Training")
        else: pbar = enumerate(self.train_loader)

        
        for batch_idx, ((x,avail_lbl,ctx), y) in pbar:
            self.optimizer.zero_grad()

            x = x.to(self.device)
            avail_lbl = avail_lbl.to(self.device)
            ctx = ctx.to(self.device)
            y = y.to(self.device)


            x = self.aug(x)
            x,y = self.mixup(x,y)
            
            # Augment + logits
            yp = self.model(x, ctx)
            
            # Loss
            L = self.loss_fn[0](yp, y, avail_lbl)

            L.backward()
            self.optimizer.step()

            Loss += L.detach().item()        
            

        return Loss
            
                
    def validate(self, err_analysis=False):
        self.model.eval()        
        Loss = 0

        n_steps = len(self.val_loader)

        if self.config['verbose']==2: pbar = tqdm(enumerate(self.val_loader), total=n_steps, desc="Validation")
        else: pbar = enumerate(self.val_loader)

        pred_label, pred_words = [], []
        target_label, target_words = [], []

        with torch.no_grad():
            for batch_idx, ((x,avail_lbl,ctx), y) in pbar:
                x = x.to(self.device)
                avail_lbl = avail_lbl.to(self.device)
                ctx = ctx.to(self.device)
                y = y.to(self.device) * avail_lbl.unsqueeze(-1)

                yp = self.model(x, ctx)
                L = self.loss_fn[0](yp, y, avail_lbl)

                Loss += L.detach().item()
                pred_label.append(nn.functional.sigmoid(yp).transpose(1,2).flatten(0,1).detach().cpu().numpy())
                target_label.append(y.transpose(1,2).flatten(0,1).detach().cpu().numpy())

        pred_label = np.concatenate(pred_label)
        target_label = np.concatenate(target_label)


        scores = []
        for m in self.config['metrics']:
            scores.append(m(target_label, pred_label))
            
        return scores, Loss

    
    
    def train(self, epochs=16, checkpoint_freq='once', Lab=None, config=CFG, model=None):
        self.epochs = epochs
        set_seed(seed=self.config['seed'])

        if model==None: self.model = self.config['model'](config=self.config)
        else: self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=1e-8)
        self.loss_fn = self.config['losses']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.train_loader, self.val_loader  = create_dataloaders(fold=self.fold, config=self.config)
        
        best = (np.inf,0,0)

        metrics = []
        
        for m in self.config['metrics']: metrics.append(m)

        cols = ['id', 'epoch', 'train_loss', 'val_loss', 'lr', 'timestamp', 'fold']+['val_'+m.__name__ for m in metrics]+['val_word_'+m.__name__ for m in metrics]#+list([str(x) for x in self.config.keys()])
        self.history = pd.DataFrame([], columns=cols).astype(object)
        
        Epochs = range(epochs)
        with mlflow.start_run(run_name=self.exp_id):
            Epochs = range(epochs)
            for epoch in Epochs:
                set_seed(seed=self.config['seed']+epoch)
                mlflow.log_params(self.config)
                mlflow.log_params({"epochs":epochs, "model_id":self.exp_id})
                # mlflow.log_artifact('tmp/model.py')
                # mlflow.log_artifact('dataset.py')
                # mlflow.log_artifact('modules.py')
            
                train_loss = self.train_one_epoch(epoch=epoch)
                
                if not self.config['train_only']: 
                    val_scores, val_loss = self.validate(err_analysis=(epoch==epochs-1))
                    if val_loss<best[0]:
                        if checkpoint_freq=='best': torch.save(self.model.state_dict(), f"models/{self.exp_id}.pth")
                        best = (val_loss, val_scores,epoch)
                else: val_scores, val_loss = [None] * len(self.config['metrics']), None

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                if not self.config['train_only']: 
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    for v,m in zip(val_scores, self.config['metrics']):
                        mlflow.log_metric("val_"+m.__name__, v, step=epoch)
    
                # self.history.loc[len(self.history), cols] = [self.exp_id, epoch, train_loss, val_loss, self.optimizer.param_groups[0]['lr'], str(datetime.datetime.now()), self.fold, *val_scores]#+[str(x) for x in self.config.values()]
                # self.history.to_csv(f'history/{self.exp_id}.csv', index=False)
                if checkpoint_freq=='epoch': torch.save(self.model.state_dict(), f"models/{self.exp_id}_{epoch}.pth")

                if epoch%2==0:
                    clear_output(wait=True)
                    print(self.exp_id, '\n')
                    print(f"\033[1m Epoch {epoch+1}/{epochs}")
                    print(f'\033[1m Training \t|\t loss={np.round(train_loss, 6)}' + '\033[0m')
                    if not self.config['train_only']:
                        print(f'\033[1m Validation \t|\t loss={np.round(val_loss, 6)} ' + ' - '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(metrics*2, val_scores)])+'\033[0m')
                        print()
                        print(f"\033[1m Best : {'  -  '.join([f'{m.__name__}={np.round(s,3)}' for m,s in zip(metrics, best[1])])} at epoch {best[2]}")
                    
        
                if self.config['scheduler']: self.scheduler.step()

        if not self.config['train_only']: 
                _ = self.validate(err_analysis=True)
        else:
            torch.save(self.model.state_dict(), f"models/{self.exp_id}.pth")
    
        if checkpoint_freq=='once': torch.save(self.model.state_dict(), f"models/{self.exp_id}.pth")
    
        if Lab: Lab.add_model([self.exp_id, epochs, train_loss, val_loss, str(datetime.datetime.now()), self.fold, *val_scores], self.config)#+[repr(x) for x in self.config.values()], self.config)

        return self.model
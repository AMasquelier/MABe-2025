import torch
import torch.nn as nn


class Temporal_Attention(nn.Module):
    """
        IN  : (B,C,H1,...,Hn)
        OUT : (B,C,H1,...,Hn)
    """
    def __init__(self, temp_dim, r=4, n_dim=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dense1 = nn.Linear(temp_dim, temp_dim//r)
        self.dense2 = nn.Linear(temp_dim//r, temp_dim)
        self.n_dim = n_dim

    def forward(self, x):
        s = x # (B,C,H1,...,Hn)
        for _ in range(self.n_dim): s = s.mean(dim=-1) # (B,C)
            
        s = self.dense1(s)
        s = self.relu(s)

        s = self.dense2(s)
        s = self.softmax(s)
        
        for _ in range(self.n_dim): s = s.unsqueeze(-1) # (B,C,1,...,1)

        return (x * s).sum(dim=1)

        

Conv_Dict = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d
}

Deconv_Dict = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d
}

BN_Dict = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d
}

Pooling_Dict = {
    'avg': {
        1: nn.AvgPool1d,
        2: nn.AvgPool2d,
        3: nn.AvgPool3d
    },
    'max': {
        1: nn.MaxPool1d,
        2: nn.MaxPool2d,
        3: nn.MaxPool3d
    },
}


##############################
#         Attention          #
##############################

class Temporal_Attention(nn.Module):
    """
        IN  : (B,C,H1,...,Hn)
        OUT : (B,C,H1,...,Hn)
    """
    def __init__(self, temp_dim, r=4, n_dim=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dense1 = nn.Linear(temp_dim, temp_dim//r)
        self.dense2 = nn.Linear(temp_dim//r, temp_dim)
        self.n_dim = n_dim

    def forward(self, x):
        s = x # (B,C,H1,...,Hn)
        for _ in range(self.n_dim): s = s.mean(dim=-1) # (B,C)
            
        s = self.dense1(s)
        s = self.relu(s)

        s = self.dense2(s)
        s = self.softmax(s)
        
        for _ in range(self.n_dim): s = s.unsqueeze(-1) # (B,C,1,...,1)

        return (x * s).sum(dim=1)

        

class SE_Block(nn.Module):
    def __init__(self, temp_dim, r=4, n_dim=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dense1 = nn.Linear(temp_dim, temp_dim//r)
        self.dense2 = nn.Linear(temp_dim//r, temp_dim)
        self.n_dim = n_dim

    def forward(self, x):
        s = x
        for _ in range(self.n_dim): s = s.mean(dim=-1)
        s = self.dense1(s)
        s = self.relu(s)
        s = self.dense2(s)
        s = self.sigmoid(s)
        for _ in range(self.n_dim): s = s.unsqueeze(-1)
        return x * s


        
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        attn_output, _ = self.attn(query, key_value, key_value)
        return self.norm(attn_output + query)



class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, qkv):
        attn_output, _ = self.attn(qkv, qkv, qkv)
        return self.norm(attn_output + qkv)




##############################
#            CNN             #
##############################

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0., padding='same', activation=nn.GELU, n_dim=1):
        super().__init__()

        CONV, BN = Conv_Dict[n_dim], BN_Dict[n_dim]

        self.conv = nn.Sequential(
            CONV(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            BN(out_channels),
            activation(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, h=None):
        x = self.conv(x) 
        return x
        


class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0., dilation=1, padding='same', activation=nn.ReLU, n_dim=1):
        super().__init__()

        CONV, BN = Conv_Dict[n_dim], BN_Dict[n_dim]

        self.conv = nn.Sequential(
            CONV(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            BN(out_channels),
            nn.Dropout(dropout),
            activation(),
            CONV(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            BN(out_channels),
            nn.Dropout(dropout),
        )
        self.activation = activation()

        if in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                CONV(in_channels, out_channels, kernel_size=1),
                BN(out_channels)
            )
        else: nn.Identity()
        
    def forward(self, x, h=None):
        res = self.shortcut(x)
        x = self.conv(x)
        x = self.activation(x + res)
        return x
        


class SEResConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0., dilation=1, padding='same', activation=nn.GELU, n_dim=1):
        super().__init__()

        CONV, BN = Conv_Dict[n_dim], BN_Dict[n_dim]

        self.conv = nn.Sequential(
            CONV(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            BN(out_channels),
            activation(),
            nn.Dropout(dropout),

            CONV(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            BN(out_channels),
            activation(),
            nn.Dropout(dropout),
        )
        self.activation = activation()
        self.se = SE_Block(out_channels, n_dim=n_dim)

        if in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                CONV(in_channels, out_channels, kernel_size=1, bias=False),
                BN(out_channels),
            )
        else: self.shortcut = nn.Identity()
        
    def forward(self, x, h=None):
        res = self.shortcut(x)
        x = self.conv(x)
        x = self.se(x)
        x = self.activation(x + res)
        return x



class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_dim, first_kernel=7, pooling='max', config={}, pooling_kernel=2, n_dim=1):
        super().__init__()
        self.config = {
            'depth':3,
            'seq_len':64,
            'dropout':0.2,
        }
        if config: self.config.update(config)
        depth = self.config['depth']
        dropout = self.config['dropout']

        if pooling=='max': self.pooling = Pooling_Dict[pooling][n_dim](kernel_size=pooling_kernel)
        elif pooling=='avg': self.pooling = Pooling_Dict[pooling][n_dim](kernel_size=pooling_kernel)
        else: self.pooling = nn.Identity()

        layers = [
            Conv(in_channels, out_dim//(2**(depth-1)), kernel_size=((depth)*2+1), dropout=dropout, n_dim=n_dim),
            self.pooling
        ]
        for d in range(1,depth):
            layers.append(Conv(out_dim//(2**(depth-d)), out_dim//(2**(depth-d-1)), kernel_size=((depth-d)*2+1), dropout=dropout, n_dim=n_dim))
            layers.append(self.pooling)

        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.encoder(x)
        return x



class ResCNN_Block(nn.Module):
    def __init__(self, in_channels, out_dim, pooling='max', dropout=0., padding='same', config={}, n_dim=1):
        super().__init__()
        self.config = {
            'seq_len':64,
            'dropout':0.2,
        }
        if config: self.config.update(config)
        depth = self.config['depth']
        dropout = self.config['dropout']

        if pooling=='max': self.pooling = Pooling_Dict[pooling][n_dim](kernel_size=2)
        elif pooling=='avg': self.pooling = Pooling_Dict[pooling][n_dim](kernel_size=2)
        else: self.pooling = nn.Identity()

        layers = [ResConv(in_channels, out_dim//(2**(depth-1)), kernel_size=7, padding=padding, n_dim=n_dim), self.pooling]
        for d in range(1,depth):
            layers.append(ResConv(out_dim//(2**(depth-d)), out_dim//(2**(depth-d-1)), kernel_size=3, dropout=dropout, padding=padding, n_dim=n_dim))
            layers.append(self.pooling)

        self.encoder = nn.Sequential(*layers)
        self.norm = BN_Dict[n_dim](out_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        return self.norm(x)



class SEResCNN_Block(nn.Module):
    def __init__(self, in_channels, out_dim, kernel_size=3, pooling_kernel=2, pooling='max', dilation=1, activation=nn.ReLU, padding='same', config={}, n_dim=1):
        super().__init__()
        self.config = {
            'seq_len':64,
            'dropout':0.2,
        }
        if config: self.config.update(config)
        depth = self.config['depth']
        dropout = self.config['dropout']

        if pooling=='max': self.pooling = Pooling_Dict[pooling][n_dim](kernel_size=pooling_kernel)
        elif pooling=='avg': self.pooling = Pooling_Dict[pooling][n_dim](kernel_size=pooling_kernel)
        else: self.pooling = nn.Identity()

        layers = [SEResConv(in_channels, out_dim//(2**(depth-1)), kernel_size=kernel_size, dilation=dilation, padding=padding, n_dim=n_dim), self.pooling]
        for d in range(1,depth):
            layers.append(SEResConv(out_dim//(2**(depth-d)), out_dim//(2**(depth-d-1)), kernel_size=kernel_size, dilation=dilation, padding=padding, dropout=dropout, n_dim=n_dim))
            layers.append(self.pooling)

        self.encoder = nn.Sequential(*layers)
        self.norm = BN_Dict[n_dim](out_dim)
        self.activation = activation()
        
    def forward(self, x):
        x = self.encoder(x)
        return self.activation(self.norm(x))
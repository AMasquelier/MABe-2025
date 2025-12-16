import torch
import torch.nn as nn



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
        


def Upsample(n_dim, scale_factor=2, mode="nearest"):
    """
    Returns an interpolation-based upsampler that inverts the pooling step.
    mode ∈ {"nearest","linear"}; for n_dim>1 this maps to bilinear/trilinear.
    """
    mode_map = {
        1: {"nearest": "nearest", "linear": "linear"},
        2: {"nearest": "nearest", "linear": "bilinear"},
        3: {"nearest": "nearest", "linear": "trilinear"},
    }
    chosen = mode_map[n_dim][mode]
    # align_corners only valid for *linear modes; harmlessly ignored by 'nearest'
    return nn.Upsample(scale_factor=scale_factor, mode=chosen, align_corners=False if chosen != "nearest" else None)



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0., padding='same', activation=nn.ReLU, n_dim=1):
        super().__init__()

        CONV, BN = Conv_Dict[n_dim], BN_Dict[n_dim]

        self.conv = nn.Sequential(
            CONV(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            BN(out_channels),
            activation(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, h=None):
        x = self.conv(x) 
        return x



class Deconv(nn.Module):
    def __init__( self, in_channels, out_channels, kernel_size, dropout=0.0, activation=nn.ReLU, n_dim=1, stride=1, padding=0, output_padding=0, dilation=1, bias=True):
        super().__init__()
        DECONV, BN = Deconv_Dict[n_dim], BN_Dict[n_dim]

        self.block = nn.Sequential(
            DECONV(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, bias=bias),
            BN(out_channels),
            activation(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)



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



class ResDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0., dilation=1, padding='same', activation=nn.ReLU, n_dim=1):
        super().__init__()

        DECONV, BN = Deconv_Dict[n_dim], BN_Dict[n_dim]

        self.conv = nn.Sequential(
            DECONV(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            BN(out_channels),
            nn.Dropout(dropout),
            activation(),
            DECONV(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            BN(out_channels),
            nn.Dropout(dropout),
        )
        self.activation = activation()

        if in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                DECONV(in_channels, out_channels, kernel_size=1),
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



class SEResDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0., dilation=1, padding='same', activation=nn.GELU, n_dim=1):
        super().__init__()

        DECONV, BN = Deconv_Dict[n_dim], BN_Dict[n_dim]

        self.conv = nn.Sequential(
            DECONV(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            BN(out_channels),
            activation(),
            nn.Dropout(dropout),

            DECONV(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            BN(out_channels),
            activation(),
            nn.Dropout(dropout),
        )
        self.activation = activation()
        self.se = SE_Block(out_channels, n_dim=n_dim)

        if in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                DECONV(in_channels, out_channels, kernel_size=1, bias=False),
                BN(out_channels),
            )
        else: self.shortcut = nn.Identity()
        
    def forward(self, x, h=None):
        res = self.shortcut(x)
        x = self.conv(x)
        x = self.se(x)
        x = self.activation(x + res)
        return x
        


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


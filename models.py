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



class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, activation=nn.GELU, padding='same', return_skip=True, bias=True):
        super().__init__()
        self.pool = nn.AvgPool1d(2)
        self.conv = SEResConv(in_channels, out_channels, kernel_size, dilation=dilation, activation=activation, padding=padding, dropout=0.3)
        self.return_skip = return_skip
        
    def forward(self, x):
        high = self.conv(x)
        low = self.pool(high)        

        if self.return_skip: 
            return low, high
        return low



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, activation=nn.GELU, padding='same', bias=True):
        super().__init__()
        if padding=='same': padding = kernel_size//2*dilation
        self.conv = SEResDeconv(in_channels, out_channels, kernel_size, dilation=dilation, activation=activation, padding=padding, dropout=0.3)
        self.upsample = Upsample(1,2)
        self.attention = CrossAttention(in_channels, 8)
        
    def forward(self, x, skip):
        x = self.upsample(x)
        #x = self.conv(x)
        x = self.attention(x.transpose(-1,-2),skip.transpose(-1,-2)).transpose(-1,-2)
        x = self.conv(x)

        return x



class UNet(nn.Module):
    def __init__(self, in_channels, hidden_dim, padding='same', activation=nn.GELU, n_dim=1):
        super().__init__()

        self.enc_1 = EncoderBlock(in_channels, hidden_dim, 5, 1)
        self.enc_2 = EncoderBlock(hidden_dim, hidden_dim*2, 5, 2)
        self.enc_3 = EncoderBlock(hidden_dim*2, hidden_dim*4, 5, 4)

        self.bottleneck = SEResConv(hidden_dim*4, hidden_dim*4, 3, dropout=0.3)

        self.dec_1 = DecoderBlock(hidden_dim*4, hidden_dim*2, 5, 4, padding=8)
        self.dec_2 = DecoderBlock(hidden_dim*2, hidden_dim, 5, 2, padding=4)
        self.dec_3 = DecoderBlock(hidden_dim, hidden_dim, 5, 1, padding=2)
        self.head =  nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 37, 5, padding=2, dilation=1, bias=True)
        ) 
        
    def forward(self, x, h=None):
        
        x,skip1=self.enc_1(x)
        x,skip2=self.enc_2(x)
        x,skip3=self.enc_3(x)

        x = self.bottleneck(x)

        x=self.dec_1(x,skip3)
        x=self.dec_2(x,skip2)
        x=self.dec_3(x,skip1)

        x=self.head(x)
            
        return x

    

class Feature_Engineering(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config: self.config.update(config)
        self.pdist = torch.nn.PairwiseDistance(p=2)

    def forward(self, x, context):
        B,C,N,_ = x.shape
        C = C//2
        x1 = x[:,:C]
        x2 = x[:,C:]

        arena_wh = context[:,[-2,-1]].unsqueeze(1).unsqueeze(1).to(x.device)
        zero = torch.zeros((1,1,1,2)).to(x.device)
        circle = context[:,-3].unsqueeze(1).unsqueeze(1)

        dist_wall_circle_1 = torch.abs((x1-arena_wh/2).norm(dim=-1)-context[:,-1].unsqueeze(1).unsqueeze(1)/2)
        dist_wall_rect_1,_ = torch.cat([torch.abs(x1-arena_wh), torch.abs(x1-zero)], dim=-1).min(dim=-1)
        dist_wall_1 = circle * dist_wall_circle_1 + (1-circle) * dist_wall_rect_1

        dist_wall_circle_2 = torch.abs((x2-arena_wh/2).norm(dim=-1)-context[:,-1].unsqueeze(1).unsqueeze(1)/2)
        dist_wall_rect_2,_ = torch.cat([torch.abs(x2-arena_wh), torch.abs(x2-zero)], dim=-1).min(dim=-1)
        dist_wall_2 = circle * dist_wall_circle_2 + (1-circle) * dist_wall_rect_2
        

        rel_x = (x1-x2)
        rel_dist = rel_x.norm(dim=-1)

        dx1 = x1.diff(dim=2, prepend=x1[:, :, :1])
        dx2 = x2.diff(dim=2, prepend=x2[:, :, :1])

        adx1 = torch.einsum('...i,...i->...', dx1[:, :, 1:], dx1[:, :, :-1]) / (dx1[:, :, 1:].norm(dim=-1) * dx1[:, :, :-1].norm(dim=-1) + 1e-4)
        adx1 = torch.cat([torch.zeros_like(adx1[:, :, :1]), adx1], dim=2)

        adx2 = torch.einsum('...i,...i->...', dx2[:, :, 1:], dx2[:, :, :-1]) / (dx2[:, :, 1:].norm(dim=-1) * dx2[:, :, :-1].norm(dim=-1) + 1e-4)
        adx2 = torch.cat([torch.zeros_like(adx2[:, :, :1]), adx2], dim=2)

        dx1_thresh = (dx1.norm(dim=-1)>.1).float()
        dx2_thresh = (dx2.norm(dim=-1)>.1).float()

        dx1_= dx1.norm(dim=-1)
        dx2_= dx2.norm(dim=-1)

        ddx1 = dx1.diff(dim=2, prepend=dx1[:, :, :1])
        ddx2 = dx2.diff(dim=2, prepend=dx2[:, :, :1])

        cross_prod_1 = (dx1[:,:,:,[0]] * ddx1[:,:,:,[1]] - dx1[:,:,:,[1]] * ddx1[:,:,:,[0]]).squeeze(dim=-1)
        cross_prod_2 = (dx2[:,:,:,[0]] * ddx2[:,:,:,[1]] - dx2[:,:,:,[1]] * ddx2[:,:,:,[0]]).squeeze(dim=-1)
        
        dirs = torch.einsum('...i,...i->...', dx1, dx2) / (dx1.norm(dim=-1) * dx2.norm(dim=-1) + 1e-6)
        cross = (dx1[:,:,:,[0]] * dx2[:,:,:,[1]] - dx1[:,:,:,[1]] * dx2[:,:,:,[0]]).squeeze(dim=-1)
        d = torch.cat([
            self.pdist(x1.roll(i+1, dims=1), x2) #* mask_x1.roll(i+1, dims=1) * mask_x2
        for i in range(x1.size(1))], dim=1)
        
        dd = torch.cat([d,torch.zeros_like(d[:,:,:1])], dim=2).diff(dim=-1)


        lead_1 = torch.einsum('...i,...i->...', dx1, rel_x) / (dx1.norm(dim=-1) * rel_x.norm(dim=-1) + 1e-6)
        lead_2 = torch.einsum('...i,...i->...', dx2, -rel_x) / (dx2.norm(dim=-1) * rel_x.norm(dim=-1) + 1e-6)
        drel_x = rel_x.diff(dim=2, prepend=rel_x[:, :, :1])

        
        tail_to_neck_1 = (x1[:, [6]] - x1[:, [4]])
        neck_to_nose_1 = (x1[:, [4]] - x1[:, [5]])
        tail_to_nose_1 = (x1[:, [6]] - x1[:, [5]])
        head_angle_1 = torch.einsum('...i,...i->...', tail_to_neck_1, neck_to_nose_1) / (tail_to_neck_1.norm(dim=-1) * neck_to_nose_1.norm(dim=-1) + 1e-6)

        tail_to_neck_2 = (x2[:, [6]] - x2[:, [4]])
        neck_to_nose_2 = (x2[:, [4]] - x2[:, [5]])
        tail_to_nose_2 = (x2[:, [6]] - x2[:, [5]])
        head_angle_2 = torch.einsum('...i,...i->...', tail_to_neck_2, neck_to_nose_2) / (tail_to_neck_2.norm(dim=-1) * neck_to_nose_2.norm(dim=-1) + 1e-6)

        ears_span_1 = (x1[:, [0]] - x1[:, [1]]).norm(dim=-1)
        ears_span_2 = (x2[:, [0]] - x2[:, [1]]).norm(dim=-1)

        curl1 = (tail_to_neck_1.norm(dim=-1) + neck_to_nose_1.norm(dim=-1)) / (tail_to_nose_1.norm(dim=-1) + 1e-4)
        curl2 = (tail_to_neck_2.norm(dim=-1) + neck_to_nose_2.norm(dim=-1)) / (tail_to_nose_2.norm(dim=-1) + 1e-4)

        # f_1 = torch.einsum('...i,...i->...', dx1, tail_to_nose_2) / (dx1.norm(dim=-1) * tail_to_nose_2.norm(dim=-1) + 1e-6)
        # f_2 = torch.einsum('...i,...i->...', dx2, tail_to_nose_1) / (dx2.norm(dim=-1) * tail_to_nose_1.norm(dim=-1) + 1e-6)

        # f_1 = f_1 * dx1_thresh
        # f_2 = f_2 * dx2_thresh

        # wh_ratio_1 = self.pdist(x1[:,[2]],x1[:,[3]])/self.pdist(x1[:,[5]],x1[:,[6]])
        # wh_ratio_2 = self.pdist(x2[:,[2]],x2[:,[3]])/self.pdist(x2[:,[5]],x2[:,[6]])

        x = torch.concat([dx1_, dx2_, cross_prod_1, cross_prod_2, adx1, adx2, dirs, d, dd, lead_1, lead_2], dim=1)
        return x



class ARModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = {
            'dropout':0.2,
        }
        if config: self.config.update(config)

        self.training = True

        self.feature_eng = Feature_Engineering()
        n_channels = 7
        n_features = n_channels*(n_channels*2+9)

        self.context_encoder = nn.Sequential(
            nn.Linear(71, n_features),
            nn.ReLU(),
        )

        base_h_dim = 128

        # self.unet = UNet(n_features, base_h_dim)

        self.encoder = nn.Sequential(
            SEResConv(n_features, base_h_dim, 5, dilation=1, dropout=0.3),
            nn.AvgPool1d(2),
            SEResConv(base_h_dim, base_h_dim*2, 5, dilation=2, dropout=0.3),
            nn.AvgPool1d(2),
            SEResConv(base_h_dim*2, base_h_dim*4, 5, dilation=4, dropout=0.3),
            # nn.AvgPool1d(2),
            # Conv(base_h_dim*4, base_h_dim*8, 9),
        )
        
        self.decoder = nn.Sequential(
            # Deconv(base_h_dim*8, base_h_dim*4, 9, padding=4),
            # Upsample(1,2),
            SEResDeconv(base_h_dim*4, base_h_dim*2, 5, padding=8, dilation=4, dropout=0.3),
            Upsample(1,2),
            SEResDeconv(base_h_dim*2, base_h_dim, 5, padding=4, dilation=2, dropout=0.3),
            Upsample(1,2),
            nn.ConvTranspose1d(base_h_dim, 37, 5, padding=2, dilation=1, bias=True),
        )
        #self.ca = CrossAttention(n_features, 7)


    def forward(self, x, context):
        x = self.feature_eng(x, context)
        #c = self.context_encoder(context)
        
        #x = self.ca(x.transpose(-2,-1), c.unsqueeze(-2)).transpose(-2,-1)
        
        x = self.encoder(x)
        x = self.decoder(x)

        # x = self.unet(x)
        
        return x
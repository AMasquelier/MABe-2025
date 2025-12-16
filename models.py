import torch
import torch.nn as nn

from modules import *



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
        
        x = torch.concat([dx1_, dx2_, cross_prod_1, cross_prod_2, adx1, adx2, dirs, d, dd, lead_1, lead_2], dim=1)
        return x



class ARModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = {
        }
        if config: self.config.update(config)

        self.training = True

        self.feature_eng = Feature_Engineering()
        n_channels = 7
        n_features = n_channels*(n_channels*2+9)

        
        base_h_dim = 128

        self.encoder = nn.Sequential(
            SEResConv(n_features, base_h_dim, 5, dilation=1, dropout=0.3),
            nn.AvgPool1d(2),
            SEResConv(base_h_dim, base_h_dim*2, 5, dilation=2, dropout=0.3),
            nn.AvgPool1d(2),
            SEResConv(base_h_dim*2, base_h_dim*4, 5, dilation=4, dropout=0.3),
        )
        
        self.decoder = nn.Sequential(
            SEResDeconv(base_h_dim*4, base_h_dim*2, 5, padding=8, dilation=4, dropout=0.3),
            Upsample(1,2),
            SEResDeconv(base_h_dim*2, base_h_dim, 5, padding=4, dilation=2, dropout=0.3),
            Upsample(1,2),
            nn.ConvTranspose1d(base_h_dim, 37, 5, padding=2, dilation=1, bias=True),
        )


    def forward(self, x, context):
        x = self.feature_eng(x, context)
        
        x = self.encoder(x)
        x = self.decoder(x)

        return x

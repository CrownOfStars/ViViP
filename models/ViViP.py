import torch
import torch.nn as nn
from einops import rearrange, repeat

from toolkit.Transformer import *
from toolkit.Poolformer import *
from toolkit.misc import *

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

poolformer = {
            "s12":poolformer_s12,
            "s24":poolformer_s24,
            "s36":poolformer_s36
        }

class ViViP(nn.Module):
    def __init__(self, num_classes = 101, depth = 8, heads = 8,  dropout = 0.5,pool_model="s24"):
        super().__init__()

        self.pool_model = pool_model

        self.space_poolformer = poolformer.get(self.pool_model)(pretrained=True)

        self.dim = self.space_poolformer.embed_dims[-1]
        
        self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        dim_head = 64

        mlp_scale = 4

        self.temporal_transformer = Transformer(self.dim, depth, heads, dim_head, self.dim*mlp_scale, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, num_classes)
        )

    def forward(self, x):

        b, c, t, h, w = x.shape
        
        x = rearrange(x,'b c t h w -> (b t) c h w')

        x = self.space_poolformer.forward_embeddings(x)

        x = self.space_poolformer.forward_tokens(x)

        x = rearrange(x, '(b t) n h w -> b (t h w) n', b=b,t=t)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x[:,0]

        return self.mlp_head(x)

if __name__ == "__main__":
    
    imgs = torch.randn([4,3,8,224,224]).cuda()
    labels = torch.randn([4,101]).ge(0).float().cuda()
    model = ViViP(pool_model="s24").cuda()

    loss_fn = nn.CrossEntropyLoss()

    preds = model(imgs)
    
    loss = loss_fn(preds,labels)

    print(loss.item())
    print(get_parameter_num(model))

    print(get_macs(model,imgs))
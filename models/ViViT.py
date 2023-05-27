from toolkit.Transformer import *
from toolkit.misc import *

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ViViT(nn.Module):
    def __init__(self, crop_size = 224,patch_size = 16, num_classes = 101, num_frames = 16,\
                  dim = 512, depth = 8, heads = 8, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        num_patches = (crop_size//patch_size)**2

        patch_dim = 3 * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        
        x = x.permute(0,2,1,3,4)
        print(x.shape)
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        
        x = x[:, 0]

        return self.mlp_head(x)


def get_vivit_model(cfg=None):
    if cfg:
        pass
    else:
        # default_cfg = 
        # print("use defalut cfg\n:{}".format(default_cfg))
        return ViViT(224,16,101,16)

if __name__ == "__main__":
    
    img = torch.randn([8, 3, 16, 224, 224]).cuda()
    labels = torch.randn([8,101]).ge(0).float().cuda()
    model = ViViT(224, 16, 101, 16).cuda()

    pred = model(img)
    
    loss_fn = nn.CrossEntropyLoss()

    loss = loss_fn(pred,labels)
    
    print(loss.item())
    
    print(get_parameter_num(model))
    print(get_macs(model,img))
    
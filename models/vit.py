# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
#import pdb

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):  # layerNorm
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # scaling: 1/root(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        # print("self.to_qkv", self.to_qkv)  # Linear(in_features=512, out_features=1536, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # print("self.to_qkv(x)", self.to_qkv(x).shape)  # torch.Size([16, 65, 1536])  B, N+1, dim * 3
        # print("qkv[0]", qkv[0].shape)  # torch.Size([16, 65, 512])  --> B, N+1, dim
        
        # map fuction: map(function, iterable)
        # explanation: https://blockdmask.tistory.com/531
        # h --> multi-head
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) # dim = h * d --> d = dim / h(==8) = 64
        # pdb.set_trace()
        # print("q shape", q.shape)  # torch.Size([16, 8, 65, 64])  --> B, h, N+1, d

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # Q x K_T * scale
        # (Batch, multi-head, i, dim), (Batch, multi-head, j, dim) --> (Batch, multi-head, i, j) , i = j = 65
        # print("dots shape", dots.shape)  # torch.Size([16, 8, 65, 65]

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # attention score
        # print("attn shape", attn.shape)  # torch.Size([16, 8, 65, 65])

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # [16, 8, 65, 65] x [16, 8, 65, 64] -> [16, 8, 65, 64]
        # print("matmul out shape", out.shape)  # torch.Size([16, 8, 65, 64])
        
        out = rearrange(out, 'b h n d -> b n (h d)')  # [16, 8, 65, 64] -> [16, 65, 512]  --> concat head
        # print("rearrange out shape", out.shape)  # torch.Size([16, 65, 512])
        
        out =  self.to_out(out)
        # print("linear out shape", out.shape)  # torch.Size([16, 65, 512])
        
        return out


## Transformer encoder block
# 2 subblock: attention subblock + feed forward subblock
# attention subblock: input embedding ==> Multi-Head attention --> LayerNorm --> Residual
# feed forward subblock: attention subblock output ==> Feed forward --> LayerNorm --> Residual
# depth: Transformer * depth block

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # 6
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        
        ##### Patch Embedding instants #####
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2  # 3//2=1 vs 3/2=1.5
        patch_dim = channels * patch_size ** 2  # image patch dim: P^2 * C
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.patch_size = patch_size  # default patch size = 16( == 4*4)
        # print('patch size: ', patch_size)  # 4 (for cifar10)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # Epos: (num_patches + 1) * D; (total patchs + class token) * D
        # print("self.pos_embedding: ",self.pos_embedding.shape)  # torch.Size([1, 65 (N + 1), 512])
        self.patch_to_embedding = nn.Linear(patch_dim, dim)  # linear projection: P^2 * C ---> target dim (512)
        # print("self.patch_to_embedding: ",self.patch_to_embedding)  # Linear(in_features=48 (4*4*3), out_features=512, bias=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # define cls token as 1 * dim
        # print("self.cls_token: ",self.cls_token.shape)  # torch.Size([1, 1, 512])
        ##### Patch Embedding instants #####
        
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask = None):
        
        ##### Patch Embedding start #####
        p = self.patch_size

        # print("img.shape(): ", img.shape)  # 16, 3, 32, 32
        
        # Batch x C x H x W --> Batch x N x PSIZE
        # b c (h p1) (w p2) -> b (h w) (p1 p2 c); H = h * p1, W = w * p2 ==> h*w = H*W/(P1*P2)
        # N = H*W/P^2, PSIZE = C * P^2
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # print("rearranged x: ", x.shape)  # 16, 64(N = 8*8), 48(4*4*3)
        
        x = self.patch_to_embedding(x)  # linear projection: P^2 * C ---> target dim (512)
        b, n, _ = x.shape
        # print("embedinng x: ", x.shape)  # 16, 64, 512(target dim)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        # print("cls_tokens", cls_tokens.shape)  # torch.Size([16, 1, 512])

        x = torch.cat((cls_tokens, x), dim=1)  # concat
        x += self.pos_embedding[:, :(n + 1)]  # sum pos embedding
        # print("final embedding: ", x.shape)  # torch.Size([16, 65, 512])
        ##### Patch Embedding end #####
        
        x = self.dropout(x)

        x = self.transformer(x, mask)
        # print("transformer output", x.shape)  # torch.Size([16, 65, 512]) : same as input shape
        # print ("x[:,0]", x[:,0].shape)  # torch.Size([16, 512])

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_cls_token(x[:, 0])
        # print("fowarded cls_token", x.shape)  # torch.Size([16, 512]) : only use class token
        
        rst = self.mlp_head(x)
        # print("mlp: ", rst.shape)  # torch.Size([16, 10]) --> 10 classes

        return rst

        # return self.mlp_head(x)
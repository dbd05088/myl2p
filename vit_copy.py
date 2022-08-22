import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

x = torch.randn(8, 3, 224, 224) # batch size 8, channel size 3, h,w=(224,224)의 size

patch_size = 16 # 16 pixels
print('x :', x.shape)

'''
# [8, 3, 224, 224] => patch_size=3(channel)*16*16 = 768이므로, patch의 갯수는 3*224*224 / 768 = 196개가 된다
# 이 과정을 아래와 같이 einops와 같은 Linear embedding을 통해서도 구현할 수 있지만, Convnet 2D layer를 사용하면, performance gain이 있다고 한다.
patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
'''
patch_size = 16
in_channels = 3
img_size = 224
emb_size = 768 #(= patch size ... 3*16*16)
projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
# 이미지를 패치사이즈로 나누고 flatten
projected_x = projection(x)
print('Projected X shape :', projected_x.shape)

##### cls token과 positional encoding 하는 과정
# cls_token과 pos encoding Parameter 정의
cls_token = nn.Parameter(torch.randn(1,1, emb_size)) # size는 1이고, patch의 가장 앞부분에 덧붙여야 하기 때문에 embedding size만큼의 크기를 갖는 1개의 token이다.
positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size)) # +1을 해주는 이유는 196개의 patch에 pos encoding을 해준 후, 1개의 cls token까지 있기 때문에 총 197개! 여기도 윗줄과 같은 이유로 size = embedding size로!
print('Cls Shape :', cls_token.shape, ', Pos Shape :', positions.shape) # Cls Shape : torch.Size([1, 1, 768]) , Pos Shape : torch.Size([197, 768])

# cls_token을 반복하여 배치사이즈의 크기와 맞춰줌 (each sample마다 cls_token을 붙여주는 것)
batch_size = 8
cls_tokens = repeat(cls_token, '() n e -> b n e', b=batch_size)
print('Repeated Cls shape :', cls_tokens.shape)

# cls_token과 projected_x를 concatenate
cat_x = torch.cat([cls_tokens, projected_x], dim=1)

# position encoding을 더해줌
cat_x += positions
print('output : ', cat_x.shape)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.name = "PatchEmbed"
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape # b는 batch size
        x = self.projection(x)

        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)

        # add position embedding
        x += self.positions

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.1):
        linear1 = nn.Linear(emb_size, expansion * emb_size)
        linear2 = nn.Linear(expansion * emb_size, emb_size)

        # linear layer's xavier initialization
        torch.nn.init.xavier_uniform_(linear1.weight)
        torch.nn.init.xavier_uniform_(linear2.weight)
        super().__init__(
            linear1,
            nn.GELU(),
            nn.Dropout(drop_p),
            linear2,
            nn.Dropout(drop_p)
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))

class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
        
summary(ViT(), (3, 224, 224), device='cpu')
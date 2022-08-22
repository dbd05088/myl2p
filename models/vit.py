from statistics import mode
from sys import prefix
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import functools
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from zmq import NULL
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from models.prompt import Prompt
from models.prompt import expand_to_batch
from models.prompt import prepend_prompt
import ml_collections

x = torch.randn(8, 3, 224, 224) # batch size 8, channel size 3, h,w=(224,224)의 size

patch_size = 16 # 16 pixels
#print('x :', x.shape)

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
#print('Projected X shape :', projected_x.shape)

##### cls token과 positional encoding 하는 과정
# cls_token과 pos encoding Parameter 정의
cls_token = nn.Parameter(torch.randn(1,1, emb_size)) # size는 1이고, patch의 가장 앞부분에 덧붙여야 하기 때문에 embedding size만큼의 크기를 갖는 1개의 token이다.
positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size)) # +1을 해주는 이유는 196개의 patch에 pos encoding을 해준 후, 1개의 cls token까지 있기 때문에 총 197개! 여기도 윗줄과 같은 이유로 size = embedding size로!
#print('Cls Shape :', cls_token.shape, ', Pos Shape :', positions.shape) # Cls Shape : torch.Size([1, 1, 768]) , Pos Shape : torch.Size([197, 768])

# cls_token을 반복하여 배치사이즈의 크기와 맞춰줌 (each sample마다 cls_token을 붙여주는 것)
batch_size = 8
cls_tokens = repeat(cls_token, '() n e -> b n e', b=batch_size)
#print('Repeated Cls shape :', cls_tokens.shape)

# cls_token과 projected_x를 concatenate
cat_x = torch.cat([cls_tokens, projected_x], dim=1)

# position encoding을 더해줌
cat_x += positions
#print('output : ', cat_x.shape)

'''
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 32):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # queries = rearrange(self.proj_q(x), "b n (h d) -> b h n d", h=self.num_heads)
            #Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape # b는 batch size
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        return x
'''

class PositionalEmbedding(nn.Module):
    def __init__(self, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, (img_size // patch_size) **2 + 1, emb_size))
        
    def forward(self, x: Tensor) -> Tensor:
        x += self.pos_embedding # 이 과정에서 dimension이 바뀌지는 않는다.
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        #self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.proj_q = nn.Linear(emb_size, emb_size)
        self.proj_k = nn.Linear(emb_size, emb_size)
        self.proj_v = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.proj_q(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.proj_k(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.proj_v(x), "b n (h d) -> b h n d", h=self.num_heads)
        # queries, keys, values = qkv[0], qkv[1], qkv[2]
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

class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, expansion * emb_size)
        self.fc2 = nn.Linear(expansion * emb_size, emb_size)
        self.GELU = nn.GELU()
        self.Dropout = nn.Dropout(drop_p)

        # linear layer's xavier initialization
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.GELU(out)
        out = self.Dropout(out)
        out = self.fc2(out)
        out = self.Dropout(out)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 index: int = 0,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__()
        self.name = f'encoderblock_{index}'
        self.drop_p = drop_p
        self.emb_size = emb_size
        self.norm1 = nn.LayerNorm(self.emb_size)
        self.norm2 = nn.LayerNorm(self.emb_size)
        self.proj = nn.Linear(self.emb_size, self.emb_size)
        self.attn = MultiHeadAttention(emb_size, **kwargs)
        self.Dropout = nn.Dropout(drop_p) 
        self.pwff = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        
    def forward(self, x):
        out = self.norm1(x)
        out = self.attn(out)
        out = self.Dropout(out)
        out += x

        out = self.norm2(x)
        out = self.pwff(out)
        out = self.Dropout(out)
        out += x
        
        return out

class TransformerEncoder(nn.Module):
    '''
                                    depth, 
                                    self.prefix, 
                                    use_prefix_tune_for_e_prompt=self.use_prefix_tune_for_e_prompt,
                                    use_prefix_tune_for_g_prompt=self.use_prefix_tune_for_g_prompt, 
                                    g_prompt_layer_idx=self.g_prompt_layer_idx,
                                    e_prompt_layer_idx=self.e_prompt_layer_idx,
                                    emb_size=emb_size
    '''
    def __init__(self, 
            prefix, 
            use_prefix_tune_for_e_prompt, 
            use_prefix_tune_for_g_prompt, 
            g_prompt_layer_idx, 
            e_prompt_layer_idx, 
            emb_size, 
            depth: int = 12, 
            **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        self.prefix = prefix
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt 
        self.use_prefix_tune_for_g_prompt = use_prefix_tune_for_g_prompt
        self.g_prompt_layer_idx = g_prompt_layer_idx 
        self.e_prompt_layer_idx = e_prompt_layer_idx
        self.emb_size = emb_size

    def forward(self, x, train, prompt):
        # pos embedding is already added to input x (in before step)
        prompt_counter = -1
        prefix_layer = None
        batched_prompt = None
        for lyr in range(len(self.blocks)):
            if (prefix is not None) and (lyr in self.g_prompt_layer_idx):
                if not self.use_prefix_tune_for_g_prompt:
                    batched_prompt = prefix[lyr]  # len * hiddensize
                    batched_prompt = expand_to_batch(
                        batched_prompt, batch_size=x.shape[0])
                else:
                    prefix_layer = prefix[lyr]
                    # batch it here
                    prefix_layer = expand_to_batch(
                        prefix_layer, batch_size=x.shape[0], axis=1)

            else:
                prefix_layer = None

            if prompt is not None:
                if isinstance(self.e_prompt_layer_idx, int):
                    self.e_prompt_layer_idx = [self.e_prompt_layer_idx]
                if lyr in self.e_prompt_layer_idx:
                    prompt_counter += 1
                    if self.use_prefix_tune_for_e_prompt:
                        if prefix_layer is not None:
                        # concatenate shared prompt/prefix with this prefix
                            prefix_layer = torch.concat(
                            [prefix_layer, prompt[prompt_counter]], axis=-3)
                        else:
                            prefix_layer = prompt[prompt_counter]
                else:
                    # do the concatenation here
                    if batched_prompt is not None:
                        batched_prompt = torch.concat(
                            [batched_prompt, prompt[prompt_counter]], axis=-2)
                        x = prepend_prompt(batched_prompt, x)
                    else:
                        batched_prompt = prompt[prompt_counter]
                        x = prepend_prompt(batched_prompt, x)

            x = self.blocks[lyr](x)
        # TODO Check!!
        #encoded = nn.LayerNorm(name='encoder_norm')(x)
        return x

'''
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 100):
        super().__init__(
            #Reduce('b n e -> b e', reduction='mean'),
            #nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))
'''

class VisionTransformer(nn.Module):
    def __init__(self,
                rep_input_size,
                num_layers,
                num_heads,
                num_classes,
                hidden_size,
                device,
                training: bool = False,
                norm_pre_logits: bool = False,
                temperature: float = 1.0,
                representation_size = None,
                classifier: str = 'token',
                use_cls_token: bool = True,
                prompt_params= None,
                reweight_prompt: bool = False,
                num_tasks: int = -1,
                prefix_params = None,
                prompt_contrastive_temp: float = -1.0,
                num_classes_per_task: int = -1,
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 100,
                **kwargs):
        '''
            num_classes: int
            patches: ml_collections.ConfigDict #16x16
            transformer: ml_collections.ConfigDict
            hidden_size: int
            train: bool = False
            norm_pre_logits: bool = False
            temperature: float = 1.0
            representation_size: Optional[int] = None
            classifier: str = 'token'
            use_cls_token: bool = True
            prompt_params: Any = None
            reweight_prompt: bool = False
            num_tasks: int = -1
            prefix_params: Any = None
            prompt_contrastive_temp: float = -1.0
            num_classes_per_task: int = -1
        '''
        super().__init__()
        self.depth = depth
        self.rep_input_size = rep_input_size
        self.device = device
        self.prefix_layer = None
        self.batched_prompt = None
        self.prompt_counter = -1
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_params = prefix_params
        if self.prefix_params is not None:
            self.g_prompt_length = prefix_params['g_prompt_length']
            self.g_prompt_layer_idx = prefix_params['g_prompt_layer_idx']
        self.training = training
        self.norm_pre_logits = norm_pre_logits
        self.temperature = temperature
        self.representation_size = representation_size
        self.classifier = classifier
        self.use_cls_token = use_cls_token
        self.prompt_params = prompt_params
        self.reweight_prompt = reweight_prompt
        self.num_tasks = num_tasks
        self.prompt_contrastive_temp = prompt_contrastive_temp
        self.num_classes_per_task = num_classes_per_task

        if self.representation_size is not None:
            self.pre_logit = nn.Linear(self.rep_input_size, self.representation_size)
            self.tanh = nn.tanh()
        # init prefix
        self.use_prefix_tune_for_g_prompt = True
        if self.prefix_params is not None:
            n_layers = self.num_layers
            n_heads = self.num_heads

            self.g_prompt_length = self.g_prompt_length
            self.g_prompt_layer_idx = self.g_prompt_layer_idx
            self.embedding_size = self.hidden_size // n_heads

            '''
            self.prompt = torch.nn.Parameter(w)
            self.prompt.normal_(w, mean=0.0, std=0.01) 
            '''

            if not self.prefix_params['use_prefix_tune_for_g_prompt']:
                self.use_prefix_tune_for_g_prompt = False
                w = torch.empty((n_layers, self.g_prompt_length, self.hidden_size))
                self.prefix = torch.nn.Parameter(w)
                self.prefix.uniform_(0, 0.01)
                
            else:
                # 1.4: added for the same key and value
                if self.prefix_params['same_key_value']:
                    w = torch.empty((n_layers, 1, self.g_prompt_length, n_heads, self.embedding_size))
                    self.prefix = torch.nn.Parameter(w)
                    self.prefix.uniform_(0, 0.01)
                    self.prefix = np.tile(self.prefix, (1, 2, 1, 1, 1))
                else:
                    w = torch.empty((n_layers, 2, self.g_prompt_length, n_heads, self.embedding_size))
                    self.prefix = torch.nn.Parameter(w)
                    self.prefix.uniform_(0, 0.01)
        else:
            self.prefix = None
            self.g_prompt_layer_idx = []

        if self.prefix_params is not None:
            if not self.prefix_params['use_prefix_tune_for_g_prompt']:
                self.total_prompt_len = self.prefix_params['g_prompt_length'] * len(
                    self.prefix_params['g_prompt_layer_idx'])

        # res_vit["embedding"] = x
        # put it after class token for now

        if self.prompt_params is not None:
            # set up number of layers
            if isinstance(self.prompt_params['e_prompt_layer_idx'], int):
                self.num_prompted_layers = 1
            else:
                self.num_prompted_layers = len(self.prompt_params['e_prompt_layer_idx'])
            # set up if using prefix-style prompts or not
            self.use_prefix_tune_for_e_prompt = self.prompt_params['use_prefix_tune_for_e_prompt']
            if self.use_prefix_tune_for_e_prompt:
                self.same_key_value_for_pool = self.prompt_params['same_key_value']
            self.e_prompt_layer_idx = self.prompt_params['e_prompt_layer_idx']
            # set up number of heads for prefix
            self.num_heads = self.num_heads

            # pylint: disable=unsupported-membership-test
            prompt_pool_params = self.prompt_params['prompt_pool']
            '''
            if prompt_pool_params.initializer == 'normal':
                initializer = nn.initializers.normal()
            # for now we don't have other initilizers besides uniform and normal
            else:
                initializer = nn.initializers.uniform()
            '''
            self.prompt_pool_module = Prompt(
                x_embedded_dim = emb_size, #x_embed.shape[-1]
                length=prompt_pool_params['length'],
                embedding_key=prompt_pool_params['embedding_key'],
                prompt_init=prompt_pool_params['initializer'], 
                prompt_pool=True,
                prompt_key=prompt_pool_params['prompt_key'],
                pool_size=prompt_pool_params['pool_size'],
                top_k=prompt_pool_params['top_k'],
                batchwise_prompt=prompt_pool_params['batchwise_prompt'],
                prompt_key_init=prompt_pool_params['prompt_key_init'],
                num_classes_per_task=self.num_classes_per_task,
                num_layers=self.num_prompted_layers,
                use_prefix_tune_for_e_prompt=self.use_prefix_tune_for_e_prompt,
                num_heads=num_heads,
                num_tasks=self.num_tasks,
            )
        
        # using a conv layer instead of a linear one -> performance gains
        self.patch_embedding = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

        #self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.positional_embedding = PositionalEmbedding(patch_size=patch_size, emb_size=emb_size, img_size = img_size)

        self.class_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.transformer= TransformerEncoder(
                                    depth= depth, 
                                    prefix = self.prefix, 
                                    use_prefix_tune_for_e_prompt=self.use_prefix_tune_for_e_prompt,
                                    use_prefix_tune_for_g_prompt=self.use_prefix_tune_for_g_prompt, 
                                    g_prompt_layer_idx=self.g_prompt_layer_idx,
                                    e_prompt_layer_idx=self.e_prompt_layer_idx,
                                    emb_size=emb_size)

        #self.ClassificationHead = ClassificationHead(emb_size, n_classes)
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, n_classes)


    def forward(self, input, prompt_mask = None, task_id = -1, cls_features = None, label = None):
        x = input
        n, c, h, w = x.shape
        res_vit = dict()

        #print("###input shape###")
        #print(x.shape)
        
        ###### step 1. Patch Embedding 
        x = self.patch_embedding(x)
        #print("###after patch###")
        #print(x.shape)
        x = rearrange(x, "b e (h) (w) -> b (h w) e")
        #print("###after rearramge###")
        #print(x.shape)

        ###### step 2. Add cls_token
        cls_tokens = repeat(self.class_token, '() a c -> b a c', b=n)
        x = torch.cat([cls_tokens, x], dim=1)

        ###### step 3. PosEmbedding
        x = self.positional_embedding(x)

        # Here, x is a grid of embeddings.
        #x = np.reshape(x, [n, h * w, c])
        
        # res_vit["embedding"] = x
        # put it after class token for now
        if self.prompt_params is not None:
            res_vit = self.prompt_pool_module(
                x,
                self.device, 
                prompt_mask = prompt_mask,
                task_id=task_id,
                cls_features=cls_features,
                label=label)
            self.batched_prompt = res_vit['batched_prompt']

            self.total_prompt_len = 0
            if self.prefix_params:
                if not self.prefix_params['use_prefix_tune_for_g_prompt']:
                    self.total_prompt_len += self.prefix_params['g_prompt_length'] * len(self.prefix_params['g_prompt_layer_idx'])

            for key in self.prompt_params:  # pylint: disable=not-an-iterable
                if not self.use_prefix_tune_for_e_prompt:
                    if key == 'prompt_pool':
                    # make it multi-layered prompts
                        self.total_prompt_len += self.prompt_params[key]['length'] * self.prompt_params[key]['top_k'] * self.num_prompted_layers
                    elif key == 'shared_prompt' or key == 'task_specific_prompt':
                        self.total_prompt_len += self.prompt_params[key]['length'] * self.num_prompted_layers

        # If we want to add a class token, add it here.
        # already
        ''' 
        if self.use_cls_token:
            cls = torch.zeros((1, 1, c))
            cls = np.tile(cls, [n, 1, 1])
            x = np.concatenate([cls, x], axis=1)
        '''
        #print("batched_prompt")

        s0, s1, s2, s3 = self.batched_prompt.shape
        self.batched_prompt = torch.reshape(self.batched_prompt, (s1, s0, s2, s3))
        #print(self.batched_prompt.shape)
        x = self.transformer(
            x, 
            train=self.training,
            prompt=self.batched_prompt,
            )
        #print("x shape", self.batched_prompt.shape)
        #print("classifier", self.classifier)
        if self.use_cls_token and self.classifier == 'token':
            if self.prompt_params:
                x = x[:, self.total_prompt_len]
            else:
                x = x[:, 0]
        elif self.classifier == 'gap':
            x = np.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
        elif self.classifier == 'prompt':
            x = x[:, 0:self.total_prompt_len]
            if self.reweight_prompt:
                w = torch.empty((self.total_prompt_len,))
                reweight = torch.nn.init.uniform_(w, 0, 0.01)
                reweight = nn.softmax(reweight)
                x = np.average(x, axis=1, weights=reweight)
            else:
                x = np.mean(x, axis=1)
        elif self.use_cls_token and self.prompt_params and self.classifier == 'token+prompt':
            x = x[:, 0:self.total_prompt_len + 1]
            x = np.mean(x, axis=1)
        else:
            raise ValueError(f'Invalid classifier={self.classifier}')
        # Added for utilizing pretrained features
        res_vit['pre_logits'] = x
        if self.representation_size is not None:
            x = self.pre_logit(x)
            x = self.tanh(x)

        '''
        if self.norm_pre_logits:
            eps = 1e-10
            x_norm = torch.linalg.norm(x, ord=2, dim=-1, keepdims=True)
            x = x / (x_norm + eps)
        x = self.ClassificationHead(x)
        '''

        x = self.norm(x)
        x = self.fc(x)
        
        x = x / self.temperature
        #print("final_x.shape", x.shape)
        res_vit['logits'] = x 
        return res_vit

def create_model(name, sgd_momentum, optim, weight_decay, norm_pre_logits, temperature, use_e_prompt, e_prompt_layer_idx, use_prefix_tune_for_e_prompt, use_cls_token,
    vit_classifier,
    num_tasks,
    num_total_class,
    num_classes_per_task,
    device,
    prompt_pool_param):
    """Creates model partial function."""
    # add pre logits normalization or not

    prompt_params = {}
    # Specify which layer the prompt should be add on
    prompt_params['e_prompt_layer_idx'] = e_prompt_layer_idx
    # Using prefix-tuning for E-Prompt
    prompt_params['use_prefix_tune_for_e_prompt'] = use_prefix_tune_for_e_prompt
    # If using the same key and value in prefix
    prompt_params['same_key_value'] = False #TODO config.get('same_key_value_for_pool')
    prompt_params['prompt_pool'] = prompt_pool_param

    '''
    # l2p에서는 g_prompt 사용하지 X
    if config.get('use_g_prompt'):
        prefix_params = {}
        prefix_params['g_prompt_length'] = config.g_prompt_length
        prefix_params['g_prompt_layer_idx'] = config.g_prompt_layer_idx
        prefix_params['same_key_value'] = config.get('same_key_value_for_shared')
        prefix_params['use_prefix_tune_for_g_prompt'] = config.get(
            'use_prefix_tune_for_g_prompt')
        model_config['prefix_params'] = prefix_params
    '''

    dic = get_b16_config()
    transformer_dic = dic['transformer']
    return VisionTransformer(num_classes=num_total_class,
                            num_classes_per_task=num_classes_per_task,
                            num_tasks=num_tasks,
                            device = device,
                            hidden_size=dic['hidden_size'], 
                            num_layers=transformer_dic['num_layers'], 
                            num_heads=transformer_dic['num_heads'], 
                            rep_input_size=dic['representation_size'], 
                            patch_size=dic['size'][0], prompt_params=prompt_params, norm_pre_logits=norm_pre_logits, temperature=temperature)
        
#summary(ViT(), (3, 224, 224), device='cpu')
#mymodel = ViT()
#from pytorch_pretrained_vit import ViT
#mymodel = ViT('B_16_imagenet1k', pretrained=True)
'''
for name, param in mymodel.named_parameters():
    if param.requires_grad:
        print (name)
'''
def get_b16_config():
    dic = {}
    dic['size'] = (16,16)
    dic['hidden_size'] = 768
    transformer = {}
    transformer['mlp_dim'] = 3072
    transformer['num_heads'] = 12
    transformer['num_layers'] = 12
    transformer['attention_dropout_rate'] = 0.0
    transformer['dropout_rate'] = 0.0
    dic['transformer'] = transformer
    dic['classifier'] = 'token'
    dic['representation_size'] = None
    return dic
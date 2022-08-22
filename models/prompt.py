from multiprocessing import pool
from unittest.case import DIFF_OMITTED
from torch import nn
from typing import Any, Callable, Sequence
import numpy as np
import torch
import copy
Initializer = Callable[[np.array, Sequence[int]], np.array]

def l2_normalize(x, device, axis=None, epsilon=1e-12):
    """l2 normalizes a tensor on an axis with numerical stability."""
    square_sum = torch.sum(torch.square(x), axis=axis, keepdims=True)
    #x_inv_norm = torch.reciprocal(torch.sqrt(torch.maximum(square_sum, torch.full(square_sum.shape, epsilon).to('cuda:0'))))
    x_inv_norm = torch.reciprocal(torch.sqrt(torch.maximum(square_sum, torch.full(square_sum.shape, epsilon).to(device))))
    return x * x_inv_norm

def expand_to_batch(x, batch_size: int, axis=0):
    """Expands unbatched `x` to the specified batch_size`."""
    #print("x's shape", x.shape)
    shape = [1 for _ in x.shape]
    #print("shape", shape)
    shape.insert(axis, batch_size)
    #print("expanded", torch.tile(np.expand_dims(x, axis=axis), shape).shape)
    return torch.tile(np.expand_dims(x, axis=axis), shape)

def prepend_prompt(prompt, x_embed):
    '''
    print("prompt shape")
    print(prompt.shape)
    print("x_embed shape")
    print(x_embed.shape)
    '''
    """Concatenates `prompt` to the beginning of `x_embed`.

    Args:
        prompt: [B, P, H] The prompt.
        x_embed: [B, T, H] The embedded input.

    Returns:
        The input with the prompt concatenated to the front. [B, P + T, H]
    """
    return torch.concat([prompt, x_embed], axis=1)

'''
L2P에서 Learnable한 것들은?

Q) input x에 대해서 input=>same dimension as the key로 바꿔주는 query function q(x)는 learnable할까?
No! 모든 Task에서 같은 input x에 대해서 same q(x)를 output으로 내보내줘야 하기 때문에, deterministic하게 하기 위해서 No learnable하게!

Then, 뭐가 Learnable?? => 3가지
1. Key
l2p는 task들이 prompt pool을 share하는 구조이다.
이때, prompt들이 (key, prompt)와 같이 key, value의 구조로 되어 있다.
q(x)와 key들 간의 similarity를 비교하여 top-k를 추출하기 때문에, 저 key가 learnable하여 올바른 prompt들이 input에 맞게 뽑히도록 학습된다.

2. Prompt
Prompt가 Learnable하여 input x의 embedding에 prepend되었을 때, classification시 더 잘 context 역할을 할 수 있도록 해준다.

3. Classifier
x의 embedding에 prompt가 prepend된 것을 pretrained-transformer-encoder에 통과시킨 후, 나온 결과 값을 classifier에 넣고 ground truth y와 비교한다.
이때, cross entropy loss를 이용해서 classifier의 parameter phi를 학습시킨다.
'''
class Prompt(nn.Module):
    """Promp module including prompt pool and prompt selection mechanism.

    This is the training time version of prompting a model. Calling the injected
    `prompt` module will generate your unbatched prompt. This model then
    replicates it for the batched input and concatenates them together.

    Attributes:
        length: Length of each prompt.
        embedding_key: Way of calculating the key feature. Choose from "mean",
        "max", "mean_max", "cls".
        prompt_init: Initilaizer of the prompt parameters.
        prompt_pool: If use prompt pool or not.
        prompt_key: If use separate prompt key parameters or not.
        pool_size: Size of the prompt pool (number of prompts).
        top_k: Top k prompt to prehend to the input. (0 < top_k < pool_size)
        batchwise_prompt: If use the same set or prompts for the same batch,
        following majority vote rule.
        prompt_key_init: Initialization ways of the prompt key parameter.
        num_classes_per_task: Num of classes per task.
        num_layers: int = 1 Number of layers to add prompts
        use_prefix_tune_for_e_prompt: If use prefix-tuning for E-Prompt
        num_heads: Number of heads for MSA
        same_key_value: If use the same key and value for prefix-tuning
        num_tasks: Total number of tasks in the continual learning setting.
    """
    def __init__(self,
                x_embedded_dim: int,
                length: int,
                embedding_key: str = "mean",
                prompt_init: str = "uniform", # upper bound of uniform distrib is 0.01
                prompt_pool: bool = False,
                prompt_key: bool = False,
                pool_size: int = None,
                top_k: int = None,
                batchwise_prompt: bool = False,
                prompt_key_init: str = "zero",
                num_classes_per_task: int = -1,
                num_layers: int = 1,
                use_prefix_tune_for_e_prompt: bool = False,
                num_heads: int = -1,
                same_key_value: bool = False,
                num_tasks: int = 5):
        super().__init__()
        self.length = length
        self.x_embedded_dim = x_embedded_dim
        print("x_embedded_dim", x_embedded_dim)
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_pool = prompt_pool
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.prompt_key_init = prompt_key_init
        self.num_classes_per_task = num_classes_per_task
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.num_tasks = num_tasks

        if self.prompt_pool:
            if self.use_prefix_tune_for_e_prompt:  # use prefix style
                assert self.x_embedded_dim % self.num_heads == 0
                if self.same_key_value:
                    '''
                    prompt = self.param("prompt", self.prompt_init,
                    (self.num_layers, 1, self.pool_size, self.length,
                    self.num_heads, embed_dim // self.num_heads)) # (): parameter about dimension
                    prompt = jnp.tile(prompt, (1, 2, 1, 1, 1, 1))
                    '''
                    w = torch.empty(self.num_layers, 1, self.pool_size, self.length,
                    self.num_heads, self.x_embedded_dim // self.num_heads)
                    
                    if self.prompt_init == "normal":
                        w = torch.nn.init.normal_(w, mean=0.0, std=0.01)
                    else:
                        w = torch.nn.init.uniform_(w, 0, 0.01)
                    self.prompt = torch.nn.Parameter(w, requires_grad=True)
                    self.prompt = torch.tile(self.prompt, (1, 2, 1, 1, 1, 1)) # why? in prefix tuning??
                else:
                    w = torch.empty(self.num_layers, 2, self.pool_size, self.length,
                                        self.num_heads, self.x_embedded_dim // self.num_heads)
                    if self.prompt_init == "normal":
                        w = torch.nn.init.normal_(w, mean=0.0, std=0.01)
                    else:
                        w = torch.nn.init.uniform_(w, 0, 0.01)
                    
                    self.prompt = torch.nn.Parameter(w, requires_grad=True)
            else:
                # R(LxD) ... token length L
                # # of prompt = pool_size
                w = torch.empty(self.num_layers, self.pool_size, self.length, self.x_embedded_dim)
                if self.prompt_init == "normal":
                    w = torch.nn.init.normal_(w, mean=0.0, std=0.01)  
                else:
                    w = torch.nn.init.uniform_(w, 0, 0.01)
                self.prompt = torch.nn.Parameter(w, requires_grad=True)

            # if using learnable prompt keys
            if self.prompt_key:
                key_shape = (self.pool_size, self.x_embedded_dim)
                if self.prompt_key_init == "zero":
                    w = torch.zeros(key_shape)
                elif self.prompt_key_init == "uniform":
                    w = torch.empty(key_shape)
                    w = nn.init.uniform_(w, 0, 0.01)
                self.prompt_key = torch.nn.Parameter(w, requires_grad=True)
            else:
                # only compatible with prompt, not prefix
                prompt_mean = np.mean(self.prompt, axis=[0, 2])  # pool_size, emb
                self.prompt_key = prompt_mean

        else:
            if self.use_prefix_tune_for_e_prompt:  # use prefix style
                assert self.x_embedded_dim % self.num_heads == 0
                if self.same_key_value:
                    w = torch.empty((self.num_layers, 1, self.length, self.num_heads,
                                        self.x_embedded_dim // self.num_heads))
                    if self.prompt_init == "normal":
                        w = torch.nn.init.normal_(w, mean=0.0, std=0.01)  
                    else:
                        w = torch.nn.init.uniform_(w, 0, 0.01)
                    self.prompt = torch.nn.Parameter(w, requires_grad=True)
                    self.prompt = torch.tile(self.prompt, (1, 2, 1, 1, 1)) # prompt => (1, 2, 1, 1, 1) dimension array
                else:
                    w = torch.empty((self.num_layers, 2, self.length, self.num_heads,
                                        self.x_embedded_dim // self.num_heads))
                    if self.prompt_init == "normal":
                        w = torch.nn.init.normal_(w, mean=0.0, std=0.01)  
                    else:
                        w = torch.nn.init.uniform_(w, 0, 0.01)
                    self.prompt = torch.nn.Parameter(w, requires_grad=True)
            else:
                w = torch.empty((self.num_layers, self.length, self.x_embedded_dim))
                if self.prompt_init == "normal":
                    w = torch.nn.init.normal_(w, mean=0.0, std=0.01)  
                else:
                    w = torch.nn.init.uniform_(w, 0, 0.01)
                self.prompt = torch.nn.Parameter(w, requires_grad=True)


    def __call__(self, x_embed, device, prompt_mask=None, task_id=-1, cls_features=None, label=None):
        #print("x_embed shape", x_embed.shape)
        if self.prompt_pool:
            # now get key matching part.
            if self.embedding_key == "mean":
                x_embed_mean = torch.mean(x_embed, dim=1)  # bs, emb
            elif self.embedding_key == "max":
                x_embed_mean = torch.max(x_embed, dim=1)
            elif self.embedding_key == "mean_max":
                x_embed_mean = torch.max(x_embed, dim=1) + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == "cls":
                if cls_features is None:  # just for init
                    x_embed_mean = torch.mean(x_embed, dim=1)
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError(
                    "Not supported way of calculating embedding keys!")

            '''
            prompt_key_norm = l2_normalize(self.prompt_key.data, device, axis=-1)
            if type(x_embed_mean) != torch.Tensor:
                #print("텐서아니네")
                x_embed_norm = l2_normalize(x_embed_mean.values, device, axis=-1)
            else:
                #print("텐서네")
                x_embed_norm = l2_normalize(x_embed_mean, device, axis=-1)
            '''
            #TODO nn.norm을 통해서 해야 backprop이 잘되려나??
            prompt_key_norm = torch.nn.functional.normalize(self.prompt_key)
            x_embed_norm = torch.nn.functional.normalize(x_embed_mean)

            # prompt key query and x_embeded_norm's similarity calculate by mat mul, transpose
            sim = torch.matmul(prompt_key_norm, torch.transpose(x_embed_norm, 0, 1))  # pool_size, bs or pool_size, #class, bs

            sim = torch.transpose(sim, 0, 1)  # bs, pool_size
            #(sim_top_k, idx) = jax.lax.top_k(sim, self.top_k)
            
            (sim_top_k, idx) = torch.topk(sim, self.top_k) #select top-k which has high similarity 

            if self.batchwise_prompt: # use same prompt in same batch
                prompt_id, id_counts = np.unique(
                    idx, return_counts=True) # id_counts: each unique item appears counting
                cp_prompt_id = copy.deepcopy()
                size = len(id_counts)
                while size<self.pool_size:
                    if size+len(id_counts) > self.pool_size:
                        cp_prompt_id.extend(prompt_id[:self.pool_size % len(id_counts)])
                        size += (self.pool_size % len(id_counts))
                    else:
                        cp_prompt_id.extend(prompt_id)
                        size += len(id_counts)
                _, major_idx = torch.topk(id_counts, self.top_k) # Return tok_k value and their indexes
                major_prompt_id = prompt_id[major_idx]
                idx = expand_to_batch(major_prompt_id, x_embed.shape[0])

            res = dict()
            # np.take example
            '''
            a = [4, 3, 5, 7, 6, 8]
            np.take(a, [[0, 1], [2, 3]])
            array([[4, 3],
                [5, 7]])
            '''

            if prompt_mask is not None:
                idx = prompt_mask  # bs, allowed_size
            idx = idx.detach().cpu().to(device)
            res["prompt_idx"] = idx
            #print("idx", idx)
            # Using expanded idx, make key_norm

            '''
            batched_key_norm = torch.index_select(
                prompt_key_norm, 0, idx)  # bs, top_k, embed_dim / From prompt_key_norm, get data of idx.
            '''
            batched_key_norm = []
            for i in idx:
                batched_key_norm.append(torch.index_select(prompt_key_norm, 0, i))
            batched_key_norm = torch.stack(batched_key_norm, dim=0)

            res["selected_key"] = batched_key_norm 

            '''
            In prefix tuning
            w = torch.empty(self.num_layers, 2, self.pool_size, self.length, self.num_heads, embed_dim // self.num_heads)
            In here, np.take(~, idx, axis=2) => then, self.pool_size => allowed_size

            In not prefix tuning, not 2=> 1. So, axis = 1 to reach self.pool_size
            w = torch.empty(self.num_layers, 1, self.pool_size, self.length, self.num_heads, embed_dim // self.num_heads)
            In here, np.take(~, idx, axis=1) => then, self.pool_size => allowed_size           

            '''
            # bs means batch size
            # check prefix tuning
            if self.use_prefix_tune_for_e_prompt:
                # by taking using np.take => add 1 more dimension which means batch size(bs)

                batched_prompt_row = []
                for i in idx:
                    batched_prompt_row.append(torch.index_select(self.prompt, 2, i))
                batched_prompt_row = torch.stack(batched_prompt_row, dim=0)

                # batched_prompt = torch.index_select(self.prompt, 2, idx)  # num_layers, bs, allowed_size, prompt_len, embed_dim
                num_layers, bs, dual, allowed_size, prompt_len, num_heads, heads_embed_dim = batched_prompt_row.shape
                batched_prompt = torch.reshape(batched_prompt_row, (num_layers, bs, dual, allowed_size *prompt_len, num_heads, heads_embed_dim))
            else:
                batched_prompt_row = []
                for i in idx:
                    batched_prompt_row.append(torch.index_select(self.prompt, 1, i))
                batched_prompt_row = torch.stack(batched_prompt_row, dim=0)

                num_layers, bs, allowed_size, prompt_len, embed_dim = batched_prompt_row.shape
                batched_prompt = torch.reshape(batched_prompt_row, (num_layers, bs, allowed_size * prompt_len, embed_dim))
                    
            res["batched_prompt"] = batched_prompt
            res["prompt_key_norm"] = prompt_key_norm
            res["x_embed_norm"] = x_embed_norm
            res["sim"] = sim

            # Put pull_constraint loss calculation inside
            # By using np.newaxis => expand 1 dimension

            # x_embed_norm = x_embed_norm[:, np.newaxis, :]  # bs, 1, embed_dim
            x_embed_norm = torch.unsqueeze(x_embed_norm, 1)
            sim_pull = batched_key_norm * x_embed_norm
            reduce_sim = torch.sum(sim_pull) / x_embed.shape[0] # divide by bs
            res["reduce_sim"] = reduce_sim

        else:
            if self.use_prefix_tune_for_e_prompt:  # use prefix style
                batched_prompt = expand_to_batch(self.prompt, x_embed.shape[0], axis=2)
            else:
                batched_prompt = expand_to_batch(self.prompt, x_embed.shape[0], axis=1)

        res["batched_prompt"] = batched_prompt
        return res


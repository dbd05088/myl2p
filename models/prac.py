import logging
from typing import Any, Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


Array = jnp.ndarray
Initializer = Callable[[Array, Sequence[int]], Array] # Array와 int type의 sequence를 input으로 받으며 Array를 return한다는 것을 의미

class Prompt(nn.Module):
  prompt_init: Initializer = nn.initializers.uniform()

  @nn.compact
  def __call__(self,
               label=None):
    
    prompt = self.param("prompt", self.prompt_init,
                        (2, 1, 4, 3,
                        1, 3))
    prompt = jnp.tile(prompt, (1, 2, 1, 1, 1, 1))
    print("prompt")
    print(prompt)


p = Prompt
p()
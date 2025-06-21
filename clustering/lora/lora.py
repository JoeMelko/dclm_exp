from typing import List, Optional

import torch.nn as nn
import torch

from .logger import Logger
from functools import partial
import re

def _get_submodules(model, key):
    """
    Helper function to replace a module with transformers model
    https://github.com/huggingface/peft/blob/c0dd27bc974e4a62c6072142146887b75bb2de6c/src/peft/utils/other.py#L251
    """
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

class LoraLinear(nn.Module):
    def __init__(self, rank: int, linear: nn.Linear):
        """Transforms a linear layer into a LoraLinear layer.

        Args:
            rank (int): The rank of lora
            linear (nn.Linear): The linear layer to transform
        """
        super().__init__()

        in_features = linear.in_features
        out_features = linear.out_features

        # self.rank = min(rank, in_features, out_features)
        self.rank = rank
        init_std = 0.02

        self.logix_lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.logix_lora_B = nn.Linear(self.rank, self.rank, bias=False)
        self.logix_lora_C = nn.Linear(self.rank, out_features, bias=False)

        nn.init.normal_(self.logix_lora_A.weight, std=init_std)
        nn.init.zeros_(self.logix_lora_B.weight)
        nn.init.normal_(self.logix_lora_C.weight, std=init_std)

        self._linear = linear  # wrapped linear layer

    def forward(self, x):
        base = self._linear(x)             # original matmul
        lora = self.logix_lora_C(self.logix_lora_B(self.logix_lora_A(x)))
        return base + lora
    
    # ---------- proxy weight & bias so external code can access them ---------
    @property
    def weight(self):
        return self._linear.weight

    @property
    def bias(self):
        return self._linear.bias

class LoRAHandler:
    """
    Transforms a model into a Lora model.
    """

    _SUPPORTED_MODULES = {nn.Linear}

    def __init__(self, rank):
        self.rank = rank
        torch.manual_seed(0)

    def add_lora(
        self,
        model: nn.Module,
        logger: Logger,
        batch_size: int,
    ):

        device = next(model.parameters()).device
        index = 0  # track how many modules are being added

        for name, module in model.named_modules():
            if not module_check(module, name, self._SUPPORTED_MODULES):
                continue
            # Skip modules with "output" in their name
            if "output" in name.lower():
                continue
            
            # determine which module type
            lora_fn = LoraLinear
            # create the attachment
            lora_module = lora_fn(self.rank, module)
            lora_module.to(device)
            # get additional info for attachment
            parent, _, target_name = _get_submodules(model, name)
            # insert module
            setattr(parent, target_name, lora_module)
            # set up the hook
            # TODO: determine which bucket to hook on --> index
            bucket_id = get_block_id(name, model.n_layers // logger.num_blocks)
            if bucket_id is not None:
                lora_module.logix_lora_B.register_forward_hook(
                    partial(logger.hook_fn, index=bucket_id)
                )

# ---- tag lists taken from open_lm/model.py ----------------------------------
ATTN_TAGS = (           # self-attention linear leaves
    ".attention.",      # layers.<i>.attention.in_proj / out_proj
    ".in_proj",         # covers in_proj.weight / bias if you name leaves that way
    ".out_proj",
)

FFN_TAGS = (            # feed-forward (GELU, SwiGLU, GeGLU, etc.)
    ".feed_forward.",   # layers.<i>.feed_forward.{w12,w3,gate_proj,up_proj,down_proj}
    "._ff_",            # layers.<i>._ff_w1 / _ff_w2 (GELU variant)
    ".w12", ".w3",
    ".gate_proj", ".up_proj", ".down_proj",
)
# -----------------------------------------------------------------------------


def get_block_id(module_name: str, block_size: int = 4) -> Optional[int]:
    """
    Return the *block index* (0-based) for a Linear leaf inside an open-lm
    Transformer, or None if the module name doesn't look like a transformer
    attention/FFN Linear.

    A "block" is `block_size` consecutive layers, irrespective of whether the
    module is in the attention or FFN sub-stack.

    Parameters
    ----------
    module_name : str
        Fully-qualified name returned by `model.named_modules()`.
    block_size : int, default 4
        Number of transformer layers per block.

    Examples
    --------
    >>> get_block_id("layers.7.attention.in_proj", 4)
    1
    >>> get_block_id("layers.15.feed_forward.w12", 4)
    3
    """
    # 1. Grab the integer layer index ("layers.<idx>.")
    m = re.search(r"layers\.(\d+)\.", module_name)
    if m is None:
        return None  # not part of the transformer stack

    layer_idx = int(m.group(1))

    # 2. Check if this is attention or FFN, and compute separate indices
    is_attn = "attention" in module_name
    is_ffn = "feed_forward" in module_name
    
    if not (is_attn or is_ffn):
        return None

    # 3. Block id is floor-division by block_size, with separate indices for attn/ffn
    block_idx = layer_idx // block_size
    
    if is_attn:
        return block_idx * 2  # even indices for attention
    else:  # is_ffn
        return block_idx * 2 + 1  # odd indices for FFN


def module_check(
    module: nn.Module,
    module_name: str,
    supported_modules: Optional[List[nn.Module]] = None,
    type_filter: Optional[List[nn.Module]] = None,
    name_filter: Optional[List[str]] = None,
    is_lora: bool = False,
) -> bool:
    """
    Check if the module is supported for logging.

    Args:
            module (nn.Module): The module to check.
            module_name (str): Name of the module.
            supported_modules (Optional[List[nn.Module]]): List of supported module types.
            type_filter (Optional[List[nn.Module]]): List of module types to filter.
            name_filter (Optional[List[str]]): List of keywords to filter module names.
            is_lora (bool): Flag to check for specific 'analog_lora_B' in module names.

    Returns:
            bool: True if module is supported, False otherwise.
    """
    if list(module.children()):
        return False
    if supported_modules and not isinstance(module, tuple(supported_modules)):
        return False
    if type_filter and not isinstance(module, tuple(type_filter)):
        return False
    if name_filter and not any(keyword in module_name for keyword in name_filter):
        return False
    if is_lora and "logix_lora_B" not in module_name:
        return False
    return True
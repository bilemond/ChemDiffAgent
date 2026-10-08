import os
from dataclasses import dataclass
from typing import Optional, Union
from transformers import AutoConfig
from torch import nn


@dataclass
class Config:
    # model can be either a local path (str) or an already loaded HF model instance (nn.Module)
    model: Union[str, nn.Module]
    max_num_batched_tokens: int = 8192 * 2
    max_num_seqs: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.75
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: Optional[AutoConfig] = None
    # When initializing from an HF model instance, we keep a reference here
    hf_model: Optional[nn.Module] = None
    # Optional hint for where to load tokenizer from (defaults to model path if str)
    tokenizer_name_or_path: Optional[str] = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    mask_token_id: int = -1
    block_length: int = 128

    def __post_init__(self):
        # Two initialization modes:
        # 1) Path-based: model is a directory containing weights/config/tokenizer
        # 2) In-memory:  model is a Hugging Face model instance (nn.Module) with .config
        if isinstance(self.model, str):
            # Path-based init
            assert os.path.isdir(self.model), f"Model path not found: {self.model}"
            self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
            # Default tokenizer path is the model path
            if self.tokenizer_name_or_path is None:
                self.tokenizer_name_or_path = self.model
        elif isinstance(self.model, nn.Module):
            # In-memory init from an existing HF model instance
            self.hf_model = self.model
            # Try to obtain its config
            cfg = getattr(self.model, "config", None)
            assert cfg is not None, "HF model instance must have a .config"
            self.hf_config = cfg
            # Best-effort tokenizer source
            if self.tokenizer_name_or_path is None:
                name_or_path = getattr(cfg, "name_or_path", None)
                if isinstance(name_or_path, str):
                    self.tokenizer_name_or_path = name_or_path
        else:
            raise TypeError("Config.model must be a str path or an nn.Module (HF model instance)")

        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # hf_config is guaranteed to be set in the branches above
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.mask_token_id != -1, "Mask token ID must be set"

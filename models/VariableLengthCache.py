"""
改进的VariableLengthCache实现
通过attention_mask自动推断每个样本的有效长度
"""

import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache


class VariableLengthCache(Cache):
    """
    支持batch内不同样本追加不同长度的Cache (改进版)

    核心改进:
    - 通过attention_mask自动推断每个样本的有效新增长度
    - 与Qwen3的forward方法无缝集成
    - 不需要手动传递valid_new_lengths

    工作原理:
    1. 记录每个样本上一次的有效长度
    2. 通过当前attention_mask计算新增的有效长度
    3. 只追加有效的K/V到cache

    Example:
        >>> cache = VariableLengthCache()
        >>> # 第一次forward: 所有样本长度1
        >>> attention_mask = torch.ones(3, 1)  # [bs=3, seq=1]
        >>> outputs = model(input_ids, attention_mask=attention_mask,
        ...                 past_key_values=cache, use_cache=True)
        >>>
        >>> # 第二次forward: 样本1新增3个, 样本2新增1个, 样本3新增1个
        >>> # input_ids = [[X,Y,Z], [pad,pad,P], [pad,pad,Q]]
        >>> attention_mask = torch.tensor([
        ...     [1,1,1,1],  # 样本1: 4个有效token (之前1个+新增3个)
        ...     [1,1],      # 样本2: 2个有效token (之前1个+新增1个)
        ...     [1,1]       # 样本3: 2个有效token (之前1个+新增1个)
        ... ])
        >>> # Cache会自动识别每个样本新增的长度: [3, 1, 1]
    """

    def __init__(self) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # 记录每个样本上一次的有效长度
        self._prev_valid_lengths: Optional[torch.Tensor] = None

    def __len__(self):
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新cache,自动根据attention_mask推断每个样本的有效新增长度

        Args:
            key_states: [batch_size, num_heads, new_seq_len, head_dim]
            value_states: [batch_size, num_heads, new_seq_len, head_dim]
            layer_idx: 层索引
            cache_kwargs: {
                'attention_mask': [batch_size, total_seq_len] 完整的attention mask
                'cache_position': [new_seq_len] 新token的位置
            }
        """
        if cache_kwargs is None:
            cache_kwargs = {}

        attention_mask = cache_kwargs.get('attention_mask', None)
        batch_size = key_states.shape[0]
        new_seq_len = key_states.shape[2]

        # 计算每个样本的有效新增长度
        if attention_mask is not None and self._prev_valid_lengths is not None:
            # 计算当前每个样本的总有效长度
            current_valid_lengths = attention_mask.sum(dim=1).long()  # [batch_size]

            # 新增长度 = 当前总长度 - 上一次总长度
            valid_new_lengths = current_valid_lengths - self._prev_valid_lengths
            valid_new_lengths = torch.clamp(valid_new_lengths, min=0)  # 确保非负

            # 更新prev_valid_lengths
            if layer_idx == 0:
                self._prev_valid_lengths = current_valid_lengths
        elif attention_mask is not None:
            # 第一次调用,初始化
            current_valid_lengths = attention_mask.sum(dim=1).long()
            valid_new_lengths = current_valid_lengths  # 第一次全部都是新的

            if layer_idx == 0:
                self._prev_valid_lengths = current_valid_lengths
        else:
            # 没有attention_mask,默认所有token都有效
            valid_new_lengths = torch.full(
                (batch_size,),
                new_seq_len,
                dtype=torch.long,
                device=key_states.device
            )

        # 初始化cache
        if len(self.key_cache) <= layer_idx:
            # 第一次调用
            initial_keys = []
            initial_values = []

            for i in range(batch_size):
                valid_len = valid_new_lengths[i].item()
                if valid_len > 0:
                    # 从右边取valid_len个(左padding)
                    k = key_states[i:i+1, :, -valid_len:, :]
                    v = value_states[i:i+1, :, -valid_len:, :]
                else:
                    # 空tensor
                    k = torch.empty(1, key_states.shape[1], 0, key_states.shape[3],
                                   device=key_states.device, dtype=key_states.dtype)
                    v = torch.empty(1, value_states.shape[1], 0, value_states.shape[3],
                                   device=value_states.device, dtype=value_states.dtype)
                initial_keys.append(k)
                initial_values.append(v)

            # Padding到相同长度
            max_seq_len = max(k.shape[2] for k in initial_keys)
            if max_seq_len == 0:
                max_seq_len = 1  # 至少1

            padded_keys = []
            padded_values = []

            for k, v in zip(initial_keys, initial_values):
                if k.shape[2] < max_seq_len:
                    pad_len = max_seq_len - k.shape[2]
                    k = torch.cat([
                        torch.zeros(1, k.shape[1], pad_len, k.shape[3],
                                   device=k.device, dtype=k.dtype),
                        k
                    ], dim=2)
                    v = torch.cat([
                        torch.zeros(1, v.shape[1], pad_len, v.shape[3],
                                   device=v.device, dtype=v.dtype),
                        v
                    ], dim=2)
                padded_keys.append(k)
                padded_values.append(v)

            self.key_cache.append(torch.cat(padded_keys, dim=0))
            self.value_cache.append(torch.cat(padded_values, dim=0))

            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        # 更新现有cache
        updated_keys = []
        updated_values = []

        for i in range(batch_size):
            valid_len = valid_new_lengths[i].item()

            old_k = self.key_cache[layer_idx][i:i+1]
            old_v = self.value_cache[layer_idx][i:i+1]

            if valid_len > 0:
                # 提取有效的新K/V
                new_k = key_states[i:i+1, :, -valid_len:, :]
                new_v = value_states[i:i+1, :, -valid_len:, :]

                # 拼接
                updated_k = torch.cat([old_k, new_k], dim=2)
                updated_v = torch.cat([old_v, new_v], dim=2)
            else:
                updated_k = old_k
                updated_v = old_v

            updated_keys.append(updated_k)
            updated_values.append(updated_v)

        # 对齐到相同长度
        max_seq_len = max(k.shape[2] for k in updated_keys)
        aligned_keys = []
        aligned_values = []

        for k, v in zip(updated_keys, updated_values):
            if k.shape[2] < max_seq_len:
                pad_len = max_seq_len - k.shape[2]
                k = torch.cat([
                    torch.zeros(1, k.shape[1], pad_len, k.shape[3],
                               device=k.device, dtype=k.dtype),
                    k
                ], dim=2)
                v = torch.cat([
                    torch.zeros(1, v.shape[1], pad_len, v.shape[3],
                               device=v.device, dtype=v.dtype),
                    v
                ], dim=2)
            aligned_keys.append(k)
            aligned_values.append(v)

        self.key_cache[layer_idx] = torch.cat(aligned_keys, dim=0)
        self.value_cache[layer_idx] = torch.cat(aligned_values, dim=0)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """返回cache的最大序列长度"""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_per_sample_seq_lengths(self) -> torch.Tensor:
        """返回每个样本的实际序列长度"""
        if self._prev_valid_lengths is None:
            return torch.zeros(1, dtype=torch.long)
        return self._prev_valid_lengths.clone()

    def reset(self):
        """清空cache"""
        self.key_cache = []
        self.value_cache = []
        self._prev_valid_lengths = None

import re
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import EsmTokenizer, GenerationConfig

from utils.init_utils import construct_class_by_name
from utils.metrics import Scalar

from .BaseModel import BaseModel
from .Qwen3.configuration_domainconditioning_qwen3 import Qwen3Config
from .Qwen3.modeling_domainconditioning_qwen3 import Qwen3CAForCausalLM
from .ModifiedESM.modeling_modified_faesm import EsmModel as EsmModel
from .RAGQwen3CAwDomainConditioning import RAGQwen3CAwDomainConditioning
## 简化版RAG模型 - 仅支持batch_size=1的推理
## 用于对照测试,使用简单的KV cache,domain piece展开时清空cache
class RAGQwen3CAwDomainConditioningSimple(RAGQwen3CAwDomainConditioning):
    def __init__(
        self,
        qwen3_type="Qwen/Qwen3-100M",
        esm_type="facebook/esm2_t12_35M_UR50D",
        phrase_encoder_type="facebook/esm2_t12_35M_UR50D",
        criterion_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        


    def generate_simple(
        self,
        domain_ids: torch.Tensor,
        domain_masks: torch.Tensor,
        num_domains_per_protein: torch.Tensor,
        domain_pieces_ids: torch.Tensor,
        domain_pieces_masks: torch.Tensor,
        num_domain_pieces_per_protein: torch.Tensor,
        generation_config: GenerationConfig,
        use_cache: bool = True,
        verbose=True,
        **kwargs,
    ):
        """
        简化版生成方法 - 仅支持batch_size=1
        当domain piece展开时清空cache,使用简单的KV cache机制

        Args:
            domain_ids: domain序列的token ids
            domain_masks: domain序列的attention mask
            num_domains_per_protein: 每个蛋白质的domain数量
            domain_pieces_ids: domain pieces的token ids [num_pieces, seq_len]
            domain_pieces_masks: domain pieces的attention mask [num_pieces, seq_len]
            num_domain_pieces_per_protein: 每个蛋白质的domain piece数量
            generation_config: 生成配置
            use_cache: 是否使用KV cache加速生成
        Returns:
            生成的序列字典
        """
        batch_size = len(num_domains_per_protein)
        assert batch_size == 1, "简化版仅支持batch_size=1"

        # 1. 编码domain特征（用于交叉注意力）
        encoder_out, encoder_mask = self.forward_encoder(
            domain_ids, domain_masks, num_domains_per_protein
        )

        # 2. 编码domain pieces（用于扩展词表）
        phrase_out = self.phrase_feats_projector(
            self.phrase_encoder(
                input_ids=domain_pieces_ids,
                attention_mask=domain_pieces_masks,
                return_dict=True,
            ).pooler_output
        )

        # 3. 初始化输入
        input_ids = self.initialize_output_tokens(batch_size)  # [1, 1]

        generation_config.bos_token_id = self.tokenizer.cls_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        max_length = generation_config.max_length if generation_config.max_length else 512
        vocab_size = self.qwen3.config.vocab_size

        # 扩展lm_head权重
        lm_head_weights = self.qwen3.lm_head.weight
        extended_lm_head_weights = torch.cat([lm_head_weights, phrase_out], dim=0)
        total_vocab_size = extended_lm_head_weights.size(0)

        # domain_pieces_ids: [num_pieces, seq_len]
        # 将domain_pieces_ids解码为token列表,用于domain piece展开
        num_pieces = domain_pieces_ids.shape[0]
        domain_pieces_tokens = []
        for i in range(num_pieces):
            # 提取非padding的token
            piece_ids = domain_pieces_ids[i]  # [seq_len]
            piece_mask = domain_pieces_masks[i]  # [seq_len]

            # 只保留有效的token (mask=1的部分)
            valid_tokens = piece_ids[piece_mask.bool()].tolist()

            # 移除特殊token (cls, eos, pad)
            valid_tokens = [
                t for t in valid_tokens
                if t not in [self.tokenizer.cls_token_id,
                            self.tokenizer.eos_token_id,
                            self.tokenizer.pad_token_id]
            ]

            domain_pieces_tokens.append(valid_tokens)

        # 初始化
        finished = False
        step = 0
        past_key_values = None

        # Domain piece索引范围
        domain_start_in_vocab = vocab_size
        domain_end_in_vocab = vocab_size + num_domain_pieces_per_protein[0].item()

        while step < max_length and not finished:
            # 计算cache_position
            if past_key_values is not None and len(past_key_values) > 0:
                # 有cache: 只处理最后一个token
                past_seen_tokens = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + 1,
                    device=input_ids.device
                )
                # 只取最后一个token作为输入
                current_input_ids = input_ids[:, -1:]
            else:
                # 第一次调用或cache被清空: 处理整个序列
                seq_len = input_ids.shape[1]
                cache_position = torch.arange(seq_len, device=input_ids.device)
                current_input_ids = input_ids

            # Forward
            outputs = self.qwen3(
                input_ids=current_input_ids,
                encoder_hidden_states=encoder_out,
                encoder_attention_mask=encoder_mask,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
                cache_position=cache_position,
                output_hidden_states=True,
                return_dict=True,
            )

            # 更新cache
            if use_cache:
                past_key_values = outputs.past_key_values

            # 获取最后一个token的hidden state
            hidden_states = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_size]

            # 计算扩展词表的logits
            logits = hidden_states @ extended_lm_head_weights.T  # [1, total_vocab_size]

            # Mask掉不属于当前样本的domain pieces
            mask_value = -1e10
            if domain_start_in_vocab > vocab_size:
                logits[0, vocab_size:domain_start_in_vocab] = mask_value
            if domain_end_in_vocab < total_vocab_size:
                logits[0, domain_end_in_vocab:] = mask_value

            # 采样
            if generation_config.do_sample:
                temperature = generation_config.temperature if generation_config.temperature else 1.0
                logits = logits / temperature

                if generation_config.top_k and generation_config.top_k > 0:
                    top_k = min(generation_config.top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = mask_value

                if generation_config.top_p and generation_config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = mask_value

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token_id = next_token.item()

            # 检查是否结束
            if next_token_id == self.tokenizer.eos_token_id:
                finished = True
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                step += 1
                continue

            # 检查是否是domain piece
            if next_token_id >= vocab_size:
                # Domain piece展开
                relative_idx = next_token_id - vocab_size

                if 0 <= relative_idx < len(domain_pieces_tokens):
                    domain_piece_token_seq = domain_pieces_tokens[relative_idx]
                    domain_piece_tensor = torch.tensor(
                        domain_piece_token_seq,
                        device=input_ids.device
                    ).unsqueeze(0)

                    # 拼接展开的tokens
                    input_ids = torch.cat([input_ids, domain_piece_tensor], dim=1)

                    # 清空cache (关键: domain piece展开时清空cache)
                    if use_cache:
                        past_key_values = None

                    step += len(domain_piece_token_seq)
                else:
                    # 索引越界,跳过
                    step += 1
            else:
                # 普通token,直接添加
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                step += 1

            # 检查是否超过最大长度
            if input_ids.size(1) >= max_length:
                break

        # 解码序列
        tokens = input_ids[0].cpu().tolist()
        # 移除特殊tokens
        tokens = [t for t in tokens if t != self.tokenizer.cls_token_id]
        if self.tokenizer.eos_token_id in tokens:
            eos_idx = tokens.index(self.tokenizer.eos_token_id)
            tokens = tokens[:eos_idx]
        seq = self.tokenizer.decode(tokens)

        return {
            "output_seqs": self.clean_and_format_seq([seq]),
        }

    # 添加generate方法作为generate_simple的别名,以兼容测试脚本
    def generate(self, *args, **kwargs):
        """
        兼容测试脚本的generate方法
        直接调用generate_simple
        """
        return self.generate_simple(*args, **kwargs)

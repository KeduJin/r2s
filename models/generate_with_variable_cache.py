"""
使用VariableLengthCache的生成方法
支持batch内不同样本展开不同长度的domain piece,通过左padding对齐,不清空cache
"""

import torch
import torch.nn.functional as F
from transformers import GenerationConfig


def generate_with_variable_length_cache(
    self,
    domain_ids: torch.Tensor,
    domain_masks: torch.Tensor,
    num_domains_per_protein: torch.Tensor,
    domain_pieces_ids: torch.Tensor,
    domain_pieces_masks: torch.Tensor,
    num_domain_pieces_per_protein: torch.Tensor,
    domain_pieces_seqs: list[list[str]],
    generation_config: GenerationConfig,
    use_cache: bool = True,
    verbose=True,
    **kwargs,
):
    """
    使用VariableLengthCache的生成方法
    当domain piece被展开时,通过左padding对齐batch,不清空cache

    核心思想:
    - 样本1预测domain_piece展开为[X,Y,Z] (3个token)
    - 样本2预测普通token P (1个token)
    - 对齐: 样本1 [X,Y,Z], 样本2 [pad,pad,P]
    - Cache只追加有效token: 样本1追加3个, 样本2追加1个
    """
    from .VariableLengthCache import VariableLengthCache

    batch_size = len(num_domains_per_protein)

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
    input_ids = self.initialize_output_tokens(batch_size)  # [bs, 1]

    generation_config.bos_token_id = self.tokenizer.cls_token_id
    generation_config.eos_token_id = self.tokenizer.eos_token_id
    generation_config.pad_token_id = self.tokenizer.pad_token_id

    max_length = generation_config.max_length if generation_config.max_length else 512
    vocab_size = self.qwen3.config.vocab_size

    # 扩展lm_head权重
    lm_head_weights = self.qwen3.lm_head.weight
    extended_lm_head_weights = torch.cat([lm_head_weights, phrase_out], dim=0)
    total_vocab_size = extended_lm_head_weights.size(0)

    # 预计算domain起始索引
    if len(num_domain_pieces_per_protein) > 1:
        domain_start_indices = torch.cumsum(
            torch.cat([
                torch.tensor([0], device=num_domain_pieces_per_protein.device,
                            dtype=num_domain_pieces_per_protein.dtype),
                num_domain_pieces_per_protein[:-1]
            ]),
            dim=0,
        )
    else:
        domain_start_indices = torch.tensor([0], device=num_domain_pieces_per_protein.device,
                                           dtype=num_domain_pieces_per_protein.dtype)

    # 预先tokenize domain pieces
    domain_pieces_tokens = []
    for i in range(batch_size):
        sample_tokens = []
        for seq in domain_pieces_seqs[i]:
            tokens = self.tokenizer.encode(seq, add_special_tokens=False)
            sample_tokens.append(tokens)
        domain_pieces_tokens.append(sample_tokens)

    # 初始化
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    step = 0
    attention_mask = torch.ones_like(input_ids)

    # 使用自定义的VariableLengthCache
    past_key_values = VariableLengthCache() if use_cache else None

    while step < max_length and not finished.all():
        # 计算cache_position
        if past_key_values is not None and len(past_key_values) > 0:
            # 有cache: 从cache长度开始
            past_seen_tokens = past_key_values.get_seq_length()
            current_seq_len = input_ids.shape[1]
            num_new_tokens = current_seq_len - past_seen_tokens

            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + num_new_tokens,
                device=input_ids.device
            )
        else:
            # 第一次调用: cache_position = [0, 1, ..., seq_len-1]
            seq_len = input_ids.shape[1]
            cache_position = torch.arange(seq_len, device=input_ids.device)

        # Forward - 传入attention_mask以便cache能够推断有效长度
        outputs = self.qwen3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_out,
            encoder_attention_mask=encoder_mask,
            past_key_values=past_key_values if use_cache else None,
            use_cache=use_cache,
            cache_position=cache_position,
            output_hidden_states=True,
            return_dict=True,
        )

        # 获取最后一个有效位置的hidden state
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

        # 对于每个样本,取最后一个有效token的hidden state
        # 由于是左padding,最后一个有效token总是在最右边
        last_hidden_states = hidden_states[:, -1, :]  # [batch_size, hidden_size]

        # 计算扩展词表的logits
        logits = last_hidden_states @ extended_lm_head_weights.T  # [batch_size, total_vocab_size]

        # Mask掉不属于当前样本的domain pieces
        mask_value = -1e10
        for i in range(batch_size):
            if finished[i]:
                continue

            start_idx = domain_start_indices[i].item()
            end_idx = start_idx + num_domain_pieces_per_protein[i].item()

            domain_start_in_vocab = vocab_size + start_idx
            domain_end_in_vocab = vocab_size + end_idx

            if domain_start_in_vocab > vocab_size:
                logits[i, vocab_size:domain_start_in_vocab] = mask_value
            if domain_end_in_vocab < total_vocab_size:
                logits[i, domain_end_in_vocab:] = mask_value

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
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(logits, dim=-1)

        # 处理domain piece展开,计算每个样本新增的token数
        new_token_sequences = []
        max_new_length = 1

        for i in range(batch_size):
            if finished[i]:
                new_token_sequences.append([])
                continue

            next_token = next_tokens[i].item()

            if next_token == self.tokenizer.eos_token_id:
                finished[i] = True
                new_token_sequences.append([next_token])
            elif next_token >= vocab_size:
                # Domain piece展开
                domain_piece_idx = next_token - vocab_size
                sample_start_idx = domain_start_indices[i].item()
                relative_idx = domain_piece_idx - sample_start_idx

                if 0 <= relative_idx < len(domain_pieces_tokens[i]):
                    domain_piece_token_seq = domain_pieces_tokens[i][relative_idx]
                    new_token_sequences.append(domain_piece_token_seq)
                    max_new_length = max(max_new_length, len(domain_piece_token_seq))
                else:
                    new_token_sequences.append([])
            else:
                # 普通token
                new_token_sequences.append([next_token])

        # 左padding对齐所有样本
        aligned_sequences = []

        for i in range(batch_size):
            new_tokens = new_token_sequences[i]
            valid_len = len(new_tokens)

            if valid_len < max_new_length:
                # 左padding
                padding = [self.tokenizer.pad_token_id] * (max_new_length - valid_len)
                aligned_tokens = padding + new_tokens
            else:
                aligned_tokens = new_tokens

            aligned_sequences.append(aligned_tokens)

        # 拼接到input_ids
        new_input_ids = []
        for i in range(batch_size):
            new_tokens = torch.tensor(aligned_sequences[i], device=input_ids.device)
            updated_seq = torch.cat([input_ids[i], new_tokens])
            new_input_ids.append(updated_seq)

        # 左padding整个batch
        max_len = max(seq.size(0) for seq in new_input_ids)
        padded_input_ids = []
        padded_attention_mask = []

        for i, seq in enumerate(new_input_ids):
            seq_len = seq.size(0)
            if seq_len < max_len:
                pad_len = max_len - seq_len
                seq = torch.cat([
                    torch.full((pad_len,), self.tokenizer.pad_token_id, device=seq.device),
                    seq
                ])

            # attention mask: 1表示有效token, 0表示padding
            # 计算有效长度: 从右边数,直到遇到第一个非pad token
            mask = (seq != self.tokenizer.pad_token_id).long()

            padded_input_ids.append(seq)
            padded_attention_mask.append(mask)

        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)

        step += max_new_length

        if input_ids.size(1) >= max_length:
            break

    # 解码序列
    sequences = []
    for i in range(batch_size):
        tokens = input_ids[i].cpu().tolist()
        # 移除padding和特殊tokens
        tokens = [t for t in tokens if t != self.tokenizer.pad_token_id and t != self.tokenizer.cls_token_id]
        if self.tokenizer.eos_token_id in tokens:
            eos_idx = tokens.index(self.tokenizer.eos_token_id)
            tokens = tokens[:eos_idx]
        seq = self.tokenizer.decode(tokens)
        sequences.append(seq)

    return {
        "output_seqs": self.clean_and_format_seq(sequences),
    }

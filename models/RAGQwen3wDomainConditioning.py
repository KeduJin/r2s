import re
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import EsmTokenizer, GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from utils.init_utils import construct_class_by_name
from utils.metrics import Scalar

from .BaseModel import BaseModel
from .ModifiedESM.modeling_modified_faesm import EsmModel as EsmModel


class RAGQwen3wDomainConditioning(BaseModel):
    """
    Decoder-only RAG模型
    - Domain pieces作为prefix拼接在序列前: <domain_cls>domain1<domain_eos>...<cls>seq<eos>
    - 使用phrase encoder扩展词表,将domain piece token展开为完整序列
    - 纯decoder架构,无cross-attention
    """
    def __init__(
        self,
        qwen3_type="Qwen/Qwen3-100M",
        phrase_encoder_type="facebook/esm2_t12_35M_UR50D",
        criterion_kwargs: Optional[dict] = None,
        dynamic_domain_weight: bool = False,
        domain_weight: float = 1.0,
        non_domain_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger.info(
            f"{self.__class__.__name__} initialized with qwen3_type: {qwen3_type}, phrase_encoder_type: {phrase_encoder_type}"
        )
        self.dynamic_domain_weight = dynamic_domain_weight
        self.domain_weight = domain_weight
        self.non_domain_weight = non_domain_weight
        # Tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained("airkingbd/dplm_150m")
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<domain_cls>", "<domain_eos>"]}
        )

        # Decoder模型 (纯decoder,无cross-attention)
        if qwen3_type == "Qwen/Qwen3-100M":
            config = Qwen3Config(
                architectures=["Qwen3ForCausalLM"],
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=self.tokenizer.cls_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                head_dim=64,
                hidden_act="silu",
                hidden_size=768,
                initializer_range=0.02,
                intermediate_size=2304,
                max_position_embeddings=40960,
                max_window_layers=16,
                model_type="qwen3",
                num_attention_heads=12,
                num_hidden_layers=16,
                num_key_value_heads=6,
                rms_norm_eps=1e-06,
                rope_scaling=None,
                rope_theta=1000000,
                sliding_window=None,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                use_cache=True,
                use_sliding_window=False,
                vocab_size=len(self.tokenizer.get_vocab()),
            )
            self.qwen3 = Qwen3ForCausalLM(config)
        else:
            qwen3_config = Qwen3Config.from_pretrained(qwen3_type, vocab_size=len(self.tokenizer.get_vocab()))
            self.qwen3 = Qwen3ForCausalLM(qwen3_config)

        self.qwen3.gradient_checkpointing_enable()

        # Phrase encoder (用于扩展词表)
        self.phrase_encoder = EsmModel.from_pretrained(phrase_encoder_type)
        self.phrase_encoder.encoder.gradient_checkpointing = True
        self.phrase_encoder.gradient_checkpointing_enable()
        self.phrase_encoder.contact_head.regression.weight.requires_grad = False
        self.phrase_encoder.contact_head.regression.bias.requires_grad = False

        self.phrase_feats_projector = nn.Linear(
            self.phrase_encoder.config.hidden_size,
            self.qwen3.config.hidden_size
        )

        if criterion_kwargs is not None:
            self.criterion = construct_class_by_name(
                **criterion_kwargs, logger=self.logger
            )
        else:
            self.criterion = None

    def set_objective_and_metrics(self, stage: str = "train", experiment=None):
        if stage == "train":
            train_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "domain_loss": Scalar(dist_sync_on_step=True),
                "non_domain_loss": Scalar(dist_sync_on_step=True),
                "domain_weight": Scalar(dist_sync_on_step=True),
                "non_domain_weight": Scalar(dist_sync_on_step=True),
                "unweighted_loss": Scalar(dist_sync_on_step=True),
                "cls_head_l2_norm_mean": Scalar(dist_sync_on_step=True),
                "cls_head_l2_norm_std": Scalar(dist_sync_on_step=True),
                "domain_head_l2_norm_mean": Scalar(dist_sync_on_step=True),
                "domain_head_l2_norm_std": Scalar(dist_sync_on_step=True),
            }
            val_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "domain_loss": Scalar(dist_sync_on_step=True),
                "non_domain_loss": Scalar(dist_sync_on_step=True),
                "domain_weight": Scalar(dist_sync_on_step=True),
                "non_domain_weight": Scalar(dist_sync_on_step=True),
                "unweighted_loss": Scalar(dist_sync_on_step=True),
                "cls_head_l2_norm_mean": Scalar(dist_sync_on_step=True),
                "cls_head_l2_norm_std": Scalar(dist_sync_on_step=True),
                "domain_head_l2_norm_mean": Scalar(dist_sync_on_step=True),
                "domain_head_l2_norm_std": Scalar(dist_sync_on_step=True),
            }
            train_metrics = {k: v.to(experiment.device) for k, v in train_metrics.items()}
            val_metrics = {k: v.to(experiment.device) for k, v in val_metrics.items()}
            test_metrics = None
        else:
            train_metrics = None
            val_metrics = None
            test_metrics = {
                "softlcs_score": Scalar(dist_sync_on_step=True),
                "ref_softlcs_score": Scalar(dist_sync_on_step=True),
                "gt_seq_softlcs_score": Scalar(dist_sync_on_step=True),
            }
            test_metrics = {k: v.to(experiment.device) for k, v in test_metrics.items()}
        experiment.metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

    def forward(
        self,
        input_ids: torch.Tensor,  # [bs, seq_len] 已包含domain prefix
        labels: torch.Tensor,  # [bs, seq_len] 对应的标签
        domain_positions: list[list[tuple[int, int]]],
        domain_pieces_ids: torch.Tensor,
        domain_pieces_masks: torch.Tensor,
        num_domain_pieces_per_protein: torch.Tensor,
        **kwargs,
    ):
        """
        训练forward
        input_ids: [bs, seq_len] 格式为 <domain_cls>domain1<domain_eos>...<cls>seq<eos>
        labels: [bs, seq_len] domain prompt部分已设为-100
        domain_positions: 蛋白质序列中domain的位置(相对于<cls>之后)
        """
        # 1. 编码domain pieces (用于扩展词表)
        phrase_out = self.phrase_feats_projector(
            self.phrase_encoder(
                input_ids=domain_pieces_ids,
                attention_mask=domain_pieces_masks,
                return_dict=True,
            ).pooler_output
        )  # [num_domain_pieces, hidden_size]

        # 2. Forward decoder (无cross-attention)
        out_dict = self.qwen3(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        # 3. 创建RAG target
        target = labels.clone()
        target = target.masked_fill(target == self.tokenizer.pad_token_id, -100)
        
        target, domain_weight_matrix = self.create_rag_target(
            target, input_ids, domain_positions, num_domain_pieces_per_protein,
            self.dynamic_domain_weight
        )

        ## 3.1 check and mask <unk> tokens in target
        unk_mask = (target == self.tokenizer.unk_token_id)
        if unk_mask.any():
            num_unk = unk_mask.sum().item()
            print(f"\n{'='*80}")
            print(f"WARNING: Found {num_unk} <unk> tokens in target! Masking them with -100.")
            print(f"{'='*80}\n")
            target = target.masked_fill(unk_mask, -100)

        # 4. 计算loss
        loss_dict = self.rag_loss(
            out_dict, phrase_out, target,
            num_domain_pieces_per_protein, domain_weight_matrix
        )

        return {
            "target": target,
            "loss": loss_dict["loss"],
            "domain_loss": loss_dict["domain_loss"],
            "non_domain_loss": loss_dict["non_domain_loss"],
            "domain_weight": self.domain_weight,
            "non_domain_weight": self.non_domain_weight,
            "cls_head_l2_norm_mean": loss_dict["cls_head_l2_norm_mean"],
            "cls_head_l2_norm_std": loss_dict["cls_head_l2_norm_std"],
            "domain_head_l2_norm_mean": loss_dict["domain_head_l2_norm_mean"],
            "domain_head_l2_norm_std": loss_dict["domain_head_l2_norm_std"],
        }

    def create_rag_target(
        self, target, input_ids, domain_positions, num_domain_pieces_per_protein,
        dynamic_domain_weight=False
    ):
        """
        根据domain_positions修改target,将domain piece的第一个位置设为domain piece token id

        注意: domain_positions是相对于<cls>之后的位置,需要找到<cls>在input_ids中的位置
        """
        vocab_size = self.qwen3.config.vocab_size
        batch_size = target.size(0)
        seq_len = target.size(1)

        domain_weight_matrix = torch.ones_like(target, dtype=torch.float, device=target.device)

        # 计算每个样本的domain在phrase_out中的起始索引
        if len(num_domain_pieces_per_protein) > 1:
            domain_start_indices = torch.cumsum(
                torch.cat([
                    torch.tensor([0], device=num_domain_pieces_per_protein.device,
                               dtype=num_domain_pieces_per_protein.dtype),
                    num_domain_pieces_per_protein[:-1],
                ]),
                dim=0,
            )
        else:
            domain_start_indices = torch.tensor(
                [0], device=num_domain_pieces_per_protein.device,
                dtype=num_domain_pieces_per_protein.dtype
            )

        cls_token_id = self.tokenizer.cls_token_id

        for i, positions in enumerate(domain_positions):
            # 找到<cls>在input_ids中的位置
            cls_indices = (input_ids[i] == cls_token_id).nonzero(as_tuple=True)[0]
            if cls_indices.numel() == 0:
                # 找不到<cls>,跳过
                continue
            seq_start_offset = cls_indices[0].item()

            sample_domain_start_idx = domain_start_indices[i].item()

            for domain_idx_in_sample, (start, end) in enumerate(positions):
                domain_global_idx = sample_domain_start_idx + domain_idx_in_sample
                domain_token_id = vocab_size + domain_global_idx

                # domain_positions是相对于<cls>之后的位置,需要加上seq_start_offset
                # 再+1是因为要跳过<cls>本身
                domain_start_pos = seq_start_offset + start + 1
                domain_end_pos = seq_start_offset + end + 1

                if domain_start_pos >= target.size(1):
                    continue

                # 将domain的第一个位置设为domain piece token id
                target[i, domain_start_pos] = domain_token_id

                # 将domain的其余位置设为-100
                if domain_start_pos + 1 < target.size(1):
                    actual_end_pos = min(domain_end_pos, target.size(1))
                    target[i, domain_start_pos + 1 : actual_end_pos] = -100

                # 动态domain weighting
                if dynamic_domain_weight:
                    domain_length = end - start
                    domain_weight_matrix[i, domain_start_pos] = float(domain_length) * self.domain_weight
                else:
                    domain_weight_matrix[i, domain_start_pos] = self.domain_weight

        return target, domain_weight_matrix

    def rag_loss(self, out_dict, phrase_out, target,
                 num_domain_pieces_per_protein, domain_weight_matrix):
        """
        计算RAG loss,使用扩展词表
        """
        hidden_states = out_dict["hidden_states"][-1]
        lm_head_weights = self.qwen3.lm_head.weight
        vocab_size = lm_head_weights.size(0)
        batch_size = hidden_states.size(0)

        # 扩展词表
        extended_lm_head_weights = torch.cat([lm_head_weights, phrase_out], dim=0)
        logits = hidden_states @ extended_lm_head_weights.T
        total_vocab_size = extended_lm_head_weights.size(0)

        # 计算domain起始索引
        if len(num_domain_pieces_per_protein) > 1:
            domain_start_indices = torch.cumsum(
                torch.cat([
                    torch.tensor([0], device=num_domain_pieces_per_protein.device,
                               dtype=num_domain_pieces_per_protein.dtype),
                    num_domain_pieces_per_protein[:-1],
                ]),
                dim=0,
            )
        else:
            domain_start_indices = torch.tensor(
                [0], device=num_domain_pieces_per_protein.device,
                dtype=num_domain_pieces_per_protein.dtype
            )

        # Mask掉不属于当前样本的domain pieces
        mask_value = -1e10
        for i in range(batch_size):
            start_idx = domain_start_indices[i].item()
            end_idx = start_idx + num_domain_pieces_per_protein[i].item()

            domain_start_in_vocab = vocab_size + start_idx
            domain_end_in_vocab = vocab_size + end_idx

            if domain_start_in_vocab > vocab_size:
                logits[i, :, vocab_size:domain_start_in_vocab] = mask_value
            if domain_end_in_vocab < total_vocab_size:
                logits[i, :, domain_end_in_vocab:] = mask_value

        # Shift logits and targets
        shifted_logits = logits[:, :-1, :]
        flat_shifted_logits = shifted_logits.reshape(-1, shifted_logits.size(-1))
        shifted_target = target[:, 1:]
        flat_shifted_target = shifted_target.reshape(-1)

        shifted_domain_weight_matrix = domain_weight_matrix[:, 1:]
        flat_shifted_domain_weight_matrix = shifted_domain_weight_matrix.reshape(-1)

        # 计算mask
        domain_mask = (
            (flat_shifted_target >= vocab_size)
            & (flat_shifted_target < total_vocab_size)
            & (flat_shifted_target != -100)
        )
        non_domain_mask = (flat_shifted_target < vocab_size) & (
            flat_shifted_target != -100
        )

        # 计算loss
        loss = F.cross_entropy(
            flat_shifted_logits,
            flat_shifted_target,
            ignore_index=-100,
            reduction="none",
        )

        loss = loss * flat_shifted_domain_weight_matrix

        # Domain loss
        domain_loss = None
        if domain_mask.any():
            domain_loss = loss[domain_mask].mean()

        # Non-domain loss
        non_domain_loss = None
        if non_domain_mask.any():
            non_domain_loss = loss[non_domain_mask].mean()

        # Total loss
        total_token = ((flat_shifted_target != -100).float() * flat_shifted_domain_weight_matrix)
        total_loss = loss.sum() / total_token.sum()

        return {
            "loss": total_loss,
            "domain_loss": domain_loss if domain_loss is not None
                          else torch.tensor(0.0, device=total_loss.device),
            "non_domain_loss": non_domain_loss if non_domain_loss is not None
                              else torch.tensor(0.0, device=total_loss.device),
            "cls_head_l2_norm_mean": lm_head_weights.norm(p=2, dim=1).mean(),
            "cls_head_l2_norm_std": lm_head_weights.norm(p=2, dim=1).std(),
            "domain_head_l2_norm_mean": phrase_out.norm(p=2, dim=1).mean(),
            "domain_head_l2_norm_std": phrase_out.norm(p=2, dim=1).std(),
        }

    def clean_and_format_seq(self, seq: list[str]):
        cleaned_data = []
        for item in seq:
            processed_string = re.sub(r"<cls>", "", item)
            processed_string = re.sub(r"<eos>", "", processed_string)
            processed_string = re.sub(r"<unk>", "", processed_string)
            processed_string = re.sub(r"<domain_cls>", "", processed_string)
            processed_string = re.sub(r"<domain_eos>", "", processed_string)
            processed_string = processed_string.replace(" ", "")
            cleaned_data.append(processed_string)
        return cleaned_data

    def prepare_inputs_for_generation(
        self,
        domain: list[list[str]],
        **kwargs
    ):
        """
        准备生成的输入
        Args:
            domain: 每个蛋白质的domain列表 [['domain1', 'domain2'], ['domain3']]
        Returns:
            input_ids: [bs, seq_len] 格式为 <domain_cls>domain1<domain_eos>...<cls>
        """
        prompts = []
        domain_cls_token = self.tokenizer.additional_special_tokens[0]  # <domain_cls>
        domain_eos_token = self.tokenizer.additional_special_tokens[1]  # <domain_eos>
        cls_token = self.tokenizer.cls_token

        for domains_per_protein in domain:
            domain_prompt_parts = []
            for d in domains_per_protein:
                domain_prompt_parts.append(f"{domain_cls_token}{d}{domain_eos_token}")
            domain_prompt = "".join(domain_prompt_parts)
            prompt = f"{domain_prompt}{cls_token}"
            prompts.append(prompt)

        # Tokenize prompts
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).to(self.qwen3.device)

        self.tokenizer.padding_side = original_padding_side

        return inputs

    def generate(
        self,
        domain: list[list[str]],
        domain_pieces_ids: torch.Tensor,
        domain_pieces_masks: torch.Tensor,
        num_domain_pieces_per_protein: torch.Tensor,
        generation_config: GenerationConfig = None,
        use_cache: bool = True,
        verbose=True,
        **kwargs
    ):
        # 根据配置选择生成方法
        generation_method = kwargs.get('generation_method', 'generate_simple')

        if generation_method == 'generate_simple_with_unique_domain_pieces':
            return self.generate_simple_with_unique_domain_pieces(
                domain, domain_pieces_ids, domain_pieces_masks,
                num_domain_pieces_per_protein, generation_config,
                use_cache, verbose, **kwargs
            )
        else:
            return self.generate_simple(
                domain, domain_pieces_ids, domain_pieces_masks,
                num_domain_pieces_per_protein, generation_config,
                use_cache, verbose, **kwargs
            )
        """
        生成蛋白质序列,支持RAG扩展词表
        当预测到domain piece token时,将其展开为完整的氨基酸序列

        Args:
            domain: 每个蛋白质的domain列表
            domain_pieces_ids: domain pieces的token ids [num_pieces, seq_len]
            domain_pieces_masks: domain pieces的attention mask [num_pieces, seq_len]
            num_domain_pieces_per_protein: 每个蛋白质的domain piece数量
            generation_config: 生成配置
            use_cache: 是否使用KV cache
        """
        if generation_config is None:
            generation_config = self.default_generation_config

        batch_size = len(domain)

        # 1. 编码domain pieces (用于扩展词表)
        phrase_out = self.phrase_feats_projector(
            self.phrase_encoder(
                input_ids=domain_pieces_ids,
                attention_mask=domain_pieces_masks,
                return_dict=True,
            ).pooler_output
        )  # [num_domain_pieces, hidden_size]

        # 2. 准备初始输入 (domain prefix)
        model_inputs = self.prepare_inputs_for_generation(domain=domain)
        input_ids = model_inputs.input_ids  # [bs, prefix_len]
        attention_mask = model_inputs.attention_mask
        prompt_lengths = attention_mask.sum(dim=1)  # 记录每个样本的prompt长度

        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.bos_token_id = self.tokenizer.cls_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id

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
            domain_start_indices = torch.tensor(
                [0], device=num_domain_pieces_per_protein.device,
                dtype=num_domain_pieces_per_protein.dtype
            )

        # 从domain_pieces_ids解码为token列表,用于domain piece展开
        domain_pieces_tokens = []
        for i in range(batch_size):
            sample_start_idx = domain_start_indices[i].item()
            sample_end_idx = sample_start_idx + num_domain_pieces_per_protein[i].item()

            sample_tokens = []
            for piece_idx in range(sample_start_idx, sample_end_idx):
                # 提取非padding的token
                piece_ids = domain_pieces_ids[piece_idx]  # [seq_len]
                piece_mask = domain_pieces_masks[piece_idx]  # [seq_len]

                # 只保留有效的token (mask=1的部分)
                valid_tokens = piece_ids[piece_mask.bool()].tolist()

                # 移除特殊token (cls, eos, pad)
                valid_tokens = [
                    t for t in valid_tokens
                    if t not in [self.tokenizer.cls_token_id,
                                self.tokenizer.eos_token_id,
                                self.tokenizer.pad_token_id]
                ]

                sample_tokens.append(valid_tokens)
            domain_pieces_tokens.append(sample_tokens)

        # 生成循环
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        step = 0
        past_key_values = None

        while step < max_length and not finished.all():
            # 计算cache_position
            if past_key_values is not None and len(past_key_values) > 0:
                past_seen_tokens = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + input_ids.shape[1],
                    device=input_ids.device
                )
            else:
                seq_len = input_ids.shape[1]
                cache_position = torch.arange(seq_len, device=input_ids.device)

            # Forward
            outputs = self.qwen3(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
                cache_position=cache_position,
                output_hidden_states=True,
                return_dict=True,
            )

            if use_cache:
                past_key_values = outputs.past_key_values

            # 获取最后一个token的hidden state
            hidden_states = outputs.hidden_states[-1][:, -1, :]  # [bs, hidden_size]

            # 计算扩展词表的logits
            logits = hidden_states @ extended_lm_head_weights.T  # [bs, total_vocab_size]

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

            # 处理每个样本的next_token
            new_input_ids = []
            domain_piece_expanded = False

            for i in range(batch_size):
                if finished[i]:
                    new_input_ids.append(input_ids[i])
                    continue

                next_token = next_tokens[i].item()

                if next_token == self.tokenizer.eos_token_id:
                    finished[i] = True
                    new_input_ids.append(torch.cat([input_ids[i], torch.tensor([next_token], device=input_ids.device)]))
                elif next_token >= vocab_size:
                    # Domain piece展开
                    domain_piece_idx = next_token - vocab_size
                    sample_start_idx = domain_start_indices[i].item()
                    relative_idx = domain_piece_idx - sample_start_idx

                    if 0 <= relative_idx < len(domain_pieces_tokens[i]):
                        domain_piece_token_seq = domain_pieces_tokens[i][relative_idx]
                        domain_piece_tensor = torch.tensor(domain_piece_token_seq, device=input_ids.device)
                        new_input_ids.append(torch.cat([input_ids[i], domain_piece_tensor]))
                        domain_piece_expanded = True
                    else:
                        new_input_ids.append(input_ids[i])
                else:
                    # 普通token
                    new_input_ids.append(torch.cat([input_ids[i], torch.tensor([next_token], device=input_ids.device)]))

            # 如果有domain piece展开,清空cache
            if domain_piece_expanded and use_cache:
                past_key_values = None

            # 左padding对齐
            max_len = max(seq.size(0) for seq in new_input_ids)
            padded_input_ids = []
            padded_attention_mask = []

            for seq in new_input_ids:
                seq_len = seq.size(0)
                if seq_len < max_len:
                    padding_len = max_len - seq_len
                    padding = torch.full((padding_len,), self.tokenizer.pad_token_id, device=seq.device)
                    padded_seq = torch.cat([padding, seq])
                    mask = torch.cat([torch.zeros(padding_len, device=seq.device), torch.ones(seq_len, device=seq.device)])
                else:
                    padded_seq = seq
                    mask = torch.ones(seq_len, device=seq.device)

                padded_input_ids.append(padded_seq)
                padded_attention_mask.append(mask)

            input_ids = torch.stack(padded_input_ids)
            attention_mask = torch.stack(padded_attention_mask)

            step += 1

            # 检查是否超过最大长度 (只计算生成的部分,不包括prompt)
            # 由于batch中每个样本的prompt长度可能不同,检查最大生成长度
            max_generated_length = max((input_ids[i].size(0) - prompt_lengths[i].item())
                                      for i in range(batch_size) if not finished[i])
            if max_generated_length >= max_length:
                break

        # 解码序列 - 只解码生成的部分,去掉prompt
        sequences = []
        for i in range(batch_size):
            prompt_len = prompt_lengths[i].item()
            tokens = input_ids[i].cpu().tolist()

            # 找到prompt结束位置 (从右往左找第一个非padding token的位置)
            # 由于使用左padding,需要先找到实际序列的起始位置
            non_pad_start = 0
            for j, token in enumerate(tokens):
                if token != self.tokenizer.pad_token_id:
                    non_pad_start = j
                    break

            # 跳过prompt部分,只保留生成的tokens
            generated_start = non_pad_start + prompt_len
            tokens = tokens[generated_start:]

            # 移除pad tokens
            tokens = [t for t in tokens if t != self.tokenizer.pad_token_id]

            if self.tokenizer.eos_token_id in tokens:
                eos_idx = tokens.index(self.tokenizer.eos_token_id)
                tokens = tokens[:eos_idx]
            seq = self.tokenizer.decode(tokens)
            sequences.append(seq)

        return {
            "output_seqs": self.clean_and_format_seq(sequences),
        }

    def generate_simple(
        self,
        domain: list[list[str]],
        domain_pieces_ids: torch.Tensor,
        domain_pieces_masks: torch.Tensor,
        num_domain_pieces_per_protein: torch.Tensor,
        generation_config: GenerationConfig = None,
        use_cache: bool = True,
        verbose=True,
        **kwargs
    ):
        """
        简化版生成方法 - 仅支持batch_size=1
        当domain piece展开时清空cache,使用简单的KV cache机制

        Args:
            domain: 每个蛋白质的domain列表 [['domain1', 'domain2']]
            domain_pieces_ids: domain pieces的token ids [num_pieces, seq_len]
            domain_pieces_masks: domain pieces的attention mask [num_pieces, seq_len]
            num_domain_pieces_per_protein: 每个蛋白质的domain piece数量
            generation_config: 生成配置
            use_cache: 是否使用KV cache加速生成
        Returns:
            生成的序列字典
        """
        batch_size = len(domain)
        assert batch_size == 1, "简化版仅支持batch_size=1"

        if generation_config is None:
            generation_config = self.default_generation_config

        # 1. 编码domain pieces（用于扩展词表）
        phrase_out = self.phrase_feats_projector(
            self.phrase_encoder(
                input_ids=domain_pieces_ids,
                attention_mask=domain_pieces_masks,
                return_dict=True,
            ).pooler_output
        )

        # 2. 准备初始输入 (domain prefix + <cls>)
        model_inputs = self.prepare_inputs_for_generation(domain=domain)
        input_ids = model_inputs.input_ids  # [1, prefix_len]
        prompt_length = input_ids.size(1)  # 记录prompt长度

        generation_config.bos_token_id = self.tokenizer.cls_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        max_length = generation_config.max_length if generation_config.max_length else 512
        vocab_size = self.qwen3.config.vocab_size

        # 扩展lm_head权重
        lm_head_weights = self.qwen3.lm_head.weight
        extended_lm_head_weights = torch.cat([lm_head_weights, phrase_out], dim=0)
        total_vocab_size = extended_lm_head_weights.size(0)

        # 从domain_pieces_ids解码为token列表,用于domain piece展开
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
                    raise ValueError("error: index out of range")
                    # 索引越界,跳过
                    step += 1
            else:
                # 普通token,直接添加
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                step += 1

            # 检查是否超过最大长度 (只计算生成的部分,不包括prompt)
            generated_length = input_ids.size(1) - prompt_length
            if generated_length >= max_length:
                break

        # 解码序列 - 只解码生成的部分,去掉prompt
        tokens = input_ids[0, prompt_length:].cpu().tolist()  # 跳过prompt部分

        # 移除特殊tokens
        tokens = [t for t in tokens if t != self.tokenizer.pad_token_id]

        if self.tokenizer.eos_token_id in tokens:
            eos_idx = tokens.index(self.tokenizer.eos_token_id)
            tokens = tokens[:eos_idx]
        seq = self.tokenizer.decode(tokens)
        return {
            "output_seqs": self.clean_and_format_seq([seq]),
        }




    def generate_simple_with_unique_domain_pieces(
        self,
        domain: list[list[str]],
        domain_pieces_ids: torch.Tensor,
        domain_pieces_masks: torch.Tensor,
        num_domain_pieces_per_protein: torch.Tensor,
        generation_config: GenerationConfig = None,
        use_cache: bool = True,
        verbose=True,
        **kwargs
    ):
        """
        简化版生成方法 - 仅支持batch_size=1
        当domain piece展开时清空cache,使用简单的KV cache机制

        Args:
            domain: 每个蛋白质的domain列表 [['domain1', 'domain2']]
            domain_pieces_ids: domain pieces的token ids [num_pieces, seq_len]
            domain_pieces_masks: domain pieces的attention mask [num_pieces, seq_len]
            num_domain_pieces_per_protein: 每个蛋白质的domain piece数量
            generation_config: 生成配置
            use_cache: 是否使用KV cache加速生成
        Returns:
            生成的序列字典
        """
        batch_size = len(domain)
        assert batch_size == 1, "简化版仅支持batch_size=1"

        if generation_config is None:
            generation_config = self.default_generation_config

        # 1. 编码domain pieces（用于扩展词表）
        phrase_out = self.phrase_feats_projector(
            self.phrase_encoder(
                input_ids=domain_pieces_ids,
                attention_mask=domain_pieces_masks,
                return_dict=True,
            ).pooler_output
        )

        # 2. 准备初始输入 (domain prefix + <cls>)
        model_inputs = self.prepare_inputs_for_generation(domain=domain)
        input_ids = model_inputs.input_ids  # [1, prefix_len]
        prompt_length = input_ids.size(1)  # 记录prompt长度

        generation_config.bos_token_id = self.tokenizer.cls_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        max_length = generation_config.max_length if generation_config.max_length else 512
        vocab_size = self.qwen3.config.vocab_size

        # 扩展lm_head权重
        lm_head_weights = self.qwen3.lm_head.weight
        extended_lm_head_weights = torch.cat([lm_head_weights, phrase_out], dim=0)
        total_vocab_size = extended_lm_head_weights.size(0)

        # 从domain_pieces_ids解码为token列表,用于domain piece展开
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

        # 记录已使用的domain pieces (用于防止重复选择)
        used_domain_pieces = set()

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

            # Mask掉已使用的domain pieces (防止重复选择)
            for used_idx in used_domain_pieces:
                logits[0, vocab_size + used_idx] = mask_value

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
                    # 记录已使用的domain piece
                    used_domain_pieces.add(relative_idx)

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
                    raise ValueError("error: index out of range")
                    # 索引越界,跳过
                    step += 1
            else:
                # 普通token,直接添加
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                step += 1

            # 检查是否超过最大长度 (只计算生成的部分,不包括prompt)
            generated_length = input_ids.size(1) - prompt_length
            if generated_length >= max_length:
                break

        # 解码序列 - 只解码生成的部分,去掉prompt
        tokens = input_ids[0, prompt_length:].cpu().tolist()  # 跳过prompt部分

        # 移除特殊tokens
        tokens = [t for t in tokens if t != self.tokenizer.pad_token_id]

        if self.tokenizer.eos_token_id in tokens:
            eos_idx = tokens.index(self.tokenizer.eos_token_id)
            tokens = tokens[:eos_idx]
        seq = self.tokenizer.decode(tokens)
        return {
            "output_seqs": self.clean_and_format_seq([seq]),
        }

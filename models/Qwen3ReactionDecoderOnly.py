import re

import torch
import torch.nn as nn
from transformers import EsmTokenizer, GenerationConfig

from utils.metrics import Scalar

from .BaseModel import BaseModel
from .Qwen3.configuration_domainconditioning_qwen3 import Qwen3Config
from .Qwen3.modeling_domainconditioning_qwen3 import Qwen3CAForCausalLM


class LatentReactionCompressor(nn.Module):
    def __init__(self, hidden_size: int, num_latents: int, num_heads: int, dropout: float):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        latents = self.latents.expand(memory.shape[0], -1, -1)
        attn_out, _ = self.cross_attn(
            query=latents,
            key=memory,
            value=memory,
            key_padding_mask=~memory_mask.bool(),
            need_weights=False,
        )
        latents = self.norm1(latents + self.dropout(attn_out))
        latents = self.norm2(latents + self.dropout(self.ffn(latents)))
        return latents


class Qwen3ReactionDecoderOnly(BaseModel):
    def __init__(
        self,
        qwen3_type="Qwen/Qwen3-100M",
        reaction_input_dim: int = 1024,
        reaction_hidden_dim: int = 768,
        reaction_num_heads: int = 8,
        reaction_dropout: float = 0.1,
        reaction_fusion_layers: int = 1,
        num_tokens_per_molecule: int = 16,
        compress_reaction: bool = True,
        esm_tokenizer_name_or_path: str = "airkingbd/dplm_150m",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger.info(
            f"{self.__class__.__name__} initialized with qwen3_type={qwen3_type}, "
            f"reaction_input_dim={reaction_input_dim}, num_tokens_per_molecule={num_tokens_per_molecule}, "
            f"compress_reaction={compress_reaction}, "
            f"esm_tokenizer_name_or_path={esm_tokenizer_name_or_path}"
        )
        self.num_tokens_per_molecule = num_tokens_per_molecule
        self.compress_reaction = compress_reaction
        self.tokenizer = EsmTokenizer.from_pretrained(esm_tokenizer_name_or_path)

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
                add_cross_attention=False,
            )
            self.qwen3 = Qwen3CAForCausalLM(config)
        else:
            config = Qwen3Config.from_pretrained(
                qwen3_type,
                add_cross_attention=False,
                vocab_size=len(self.tokenizer.get_vocab()),
            )
            self.qwen3 = Qwen3CAForCausalLM(config)

        self.qwen3.gradient_checkpointing_enable()
        qwen_hidden_size = self.qwen3.config.hidden_size
        fusion_hidden_dim = reaction_hidden_dim or qwen_hidden_size

        self.atom_projector = nn.Sequential(
            nn.Linear(reaction_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, qwen_hidden_size),
        )
        self.substrate_role_embedding = nn.Parameter(
            torch.randn(1, 1, qwen_hidden_size) * 0.02
        )
        self.product_role_embedding = nn.Parameter(
            torch.randn(1, 1, qwen_hidden_size) * 0.02
        )

        self.substrate_to_product_attn = nn.MultiheadAttention(
            embed_dim=qwen_hidden_size,
            num_heads=reaction_num_heads,
            dropout=reaction_dropout,
            batch_first=True,
        )
        self.product_to_substrate_attn = nn.MultiheadAttention(
            embed_dim=qwen_hidden_size,
            num_heads=reaction_num_heads,
            dropout=reaction_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(reaction_dropout)
        self.substrate_norm1 = nn.LayerNorm(qwen_hidden_size)
        self.product_norm1 = nn.LayerNorm(qwen_hidden_size)
        self.substrate_norm2 = nn.LayerNorm(qwen_hidden_size)
        self.product_norm2 = nn.LayerNorm(qwen_hidden_size)
        self.substrate_ffn = nn.Sequential(
            nn.Linear(qwen_hidden_size, qwen_hidden_size * 4),
            nn.GELU(),
            nn.Dropout(reaction_dropout),
            nn.Linear(qwen_hidden_size * 4, qwen_hidden_size),
        )
        self.product_ffn = nn.Sequential(
            nn.Linear(qwen_hidden_size, qwen_hidden_size * 4),
            nn.GELU(),
            nn.Dropout(reaction_dropout),
            nn.Linear(qwen_hidden_size * 4, qwen_hidden_size),
        )

        if self.compress_reaction:
            self.substrate_compressor = LatentReactionCompressor(
                hidden_size=qwen_hidden_size,
                num_latents=num_tokens_per_molecule,
                num_heads=reaction_num_heads,
                dropout=reaction_dropout,
            )
            self.product_compressor = LatentReactionCompressor(
                hidden_size=qwen_hidden_size,
                num_latents=num_tokens_per_molecule,
                num_heads=reaction_num_heads,
                dropout=reaction_dropout,
            )
        else:
            self.substrate_compressor = None
            self.product_compressor = None

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=qwen_hidden_size,
            nhead=reaction_num_heads,
            dim_feedforward=qwen_hidden_size * 4,
            dropout=reaction_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.reaction_token_encoder = nn.TransformerEncoder(
            fusion_layer, num_layers=reaction_fusion_layers
        )

    def set_objective_and_metrics(self, stage: str = "train", experiment=None):
        if stage == "train":
            train_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "ce_loss": Scalar(dist_sync_on_step=True),
                "nonpad_ratio": Scalar(dist_sync_on_step=True),
            }
            val_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "ce_loss": Scalar(dist_sync_on_step=True),
                "nonpad_ratio": Scalar(dist_sync_on_step=True),
            }
            train_metrics = {
                k: v.to(experiment.device) for k, v in train_metrics.items()
            }
            val_metrics = {k: v.to(experiment.device) for k, v in val_metrics.items()}
            test_metrics = None
        else:
            train_metrics = None
            val_metrics = None
            test_metrics = {
                "gt_seq_softlcs_score": Scalar(dist_sync_on_step=True),
            }
            test_metrics = {k: v.to(experiment.device) for k, v in test_metrics.items()}
        experiment.metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }

    def _apply_bidirectional_reaction_fusion(
        self,
        substrate_tokens: torch.Tensor,
        substrate_mask: torch.Tensor,
        product_tokens: torch.Tensor,
        product_mask: torch.Tensor,
    ):
        substrate_context, _ = self.substrate_to_product_attn(
            query=substrate_tokens,
            key=product_tokens,
            value=product_tokens,
            key_padding_mask=~product_mask.bool(),
            need_weights=False,
        )
        product_context, _ = self.product_to_substrate_attn(
            query=product_tokens,
            key=substrate_tokens,
            value=substrate_tokens,
            key_padding_mask=~substrate_mask.bool(),
            need_weights=False,
        )

        substrate_tokens = self.substrate_norm1(
            substrate_tokens + self.dropout(substrate_context)
        )
        product_tokens = self.product_norm1(
            product_tokens + self.dropout(product_context)
        )

        substrate_tokens = self.substrate_norm2(
            substrate_tokens + self.dropout(self.substrate_ffn(substrate_tokens))
        )
        product_tokens = self.product_norm2(
            product_tokens + self.dropout(self.product_ffn(product_tokens))
        )
        return substrate_tokens, product_tokens

    def forward_encoder(
        self,
        substrate_atom_tokens: torch.Tensor,
        substrate_atom_masks: torch.Tensor,
        product_atom_tokens: torch.Tensor,
        product_atom_masks: torch.Tensor,
    ):
        _pd = next(self.atom_projector.parameters()).dtype
        substrate_atom_tokens = substrate_atom_tokens.to(dtype=_pd)
        product_atom_tokens = product_atom_tokens.to(dtype=_pd)

        substrate_tokens = self.atom_projector(substrate_atom_tokens)
        product_tokens = self.atom_projector(product_atom_tokens)

        substrate_tokens = substrate_tokens + self.substrate_role_embedding
        product_tokens = product_tokens + self.product_role_embedding

        substrate_tokens, product_tokens = self._apply_bidirectional_reaction_fusion(
            substrate_tokens,
            substrate_atom_masks,
            product_tokens,
            product_atom_masks,
        )

        if self.compress_reaction:
            compressed_substrate = self.substrate_compressor(
                substrate_tokens, substrate_atom_masks
            )
            compressed_product = self.product_compressor(
                product_tokens, product_atom_masks
            )
            reaction_tokens = torch.cat([compressed_substrate, compressed_product], dim=1)
            reaction_mask = torch.ones(
                reaction_tokens.shape[:2],
                dtype=torch.long,
                device=reaction_tokens.device,
            )
            key_padding = None
        else:
            reaction_tokens = torch.cat([substrate_tokens, product_tokens], dim=1)
            reaction_mask = torch.cat(
                [substrate_atom_masks, product_atom_masks], dim=1
            ).to(dtype=torch.long, device=reaction_tokens.device)
            key_padding = ~reaction_mask.bool()

        reaction_tokens = self.reaction_token_encoder(
            reaction_tokens, src_key_padding_mask=key_padding
        )
        reaction_tokens = reaction_tokens.to(dtype=next(self.qwen3.parameters()).dtype)
        return reaction_tokens, reaction_mask

    def _build_decoder_inputs(
        self,
        seq_ids: torch.Tensor,
        seq_masks: torch.Tensor,
        reaction_tokens: torch.Tensor,
        reaction_mask: torch.Tensor,
    ):
        token_embed_layer = self.qwen3.get_input_embeddings()
        seq_embeds = token_embed_layer(seq_ids)

        cls_ids = torch.full(
            (seq_ids.shape[0], 1),
            fill_value=self.tokenizer.cls_token_id,
            dtype=seq_ids.dtype,
            device=seq_ids.device,
        )
        cls_embeds = token_embed_layer(cls_ids)
        cls_mask = torch.ones_like(cls_ids)

        inputs_embeds = torch.cat([reaction_tokens, cls_embeds, seq_embeds], dim=1)
        attention_mask = torch.cat([reaction_mask, cls_mask, seq_masks], dim=1)

        target = seq_ids.clone()
        target = target.masked_fill(target == self.tokenizer.pad_token_id, -100)
        prefix_ignore = torch.full(
            (target.shape[0], reaction_tokens.shape[1] + 1),
            fill_value=-100,
            dtype=target.dtype,
            device=target.device,
        )
        labels = torch.cat([prefix_ignore, target], dim=1)
        return inputs_embeds, attention_mask, labels

    def _forward(
        self,
        seq_ids: torch.Tensor,
        seq_masks: torch.Tensor,
        substrate_atom_tokens: torch.Tensor,
        substrate_atom_masks: torch.Tensor,
        product_atom_tokens: torch.Tensor,
        product_atom_masks: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        reaction_tokens, reaction_mask = self.forward_encoder(
            substrate_atom_tokens=substrate_atom_tokens,
            substrate_atom_masks=substrate_atom_masks,
            product_atom_tokens=product_atom_tokens,
            product_atom_masks=product_atom_masks,
        )
        inputs_embeds, attention_mask, labels = self._build_decoder_inputs(
            seq_ids=seq_ids,
            seq_masks=seq_masks,
            reaction_tokens=reaction_tokens,
            reaction_mask=reaction_mask,
        )
        out = self.qwen3(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            return_dict=True,
        )
        return out

    def forward(
        self,
        seq_ids: torch.Tensor,
        seq_masks: torch.Tensor,
        substrate_atom_tokens: torch.Tensor,
        substrate_atom_masks: torch.Tensor,
        product_atom_tokens: torch.Tensor,
        product_atom_masks: torch.Tensor,
        **kwargs,
    ):
        out_dict = self._forward(
            seq_ids=seq_ids,
            seq_masks=seq_masks,
            substrate_atom_tokens=substrate_atom_tokens,
            substrate_atom_masks=substrate_atom_masks,
            product_atom_tokens=product_atom_tokens,
            product_atom_masks=product_atom_masks,
            labels=None,
        )
        target = seq_ids.clone()
        target = target.masked_fill(target == self.tokenizer.pad_token_id, -100)
        sample_size = target.ne(-100).float().sum()
        nonpad_ratio = sample_size / target.numel()
        ce_loss = out_dict["loss"]
        return {
            "logits": out_dict["logits"],
            "target": target,
            "loss": ce_loss,
            "ce_loss": ce_loss,
            "sample_size": sample_size,
            "nonpad_ratio": nonpad_ratio,
        }

    def to_list(self, seq: torch.Tensor):
        return [seq[i, ...].detach().cpu().numpy().tolist() for i in range(seq.shape[0])]

    def clean_and_format_seq(self, seq: list[str]):
        cleaned_data = []
        for item in seq:
            processed_string = re.sub(r"<cls>", "", item)
            processed_string = re.sub(r"<eos>", "", processed_string)
            processed_string = re.sub(r"<pad>", "", processed_string)
            processed_string = processed_string.replace(" ", "")
            cleaned_data.append(processed_string)
        return cleaned_data

    def generate(
        self,
        substrate_atom_tokens: torch.Tensor,
        substrate_atom_masks: torch.Tensor,
        product_atom_tokens: torch.Tensor,
        product_atom_masks: torch.Tensor,
        generation_config: GenerationConfig,
        verbose=True,
        **kwargs,
    ):
        reaction_tokens, reaction_mask = self.forward_encoder(
            substrate_atom_tokens=substrate_atom_tokens,
            substrate_atom_masks=substrate_atom_masks,
            product_atom_tokens=product_atom_tokens,
            product_atom_masks=product_atom_masks,
        )
        cls_ids = torch.full(
            (reaction_tokens.shape[0], 1),
            fill_value=self.tokenizer.cls_token_id,
            dtype=torch.long,
            device=reaction_tokens.device,
        )
        cls_embeds = self.qwen3.get_input_embeddings()(cls_ids).to(
            dtype=reaction_tokens.dtype
        )
        cls_mask = torch.ones_like(cls_ids)
        prompt_embeds = torch.cat([reaction_tokens, cls_embeds], dim=1)
        prompt_mask = torch.cat([reaction_mask, cls_mask], dim=1)

        generation_config.bos_token_id = self.tokenizer.cls_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        sample_results = self.qwen3.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_mask,
            generation_config=generation_config,
            num_return_sequences=1,
            return_dict_in_generate=True,
        )
        tokens = self.to_list(sample_results.sequences)
        sequences = self.tokenizer.batch_decode(tokens)
        return {"output_seqs": self.clean_and_format_seq(sequences)}

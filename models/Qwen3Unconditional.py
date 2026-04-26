import re
from typing import Optional

import torch

# from .Qwen3.modeling_domainconditioning_qwen3 import Qwen3CAForCausalLM
from transformers import EsmTokenizer, GenerationConfig, Qwen3ForCausalLM

from utils.init_utils import construct_class_by_name
from utils.metrics import Scalar

from .BaseModel import BaseModel
from .Qwen3.configuration_domainconditioning_qwen3 import Qwen3Config


class Qwen3Unconditional(BaseModel):
    def __init__(
        self,
        qwen3_type="Qwen/Qwen3-100M",
        criterion_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger.info(
            f"{self.__class__.__name__} initialized with gpt_type: {qwen3_type}"
        )

        self.tokenizer = EsmTokenizer.from_pretrained("airkingbd/dplm_150m")
        if qwen3_type == "Qwen/Qwen3-100M":
            # 创建100M配置
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
                use_cache=True,  #
                use_sliding_window=False,
                vocab_size=len(self.tokenizer.get_vocab()),
            )
            # 创建模型
            self.qwen3 = Qwen3ForCausalLM(config)
        else:
            config = Qwen3Config.from_pretrained(
                qwen3_type, vocab_size=len(self.tokenizer.get_vocab())
            )
            self.qwen3 = Qwen3ForCausalLM(config)

        self.qwen3.gradient_checkpointing_enable()

        if criterion_kwargs is not None:
            self.criterion = construct_class_by_name(
                **criterion_kwargs, logger=self.logger
            )
        else:
            self.criterion = None
        # self.cfg = self.dplm.cfg

    def set_objective_and_metrics(self, stage: str = "train", experiment=None):
        # we have loss, nll_loss, ppl, fullseq_loss, fullseq_nll_loss, bsz, sample_size, sample_ratio, nonpad_ratio, weight_diff_loss
        if stage == "train":
            train_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
            }
            val_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
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

    def _forward(
        self,
        seq_ids: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> dict:
        out_dict = self.qwen3(
            input_ids=seq_ids,
            labels=labels,
            return_dict=True,
        )
        return out_dict  # [bs, seq_len, vocab_size]

    def forward(
        self,
        seq_ids: torch.Tensor,
        **kwargs,
    ):
        """
        this function is used for training, computing the loss
        seq_ids: [bs, seq_len]
        kwargs: other arguments

        return:
            logits: [bs, seq_len, vocab_size]
            target: [bs, seq_len]
            loss_mask: [bs, seq_len]
            loss: the loss value
        """
        target = seq_ids.clone()
        target = target.masked_fill(target == self.tokenizer.pad_token_id, -100)

        out_dict = self._forward(seq_ids, labels=target)
        logits = out_dict["logits"]
        if self.criterion:
            loss, logging_output = self.criterion(logits, target)
        else:
            loss = out_dict["loss"]
            logging_output = {}
        return {"logits": logits, "target": target, "loss": loss, **logging_output}

    def initialize_output_tokens(self, bs: int, **kwargs):
        start_id = self.tokenizer.cls_token_id
        input_ids = (
            (torch.zeros((1)) + start_id).unsqueeze(0).repeat(bs, 1)
        )  # create batch dim
        input_ids = input_ids.to(torch.long)
        input_ids = input_ids.to(next(self.parameters()).device)
        return input_ids

    def to_list(self, seq: torch.Tensor):
        return [
            seq[i, ...].detach().cpu().numpy().tolist() for i in range(seq.shape[0])
        ]

    def clean_and_format_seq(self, seq: list[str]):
        cleaned_data = []
        for item in seq:
            processed_string = re.sub(r"<cls>", "", item)
            processed_string = re.sub(r"<eos>", "", processed_string)
            processed_string = processed_string.replace(" ", "")
            cleaned_data.append(processed_string)
        return cleaned_data

    def generate(
        self,
        generation_config: GenerationConfig,
        verbose=True,
        **kwargs,
    ):
        generation_config.bos_token_id = self.tokenizer.cls_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        # 1) initialized from all mask tokens
        # initial_output_tokens = self.initialize_output_tokens(bs=encoder_out.shape[0])
        sample_results = self.qwen3.generate(
            generation_config=generation_config,
            num_return_sequences=1,
            return_dict_in_generate=True,
        )
        tokens = self.to_list(sample_results.sequences)
        sequences = self.tokenizer.batch_decode(tokens)
        return {
            "output_seqs": self.clean_and_format_seq(sequences),
        }

import re
from typing import Optional

import torch
from transformers import EsmTokenizer, GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from utils.init_utils import construct_class_by_name
from utils.metrics import Scalar

from .BaseModel import BaseModel


class Qwen3wDomainConditioning(BaseModel):
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
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<domain_cls>", "<domain_eos>"]}
        )
        # Since we need to use the cross attention, we need to set the use_cache to False and is_decoder to True
        if qwen3_type == "Qwen/Qwen3-100M":
            # 创建100M配置
            config = Qwen3Config(
                architectures=["Qwen3ForCausalLM"],
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=0,
                eos_token_id=1,
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
            # self.qwen3 = Qwen3ForCausalLM.from_pretrained(
            #     qwen3_type, vocab_size=len(self.tokenizer.get_vocab())
            # )
            qwen3_config = Qwen3Config.from_pretrained(qwen3_type, vocab_size=len(self.tokenizer.get_vocab()))
            self.qwen3 = Qwen3ForCausalLM(qwen3_config)
        self.qwen3.gradient_checkpointing_enable()

        if criterion_kwargs is not None:
            self.criterion = construct_class_by_name(
                **criterion_kwargs, logger=self.logger
            )
        else:
            self.criterion = None

    def set_objective_and_metrics(self, stage: str = "train", experiment=None):
        # we have loss, nll_loss, ppl, fullseq_loss, fullseq_nll_loss, bsz, sample_size, sample_ratio, nonpad_ratio, weight_diff_loss
        if stage == "train":
            train_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "domain_loss": Scalar(dist_sync_on_step=True),
                "non_domain_loss": Scalar(dist_sync_on_step=True),
                "domain_weight": Scalar(dist_sync_on_step=True),
                "non_domain_weight": Scalar(dist_sync_on_step=True),
                "unweighted_loss": Scalar(dist_sync_on_step=True),
            }
            val_metrics = {
                "loss": Scalar(dist_sync_on_step=True),
                "domain_loss": Scalar(dist_sync_on_step=True),
                "non_domain_loss": Scalar(dist_sync_on_step=True),
                "domain_weight": Scalar(dist_sync_on_step=True),
                "non_domain_weight": Scalar(dist_sync_on_step=True),
                "unweighted_loss": Scalar(dist_sync_on_step=True),
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

    def _forward(self, input_ids: torch.Tensor, labels: torch.Tensor, **kwargs) -> dict:
        out_dict = self.qwen3(input_ids=input_ids, labels=labels, return_dict=True)
        return out_dict  # [bs, seq_len, vocab_size]

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, **kwargs):
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

        out_dict = self._forward(input_ids, labels=labels)
        logits = out_dict["logits"]
        if self.criterion:
            domain_mask = self._create_domain_mask(
                input_ids, kwargs["domain_positions"]
            )
            loss, logging_output = self.criterion(logits, labels, domain_mask)
        else:
            loss = out_dict["loss"]
            logging_output = {}
        return {"logits": logits, "target": labels, "loss": loss, **logging_output}

    def _create_domain_mask(
        self,
        input_ids: torch.Tensor,
        domain_positions: list[list[tuple[int, int]]],
    ) -> torch.Tensor:
        """
        根据domain信息创建domain_mask
        Args:
            seq_ids: [bs, seq_len] 序列
            domain_positions: bs, num_domain, domain_positions
        Returns:
            domain_mask: [bs, seq_len] 1表示domain区域，0表示非domain区域
        """
        batch_size, seq_len = input_ids.shape
        domain_mask = torch.zeros(batch_size, seq_len, device=input_ids.device)

        try:
            cls_token_id = self.tokenizer.cls_token_id
        except AttributeError:
            raise ValueError(
                "Could not find cls_token_id in model config. Please ensure it's available."
            )

        for i, positions in enumerate(domain_positions):
            # noted that here we have domain token in the input_ids, so we need to shift to seq ids
            cls_indices = (input_ids[i] == cls_token_id).nonzero(as_tuple=True)[0]
            if cls_indices.numel() == 0:
                # 如果在 input_ids 中找不到 cls_token，则跳过此样本
                continue
            seq_start_offset = cls_indices[0].item()
            for start, end in positions:
                domain_mask[
                    i, start + seq_start_offset + 1 : end + seq_start_offset + 1
                ] = 1  # shift by 1 to exclude the cls token

        return domain_mask

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
        self, generation_config: GenerationConfig = None, verbose=True, **kwargs
    ):
        if generation_config is None:
            generation_config = self.default_generation_config

        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.bos_token_id = self.tokenizer.bos_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        model_inputs = self.prepare_inputs_for_generation(**kwargs)

        original_max_length = generation_config.max_length
        generation_config.max_new_tokens = original_max_length
        generation_config.max_length = None 

        sample_results = self.qwen3.generate(
            **model_inputs,
            generation_config=generation_config,
            num_return_sequences=1,
            return_dict_in_generate=True,
        )

        # Decode and clean
        # The output includes the prompt, so we need to remove it.
        generated_tokens = sample_results.sequences[
            :, model_inputs.input_ids.shape[1] :
        ]
        sequences = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        cleaned_sequences = []
        for seq in sequences:
            # The default tokenizer might add spaces between tokens. We want to remove them for protein sequences.
            cleaned_sequences.append(seq.replace(" ", ""))
        return {
            "output_seqs": cleaned_sequences,
        }

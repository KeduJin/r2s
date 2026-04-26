# domain to sequence generation
import json
import os

import torch
import tree
from tqdm import tqdm
from transformers import GenerationConfig

from experiments.BaseExperiment import BaseExperiment
from utils.init_utils import construct_class_by_name
from utils.SoftLCS import get_batched_wildcard_lcs_score
from utils.experiments_utils import timer

class D2S(BaseExperiment):
    @timer
    def __init__(self, cfg):
        super().__init__(cfg)

        if hasattr(self.cfg.Trainer, "criterion_kwargs"):
            self.criterion = construct_class_by_name(
                class_name=self.cfg.Trainer.criterion_kwargs.class_name,
            )
        else:
            self.criterion = None

    @timer
    def load_state_from_checkpoints(self, ckpt_path: str) -> None:
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

    @timer
    def configure_testing(self) -> None:
        # config model, optimizer and dataloader
        self.model = construct_class_by_name(
            **self.cfg.Model.kwargs.to_dict(), logger=self._log
        )
        datamodule = construct_class_by_name(
            **self.cfg.Datamodule.kwargs.to_dict(),
            logger=self._log,
            tokenizer=self.model.tokenizer,
        )
        datamodule.setup(stage="test")
        self.test_loader = datamodule.test_dataloader()

        if self.cfg.ckpt_path:
            self._log.info(f"Loading checkpoint from {self.cfg.ckpt_path}")
            self.load_state_from_checkpoints(
                os.path.join(self.cfg.ckpt_path, "pytorch_model.bin")
            )
            self.output_dir = self.cfg.test_output_dir
            self.trained_steps = torch.load(
                os.path.join(self.cfg.ckpt_path, "training_state.pth")
            )["trained_steps"]
        else:
            self._log.info(
                "No checkpoint path provided, initialize the model from scratch"
            )
            self.output_dir = self.cfg.test_output_dir
            self.trained_steps = 0

        self._log.info(f"Output directory: {self.output_dir}")
        # configure object and set metrics
        self.model.set_objective_and_metrics(stage="test", experiment=self)
        # count the number of parameters
        # accelerate prepraration
        self.model, self.test_loader = self.accelerator.prepare(
            self.model, self.test_loader
        )

    def update_metrics(
        self, res: dict, stage: str = "train", weight: torch.Tensor = None
    ) -> None:
        for k, v in res.items():
            if k in self.metrics[stage]:
                # 对于验证和测试阶段，如果提供了权重且该指标需要加权，则使用权重
                # 注意：sample_size 本身不应该被加权更新
                if (
                    weight is not None
                    and stage in ["val", "test"]
                    and k != "sample_size"
                ):
                    self.metrics[stage][k].update(v, weight=weight)
                else:
                    self.metrics[stage][k].update(v)

    @torch.no_grad()
    def val_epoch(
        self, val_loader: torch.utils.data.DataLoader, val_name: str = ""
    ) -> dict:
        self.model.eval()
        for idx, batch in tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"validating val{val_name}...",
            disable=not self.is_main_process,
        ):
            batch = tree.map_structure(
                lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, batch
            )
            res = self.loss_fn(idx, batch)
            # 提取 sample_size 作为权重（如果存在）
            weight = res.get("sample_size", None)
            if weight is not None and isinstance(weight, torch.Tensor):
                # 确保 weight 是标量或可以转换为标量
                if weight.numel() > 1:
                    weight = weight.sum()
                weight = weight.detach()
            # update metrics
            self.update_metrics(res, stage="val", weight=weight)

        # log validation metrics
        reported_metrics = self.calculate_metrics(stage="val")
        log_metrics = {**reported_metrics, "epoch": self.trained_epochs}

        return log_metrics

    @torch.inference_mode()
    def test_epoch(self, test_loader: torch.utils.data.DataLoader) -> dict:
        self.model.eval()
        generation_fn = (
            self.model.generate
            if self.accelerator.num_processes == 1
            else self.model.module.generate
        )

        entry_id_list = []
        domains_list = []
        seq_list = []
        generated_seq_list = []
        for idx, batch in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc="testing...",
            disable=not self.is_main_process,
        ):
            batch = tree.map_structure(
                lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, batch
            )

            domains_list.extend(batch["domain"])
            seq_list.extend(batch["seq"])
            entry_id_list.extend(batch["entry_id"])

            # 获取生成方法（如果配置中有的话）
            generation_kwargs = {}
            if hasattr(self.cfg, "GenerationMethod") and hasattr(self.cfg.GenerationMethod, "method"):
                generation_kwargs["generation_method"] = self.cfg.GenerationMethod.method

            # generation by batch
            res = generation_fn(
                **batch,
                generation_config=GenerationConfig(
                    **self.cfg.GenerationConfig.to_dict()
                ),
                **generation_kwargs,
            )
            generated_seq_list.extend(res["output_seqs"])

            # we mainly evaluate the output_seqs in two view:
            # 1. the soft lcs score
            # 2. the plddt score
            # ? 3. the test set accuracy ?
            # Here we only evaluate the soft lcs score and output all seqs to the output_dir
            gt_seq_softlcs_score = torch.mean(
                torch.tensor(
                    get_batched_wildcard_lcs_score(batch["seq"], res["output_seqs"])
                )
            ).to(self.device)
            softlcs_score = torch.mean(
                torch.tensor(
                    get_batched_wildcard_lcs_score(batch["domain"], res["output_seqs"])
                )
            ).to(self.device)
            ref_softlcs_score = torch.mean(
                torch.tensor(
                    get_batched_wildcard_lcs_score(batch["domain"], batch["seq"])
                )
            ).to(self.device)
            # update metrics
            self.update_metrics(
                {
                    "softlcs_score": softlcs_score,
                    "ref_softlcs_score": ref_softlcs_score,
                    "gt_seq_softlcs_score": gt_seq_softlcs_score,
                },
                stage="test",
            )
        reported_metrics = self.calculate_metrics(stage="test")
        reported_metrics = {k: v.item() for k, v in reported_metrics.items()}
        log_metrics = {**reported_metrics, "steps": self.trained_steps}

        return log_metrics, [domains_list, seq_list, generated_seq_list, entry_id_list]

    def start_testing(self) -> None:
        self._log.info("Starting testing...")
        log_metrics, write_content_list = self.test_epoch(self.test_loader)
        self._log.info_dic_step(log_metrics, step=self.trained_steps)
        self._log.info("Done")

        # save log_metrics to json
        if self.accelerator.is_main_process:
            with open(os.path.join(self.output_dir, "log_metrics.json"), "w") as f:
                json.dump(log_metrics, f)

        # save the write_content_list to the output_dir
        out_file_name = (
            "sequence_output.tsv"
            if self.accelerator.num_processes == 1
            else "sequence_output_rank{}.tsv".format(self.accelerator.process_index)
        )
        out_file_path = os.path.join(self.output_dir, out_file_name)

        with open(out_file_path, "w") as f:
            f.write("entry_id\tdomains\tgt_seq\tgenerated_seq\n")
            for domains, seq, generated_seq, entry_id in zip(*write_content_list):
                f.write(f"{entry_id}\t{domains}\t{seq}\t{generated_seq}\n")

        self._log.info(f"Saved the generated sequences to {out_file_path}")
        self.accelerator.wait_for_everyone()

    @torch.inference_mode()
    def generate_epoch(self, test_loader: torch.utils.data.DataLoader) -> dict:
        self.model.eval()
        generation_fn = (
            self.model.generate
            if self.accelerator.num_processes == 1
            else self.model.module.generate
        )

        domains_list = []
        generated_seq_list = []
        for idx, batch in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc="generating...",
            disable=not self.is_main_process,
        ):
            batch = tree.map_structure(
                lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, batch
            )

            domains_list.extend(batch["domain"])

            # 获取生成方法（如果配置中有的话）
            generation_kwargs = {}
            if hasattr(self.cfg, "GenerationMethod") and hasattr(self.cfg.GenerationMethod, "method"):
                generation_kwargs["generation_method"] = self.cfg.GenerationMethod.method

            # generation by batch
            res = generation_fn(
                **batch,
                generation_config=GenerationConfig(
                    **self.cfg.GenerationConfig.to_dict()
                ),
                **generation_kwargs,
            )
            generated_seq_list.extend(res["output_seqs"])
        return [domains_list, generated_seq_list]

    @timer
    def start_generating(self) -> None:
        self._log.info("Starting generating...")
        write_content_list = self.generate_epoch(self.test_loader)
        self._log.info("Done")

        # save the write_content_list to the output_dir
        out_file_name = (
            "sequence_output.tsv"
            if self.accelerator.num_processes == 1
            else "sequence_output_rank{}.tsv".format(self.accelerator.process_index)
        )
        out_file_path = os.path.join(self.output_dir, out_file_name)

        with open(out_file_path, "w") as f:
            f.write("domains\tgenerated_seq\n")
            for domains, generated_seq in zip(*write_content_list):
                f.write(f"{domains}\t{generated_seq}\n")

        self._log.info(f"Saved the generated sequences to {out_file_path}")
        self.accelerator.wait_for_everyone()

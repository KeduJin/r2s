# reaction to sequence generation
import json
import os

import torch
import tree
from tqdm import tqdm
from transformers import GenerationConfig

from experiments.BaseExperiment import BaseExperiment
from utils.SoftLCS import get_batched_wildcard_lcs_score
from utils.experiments_utils import timer
from utils.init_utils import construct_class_by_name


class R2S(BaseExperiment):
    @timer
    def __init__(self, cfg):
        super().__init__(cfg)

    @timer
    def load_state_from_checkpoints(self, ckpt_path: str) -> None:
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

    @timer
    def configure_testing(self) -> None:
        self.model = construct_class_by_name(
            **self.cfg.Model.kwargs.to_dict(), logger=self._log
        )
        datamodule = construct_class_by_name(
            **self.cfg.Datamodule.kwargs.to_dict(),
            logger=self._log,
            tokenizer=self.model.tokenizer,
            condition_tokenizer=getattr(self.model, "condition_tokenizer", None),
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
        self.model.set_objective_and_metrics(stage="test", experiment=self)
        self.model, self.test_loader = self.accelerator.prepare(
            self.model, self.test_loader
        )

    def update_metrics(
        self, res: dict, stage: str = "train", weight: torch.Tensor = None
    ) -> None:
        for k, v in res.items():
            if k in self.metrics[stage]:
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
            weight = res.get("sample_size", None)
            if weight is not None and isinstance(weight, torch.Tensor):
                if weight.numel() > 1:
                    weight = weight.sum()
                weight = weight.detach()
            self.update_metrics(res, stage="val", weight=weight)

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
        reaction_list = []
        raw_reaction_list = []
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
            reaction_list.extend(batch["reaction"])
            raw_reaction_list.extend(batch.get("raw_reaction", batch["reaction"]))
            seq_list.extend(batch["seq"])
            entry_id_list.extend(batch["entry_id"])

            generation_kwargs = {}
            if hasattr(self.cfg, "GenerationMethod") and hasattr(
                self.cfg.GenerationMethod, "method"
            ):
                generation_kwargs["generation_method"] = self.cfg.GenerationMethod.method

            res = generation_fn(
                **batch,
                generation_config=GenerationConfig(
                    **self.cfg.GenerationConfig.to_dict()
                ),
                **generation_kwargs,
            )
            generated_seq_list.extend(res["output_seqs"])

            gt_seq_softlcs_score = torch.mean(
                torch.tensor(
                    get_batched_wildcard_lcs_score(batch["seq"], res["output_seqs"])
                )
            ).to(self.device)
            self.update_metrics(
                {"gt_seq_softlcs_score": gt_seq_softlcs_score},
                stage="test",
            )

        reported_metrics = self.calculate_metrics(stage="test")
        reported_metrics = {k: v.item() for k, v in reported_metrics.items()}
        log_metrics = {**reported_metrics, "steps": self.trained_steps}
        return log_metrics, [
            reaction_list,
            raw_reaction_list,
            seq_list,
            generated_seq_list,
            entry_id_list,
        ]

    def start_testing(self) -> None:
        self._log.info("Starting testing...")
        log_metrics, write_content_list = self.test_epoch(self.test_loader)
        self._log.info_dic_step(log_metrics, step=self.trained_steps)
        self._log.info("Done")

        if self.accelerator.is_main_process:
            with open(os.path.join(self.output_dir, "log_metrics.json"), "w") as f:
                json.dump(log_metrics, f)

        out_file_name = (
            "sequence_output.tsv"
            if self.accelerator.num_processes == 1
            else "sequence_output_rank{}.tsv".format(self.accelerator.process_index)
        )
        out_file_path = os.path.join(self.output_dir, out_file_name)
        with open(out_file_path, "w") as f:
            f.write("entry_id\treaction_smiles\traw_reaction\tgt_seq\tgenerated_seq\n")
            for reaction, raw_reaction, seq, generated_seq, entry_id in zip(
                *write_content_list
            ):
                f.write(
                    f"{entry_id}\t{reaction}\t{raw_reaction}\t{seq}\t{generated_seq}\n"
                )

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

        reaction_list = []
        raw_reaction_list = []
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
            reaction_list.extend(batch["reaction"])
            raw_reaction_list.extend(batch.get("raw_reaction", batch["reaction"]))

            generation_kwargs = {}
            if hasattr(self.cfg, "GenerationMethod") and hasattr(
                self.cfg.GenerationMethod, "method"
            ):
                generation_kwargs["generation_method"] = self.cfg.GenerationMethod.method

            res = generation_fn(
                **batch,
                generation_config=GenerationConfig(
                    **self.cfg.GenerationConfig.to_dict()
                ),
                **generation_kwargs,
            )
            generated_seq_list.extend(res["output_seqs"])
        return [reaction_list, raw_reaction_list, generated_seq_list]

    @timer
    def start_generating(self) -> None:
        self._log.info("Starting generating...")
        write_content_list = self.generate_epoch(self.test_loader)
        self._log.info("Done")

        out_file_name = (
            "sequence_output.tsv"
            if self.accelerator.num_processes == 1
            else "sequence_output_rank{}.tsv".format(self.accelerator.process_index)
        )
        out_file_path = os.path.join(self.output_dir, out_file_name)

        with open(out_file_path, "w") as f:
            f.write("reaction_smiles\traw_reaction\tgenerated_seq\n")
            for reaction, raw_reaction, generated_seq in zip(*write_content_list):
                f.write(f"{reaction}\t{raw_reaction}\t{generated_seq}\n")

        self._log.info(f"Saved the generated sequences to {out_file_path}")
        self.accelerator.wait_for_everyone()

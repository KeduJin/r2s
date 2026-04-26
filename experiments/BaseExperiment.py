import math
import os
import time
import warnings
from datetime import timedelta
from typing import Optional

import torch
import tree
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from accelerate.utils import DataLoaderConfiguration, set_seed
from tqdm import tqdm

from utils import experiments_utils as eu
from utils import logger, trackers
from utils.config_utils import Config
from utils.git_utils import get_branch_name, get_commit_hash
from utils.init_utils import construct_class_by_name
from utils.metrics import TimePerStep
from utils.experiments_utils import timer

warnings.filterwarnings("ignore")
process_group_kwargs = InitProcessGroupKwargs(
    backend="nccl", timeout=timedelta(seconds=7200)
)  # 1.5 hours
dataloader_config = DataLoaderConfiguration(
    use_seedable_sampler=os.environ.get("USE_SEEDABLE_SAMPLER", True)
)
os.environ["WANDB_MODE"] = "offline"


class BaseExperiment:
    @timer
    def __init__(self, cfg: Config) -> None:
        self.accelerator = Accelerator(
            log_with=None,
            gradient_accumulation_steps=getattr(
                cfg.Trainer, "accumulate_grad_batches", 1
            ),
            dataloader_config=dataloader_config,
            kwargs_handlers=[process_group_kwargs],
        )
        set_seed(cfg.Trainer.seed, device_specific=True)
        # init logging
        self._log = logger.MyLogger(
            output_dir=cfg.Settings.outdir if cfg.istraining else cfg.test_output_dir
        )
        self._log.info("Starting experiments")
        self._log.info(self.accelerator.state)
        self._log.info(f"outdir path: {cfg.Settings.outdir}")
        # print the git commit hash
        self._log.info(
            f"git commit hash: {get_commit_hash()}, branch: {get_branch_name()}"
        )
        eu.log_config(self._log, cfg)
        self.cfg = cfg
        self.device = self.accelerator.device
        self.is_main_process = self.accelerator.is_main_process
        self.zero_stage = (
            0
            if self.accelerator.state.deepspeed_plugin is None
            else self.accelerator.state.deepspeed_plugin.zero_stage
        )
        if cfg.istraining:
            self.configure_training()
        else:
            self.configure_testing()

    def configure_testing(self) -> None:
        pass

    def calculate_metrics(self, stage: str = "train") -> dict:
        res = {stage + "/" + k: v.compute() for k, v in self.metrics[stage].items()}
        for k, v in self.metrics[stage].items():
            v.reset()
        return res

    def update_metrics(self, res: dict, stage: str = "train") -> None:
        for k, v in res.items():
            if k in self.metrics[stage]:
                self.metrics[stage][k].update(v)

    @timer
    def configure_training(self) -> None:
        self.max_epochs = self.cfg.Trainer.max_epochs
        self.trained_epochs, self.trained_steps = 0, 0
        self.acumulate_grad_batches = self.cfg.Trainer.accumulate_grad_batches

        # configure tracker
        self.accelerator.log_with = trackers.init_trackers(self.cfg)
        self.accelerator.init_trackers(
            project_name=self.cfg.Settings.project_name, config=self.cfg.to_dict()
        )

        # config model, optimizer and dataloader
        self.model = construct_class_by_name(
            **self.cfg.Model.kwargs.to_dict(), logger=self._log
        )
        optimizer_kwargs = self.cfg.Model.Optimizer_kwargs.to_dict()
        if hasattr(self.model, "get_param_groups"):
            pretrain_lr = optimizer_kwargs.pop("pretrain_lr", 1e-5)
            new_init_lr = optimizer_kwargs.pop("new_init_lr", 1e-4)
            optimizer_kwargs.pop("lr", None)
            param_groups = self.model.get_param_groups(
                pretrain_lr=pretrain_lr, new_init_lr=new_init_lr, **optimizer_kwargs
            )
            self._log.info(
                f"Using parameter groups from model: {len(param_groups)} groups"
            )
        else:
            # 默认行为：使用所有可训练参数
            param_groups = [p for p in self.model.parameters() if p.requires_grad]
            self._log.info("Using all trainable parameters (no parameter groups)")

        self.optimizer = construct_class_by_name(
            **optimizer_kwargs,
            params=param_groups,
        )
        # count the number of parameters (在调度器修改学习率之前调用)
        eu.count_parameters(self)
        self.scheduler = construct_class_by_name(
            **self.cfg.Model.Scheduler_kwargs.to_dict(), optimizer=self.optimizer
        )
        datamodule = construct_class_by_name(
            **self.cfg.Datamodule.kwargs.to_dict(),
            logger=self._log,
            tokenizer=self.model.tokenizer,
            condition_tokenizer=getattr(self.model, "condition_tokenizer", None),
        )
        datamodule.setup(stage="train")
        self.train_loader = datamodule.train_dataloader()
        self.val_loader = datamodule.val_dataloader()
        # Define log steps
        self.val_steps = self.cfg.Trainer.val_every_n_steps
        self.log_every_n_steps = self.cfg.Trainer.log_every_n_steps
        self.save_every_n_steps = self.cfg.Trainer.save_every_n_steps
        self.max_steps = self.cfg.Trainer.max_steps
        self.steps_per_epoch = math.ceil(
            len(self.train_loader) / self.accelerator.num_processes
        )
        self.skip_first_batches = False
        self.save_states_suffix = ""
        if self.cfg.continue_training and self.cfg.not_load_optim:
            ## load weight only model before accelerate preparation.
            if not os.path.exists(
                os.path.join(self.cfg.continue_training, "pytorch_model.bin")
            ):
                raise Exception(
                    f"Checkpoint {self.cfg.continue_training} does not exist. Please gather the paramter of the model before training."
                )
            self._log.info(
                "loading checkpoint (weight only) from {}".format(
                    self.cfg.continue_training
                )
            )
            self.load_state_from_checkpoints(
                os.path.join(self.cfg.continue_training, "pytorch_model.bin")
            )

            res = torch.load(f"{self.cfg.continue_training}/training_state.pth")
            self.trained_steps = res["trained_steps"]
            self.trained_epochs = res["trained_epochs"]
            # skip batches
            self._log.info(
                f"Skipping {self.trained_epochs} epoches and {self.trained_steps} batches..."
            )
            # add datetime to the output dir
            self.save_states_suffix = f"_date-{eu.get_date()}"

        # To moniter trained time costs
        self.time_per_step = TimePerStep(
            accumulate_grad_batches=self.acumulate_grad_batches
        ).to(self.device)
        if self.cfg.Model.supports_gradient_checkpointing:
            for child_model in self.model.children():
                child_model.gradient_checkpointing_enable()

        # configure object and set metrics
        self.model.set_objective_and_metrics(stage="train", experiment=self)
        if (
            self.accelerator.state.deepspeed_plugin is not None
            and self.train_loader.batch_size is None
        ):
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = 1  # set to 1 to avoid the error of deepspeed
        # accelerate prepraration
        self.model, self.optimizer, self.scheduler, self.train_loader = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, self.train_loader
            )
        )
        if isinstance(self.val_loader, list):
            self.val_loader = [
                self.accelerator.prepare(val_loader) for val_loader in self.val_loader
            ]
        else:
            self.val_loader = self.accelerator.prepare(self.val_loader)

        if self.cfg.continue_training and not self.cfg.not_load_optim:
            self._log.info(
                "loading checkpoint from {}".format(self.cfg.continue_training)
            )
            self.skipped_dataloader = self.load_states()
            # add datetime to the output dir
            self.save_states_suffix = f"_date-{eu.get_date()}"

    def update_fn(self, idx: int, batch: dict) -> dict:
        """Updates the state using some data and returns metrics."""
        self.optimizer.zero_grad()

        res = self.loss_fn(idx, batch)
        self.accelerator.backward(res["loss"])
        self.optimizer.step()
        self.scheduler.step()
        return res

    def loss_fn(self, idx: int, batch: dict) -> dict:
        # the loss should be computed in the model
        try:
            res = self.model(**batch)
        except Exception as e:
            print(f"WARNING: Error in loss_fn: {e}")
            torch.save(batch, f"error_batch_{idx}.pth")
            print(f"WARNING: Batch saved to error_batch_{idx}.pth")
            raise e
        if torch.isnan(res["loss"]):
            raise Exception("train loss NaN encountered")
        return res

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> bool:
        self.model.train()

        for idx, batch in enumerate(train_loader):
            begin_time = time.time()
            with self.accelerator.accumulate(self.model):
                batch = tree.map_structure(
                    lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x,
                    batch,
                )
                res = self.update_fn(idx, batch)
            end_time = time.time()

            # update metrics
            self.update_metrics(res, stage="train")

            self.time_per_step.update(end_time - begin_time)
            # Logging
            if idx % self.acumulate_grad_batches == 0 and idx != 0:
                self.trained_steps += 1
                log_lr = {}
                ## Logging learning rate
                for idx, param_groups in enumerate(self.optimizer.param_groups):
                    log_lr[f"optimizer/group_{idx}"] = param_groups["lr"]
                self.accelerator.log(log_lr, step=self.trained_steps)

                ## Logging train metrics
                if self.trained_steps % self.log_every_n_steps == 0:
                    reported_metrics = self.calculate_metrics(stage="train")

                    time_cost = self.time_per_step.compute()
                    self.time_per_step.reset()
                    log_metrics = {
                        **reported_metrics,
                        "time_per_step": float(time_cost),
                        "epoch": self.trained_epochs,
                    }

                    self._log.info_dic_step(log_metrics, step=self.trained_steps)
                    self.accelerator.log(log_metrics, step=self.trained_steps)

                ## Logging validation metrics
                if self.trained_steps % self.val_steps == 0:
                    self._log.info("Staring validation ...")

                    log_metrics = self.val_epoch_warpper(self.val_loader)
                    self._log.info_dic_step(log_metrics, step=self.trained_steps)
                    self.accelerator.log(log_metrics, step=self.trained_steps)
                    self.model.train()

                if self.trained_steps % self.save_every_n_steps == 0:
                    self.save_states()
                    self.accelerator.wait_for_everyone()
                if self.trained_steps == self.max_steps:
                    return True  # finish training
        return False  # False means continue

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
            # update metrics
            self.update_metrics(res, stage="val")

        # log validation metrics
        reported_metrics = self.calculate_metrics(stage="val")
        log_metrics = {**reported_metrics, "epoch": self.trained_epochs}

        return log_metrics

    @torch.no_grad()
    def test_epoch(self, test_loader: torch.utils.data.DataLoader) -> dict:
        self.model.eval()
        for idx, batch in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc="testing...",
            disable=not self.is_main_process,
        ):
            batch = tree.map_structure(
                lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x, batch
            )
            res = self.loss_fn(idx, batch)
            # update metrics
            self.update_metrics(res, stage="test")

        reported_metrics = self.calculate_metrics(stage="test")
        log_metrics = {**reported_metrics, "epoch": self.trained_epochs}

        return log_metrics

    def val_epoch_warpper(
        self, val_loader_list: list, val_name_list: list = None
    ) -> dict:
        log_metrics = {}
        if not isinstance(val_loader_list, list):
            val_loader_list = [val_loader_list]
        if val_name_list is None:
            val_name_list = [f"_{idx}" for idx in range(len(val_loader_list))]
            val_name_list[0] = ""  # overwrite the first one
        for val_loader, val_name in zip(val_loader_list, val_name_list):
            log_metrics.update(self.val_epoch(val_loader, val_name))
        return log_metrics

    def start_testing(self) -> None:
        self.get_test_loader()
        self._log.info("Starting testing...")
        log_metrics = self.test_epoch(self.test_loader)
        self._log.info_dic_step(log_metrics, step=self.trained_steps)
        # self.accelerator.log(log_metrics, step=int(self.trained_steps / self.acumulate_grad_batches))
        self._log.info("Done")

    def start_validation(self) -> None:
        self._log.info("Starting validation...")
        # Do validation to log the performance at the begenning
        log_metrics = self.val_epoch_warpper(self.val_loader)
        self._log.info_dic_step(log_metrics, step=self.trained_steps)
        self.accelerator.log(
            log_metrics, step=int(self.trained_steps / self.acumulate_grad_batches)
        )

    def start_training(self) -> None:
        self._log.info("Starting training...")
        # Do validation to log the performance at the begenning
        if self.cfg.Trainer.val_before_train:
            log_metrics = self.val_epoch_warpper(self.val_loader)
            self._log.info_dic_step(log_metrics, step=self.trained_steps)
            self.accelerator.log(
                log_metrics, step=int(self.trained_steps / self.acumulate_grad_batches)
            )
        isfinish = False or self.trained_epochs == self.max_epochs
        while not isfinish:
            if getattr(self, "skipped_dataloader", None):
                cur_train_loader = self.skipped_dataloader
                self.skipped_dataloader = None
            else:
                cur_train_loader = self.train_loader
            isfinish = self.train_epoch(cur_train_loader)
            self.trained_epochs += 1
            isfinish = isfinish or self.trained_epochs == self.max_epochs

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        self._log.info("Done")

    def load_states(
        self, _path: str = None, weight_only: bool = False
    ) -> Optional[torch.utils.data.DataLoader]:
        if _path is None:
            _path = self.cfg.continue_training
        self.accelerator.load_state(_path)
        if weight_only:
            return
        res = torch.load(f"{_path}/training_state.pth")
        self.trained_steps = res["trained_steps"]
        self.trained_epochs = res["trained_epochs"]
        # skip batches
        self._log.info(
            f"Skipping {self.trained_epochs} epoches and {self.trained_steps} batches..."
        )
        skipped_steps = (
            self.trained_steps * self.acumulate_grad_batches
        ) % self.steps_per_epoch
        skipped_dataloader = self.accelerator.skip_first_batches(
            self.train_loader, skipped_steps
        )
        return skipped_dataloader

    def save_states(self) -> None:
        output_dir = os.path.join(self.cfg.Settings.outdir, "IntervalCheckpoints")
        self._log.info(f"Saving to {output_dir}")
        output_path = os.path.join(
            output_dir, f"step={self.trained_steps}{self.save_states_suffix}"
        )
        self.accelerator.save_state(output_dir=output_path, safe_serialization=False)
        save_dict = {
            "trained_steps": self.trained_steps,
            "trained_epochs": self.trained_epochs,
        }
        if self.accelerator.is_main_process:
            torch.save(save_dict, output_path + "/training_state.pth")
        self.accelerator.wait_for_everyone()

import os
import pickle
import time

import torch
import torch.distributed as dist

# from dataloader.datamodules.multitask_datamodule import MTDataModule
# from dataloader.datasets import TextDataset
import yaml
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

mask_strategy_dict = {
    "woAA-partialstructure": 0,
    "woAA-fullstructure": 1,
    "partialAA-wostructure": 2,
    "partialAA-partialstructure": 3,
    "partialAA-fullstructure": 4,
}


def on_main_process(func):
    def wrapper(*args, **kwargs):
        if (
            not dist.is_initialized() or dist.get_rank() == 0
        ):  # Check if this is the main process
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def timer(func):
    """装饰器：记录函数执行时间

    使用示例:
        @timer
        def my_function(logger=None):
            # 你的代码
            pass
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 检查是否是实例方法，如果是则显示类名
        if args and hasattr(args[0], "__class__"):
            class_name = type(args[0]).__name__
            func_name = f"{class_name}.{func.__name__}"
        else:
            func_name = func.__name__

        message = f"[Timer] {func_name} cost time: {elapsed_time:.4f} s"
        if "logger" in kwargs and kwargs["logger"] is not None:
            kwargs["logger"].info(message)
        elif hasattr(args[0], "_log") and args[0]._log is not None:
            args[0]._log.info(message)
        else:
            print(message)
        return result

    return wrapper


def get_date():
    import datetime

    return datetime.datetime.now().strftime("%m-%d")


def calculate_accuracy(logits, target):
    preds = logits.argmax(dim=-1)
    preds = preds[target != -100]
    target = target[target != -100]
    if target.numel() == 0:
        return 1
    assert preds.shape == target.shape
    correct = torch.sum(preds == target)
    total = target.numel()
    return correct / total


def wapper_save_top_3_checkpoint(exp, step, loss):
    # check if is deepspeed zero3
    if exp.zero_stage == 3:
        save_top_3_zero3_checkpoint(exp, step, loss)
    else:
        save_top_3_checkpoint(exp, step, loss)


def save_top_3_zero3_checkpoint(exp, step, loss):
    output_dir = os.path.join(
        exp.cfg.io["outdir"], "BestCheckpoints", f"step={step}_loss={round(loss, 4)}"
    )
    exp.model.save_checkpoint(output_dir, "pytorch_model")
    exp._log.info(
        f"DeepSpeed Model and Optimizer saved to output dir {os.path.join(output_dir, 'pytorch_model')}"
    )


@on_main_process
def save_top_3_checkpoint(exp, step, loss):
    output_dir = os.path.join(exp.cfg.io["outdir"], "BestCheckpoints")
    output_path = os.path.join(output_dir, f"step={step}_loss={round(loss, 4)}.pth")
    exp._log.info(
        f"saving checkpoint at step {step} to {output_path} for the top-3 metric {loss}"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_checkpoint(
        ckpt_path=output_path,
        state_dict=exp.model.state_dict(),
        conf=exp.cfg,
        optimizer=None,
        epoch=exp.trained_epochs,
        step=step,
        lr_scheduler=exp.scheduler.state_dict(),
    )


@on_main_process
def save_checkpoint_with_step(exp, step):
    output_dir = os.path.join(exp.cfg.io["outdir"], "IntervalCheckpoints")
    output_path = os.path.join(output_dir, f"step={step}.pth")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    exp._log.info(f"saving checkpoint at step {step} to {output_path}")
    write_checkpoint(
        ckpt_path=output_path,
        state_dict=exp.model.state_dict(),
        conf=exp.cfg,
        optimizer=exp.optimizer.state_dict(),
        epoch=exp.trained_epochs,
        step=step,
        lr_scheduler=exp.scheduler.state_dict(),
    )


def count_parameters(exp):
    model = exp.model

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_to_lr = {}
    if hasattr(exp, "optimizer") and exp.optimizer is not None:
        for group_idx, param_group in enumerate(exp.optimizer.param_groups):
            lr = param_group.get("lr", "N/A")
            for param in param_group["params"]:
                param_to_lr[id(param)] = lr

    headers = ["Layer / Module", "Params (M)", "Trainable", "LR"]
    table_data = []

    table_data.append(["[Total]", f"{total_params / 1e6:.4f}M", "Mixed", "-"])
    table_data.append(["[Trainable]", f"{trainable_params / 1e6:.4f}M", "True", "-"])
    table_data.append(["-" * 20, "-" * 10, "-" * 10, "-" * 10])  # 分割线

    for name, module in model.named_children():
        if hasattr(module, "_get_name") and module._get_name() == "Metric":
            continue

        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            is_trainable = any(p.requires_grad for p in module.parameters())

            module_lrs = set()
            for param in module.parameters():
                if param.requires_grad and id(param) in param_to_lr:
                    module_lrs.add(param_to_lr[id(param)])

            if len(module_lrs) == 0:
                lr_str = "-"
            elif len(module_lrs) == 1:
                lr = list(module_lrs)[0]
                lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else str(lr)
            else:
                lrs_sorted = sorted(
                    [lr for lr in module_lrs if isinstance(lr, (int, float))]
                )
                if lrs_sorted:
                    lr_str = f"{lrs_sorted[0]:.2e}~{lrs_sorted[-1]:.2e}"
                else:
                    lr_str = "Mixed"

            table_data.append([name, f"{params / 1e6:.4f}M", str(is_trainable), lr_str])

    if tabulate is not None:
        table_str = tabulate(
            table_data,
            headers=headers,
            tablefmt="psql",
            colalign=("center", "center", "center", "center"),
        )
    else:
        rows = [headers] + table_data
        col_widths = [
            max(len(str(row[col_idx])) for row in rows) for col_idx in range(len(headers))
        ]
        formatted_rows = []
        for row in rows:
            formatted_rows.append(
                " | ".join(
                    str(value).center(col_widths[idx]) for idx, value in enumerate(row)
                )
            )
        table_str = "\n".join(formatted_rows)

    lr_summary = ""
    if (
        hasattr(exp, "optimizer")
        and exp.optimizer is not None
        and len(exp.optimizer.param_groups) > 1
    ):
        lr_summary = "\n\nParameter Groups Learning Rates:\n"
        for group_idx, param_group in enumerate(exp.optimizer.param_groups):
            lr = param_group.get("lr", "N/A")
            num_params = sum(p.numel() for p in param_group["params"])
            lr_summary += f"  Group {group_idx}: LR = {lr:.2e}, Params = {num_params / 1e6:.4f}M\n"

    exp._log.info(f"\nModel Summary:\n{table_str}{lr_summary}")


def log_config(logger, config):
    # 如果 config 是 argparse.Namespace 或其他对象，先转成 dict
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif hasattr(config, "__dict__"):
        config_dict = vars(config)
    else:
        config_dict = config

    # dump 为 yaml 字符串
    # sort_keys=False 保持你字典原有的顺序 (Model -> Optimizer...) 而不是按字母排序
    yaml_str = yaml.dump(config_dict, sort_keys=False, indent=2, allow_unicode=True)

    logger.info(
        f"\n{'=' * 20} Experiment Configuration {'=' * 20}\n{yaml_str}{'=' * 66}"
    )


def write_pkl(save_path, pkl_data, create_dir=False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path)
    else:
        with open(save_path, "wb") as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_checkpoint(
    ckpt_path: str,
    state_dict,
    conf,
    optimizer=None,
    epoch=None,
    step=None,
    lr_scheduler=None,
    use_torch=True,
):
    """Serialize experiment state and stats to a pickle file.

    Args:
        ckpt_path: Path to save checkpoint.
        conf: Experiment configuration.
        optimizer: Optimizer state dict.
        epoch: Training epoch at time of checkpoint.
        step: Training steps at time of checkpoint.
        exp_state: Experiment state to be written to pickle.
        preds: Model predictions to be written as part of checkpoint.
    """
    write_pkl(
        save_path=ckpt_path,
        pkl_data={
            "state_dict": state_dict,
            "conf": conf,
            "optimizer": optimizer,
            "epoch": epoch,
            "step": step,
            "lr_scheduler": lr_scheduler,
        },
        use_torch=use_torch,
    )

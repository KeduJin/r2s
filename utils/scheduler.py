import math

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_with_hard_min_lr_schedule(
    optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1
):
    """
    num_training_steps: 衰减停止的步数（即进入平稳期的步数）
    min_lr_ratio: 最终学习率占初始学习率的比例 (例如 0.1)
    """

    def lr_lambda(current_step):
        # 1. Warmup 阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 2. 超过总步数后，强制返回 min_lr_ratio，不再反弹，不再下降
        if current_step >= num_training_steps:
            return min_lr_ratio

        # 3. Cosine 衰减阶段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # 将 cosine 结果映射到 [min_lr_ratio, 1.0] 范围内
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# domain token takes about 75% of the total tokens
domain_ratio = 0.75
non_domain_ratio = 0.25


class DomainReweightedCriterion(nn.Module):
    """
    自定义criterion，对domain区域和非domain区域应用不同的loss权重
    domain区域: weight = 1
    非domain区域: weight = 10
    适用于GPT模型的loss计算
    """

    def __init__(
        self, domain_weight: float = 1.0, non_domain_weight: float = 1.0, **kwargs
    ):
        super().__init__()
        self.domain_weight = domain_weight
        self.non_domain_weight = non_domain_weight
        self.logger = kwargs.get("logger", None)
        if self.logger is not None:
            self.logger.info(
                f"DomainReweightedCriterion initialized with domain_weight: {domain_weight}, non_domain_weight: {non_domain_weight}"
            )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        domain_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            logits: [bs, seq_len, vocab_size] 模型输出
            target: [bs, seq_len] 目标序列，-100表示忽略的位置
            domain_mask: [bs, seq_len] domain区域掩码，1表示domain区域，0表示非domain区域
        """

        sample_size = target.ne(-100).float().sum()

        nonpad_ratio = sample_size / target.numel()
        # 计算基础loss (CrossEntropyLoss with ignore_index=-100)
        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_target = shift_labels.view(-1)
        flat_loss = loss_fct(flat_logits, flat_target)
        flat_loss = flat_loss.view(flat_target.size())

        # 创建有效token的mask (非-100的位置)
        valid_mask = (flat_target != -100).float()

        # 创建domain权重掩码
        if domain_mask is not None:
            domain_mask = domain_mask[:, 1:].reshape(
                -1
            )  # shift the domain_mask by 1 and flatten
            # domain_mask: 1表示domain区域，0表示非domain区域
            domain_weight_mask = torch.where(
                domain_mask.bool(),  # shift the domain_mask by 1
                torch.tensor(self.domain_weight, device=domain_mask.device),
                torch.tensor(self.non_domain_weight, device=domain_mask.device),
            )
        else:
            # 如果没有提供domain_mask，使用默认权重
            domain_weight_mask = torch.ones_like(flat_loss) * self.domain_weight

        # 应用domain权重
        weighted_loss = flat_loss * domain_weight_mask

        # 只对有效token计算loss
        final_loss = weighted_loss * valid_mask

        # 计算总loss
        total_loss = final_loss.sum() / (valid_mask.sum() + 1e-8)
        total_weight = (
            domain_ratio * self.domain_weight
            + non_domain_ratio * self.non_domain_weight
        )
        total_loss = total_loss / total_weight

        # 分别计算domain和非domain区域的loss用于logging
        if domain_mask is not None:
            # domain区域的loss
            domain_valid_mask = valid_mask * domain_mask
            domain_loss = (flat_loss * domain_mask * valid_mask).sum() / (
                domain_valid_mask.sum() + 1e-8
            )

            # 非domain区域的loss
            non_domain_valid_mask = valid_mask * (1 - domain_mask)
            non_domain_loss = (flat_loss * (1 - domain_mask) * valid_mask).sum() / (
                non_domain_valid_mask.sum() + 1e-8
            )
        else:
            domain_loss = total_loss
            non_domain_loss = torch.tensor(0.0, device=total_loss.device)

        # 准备logging输出
        logging_output = {
            "domain_loss": domain_loss,
            "non_domain_loss": non_domain_loss,
            "domain_weight": self.domain_weight,
            "non_domain_weight": self.non_domain_weight,
            "unweighted_loss": (flat_loss * valid_mask).sum()
            / (valid_mask.sum() + 1e-8),
            "nonpad_ratio": nonpad_ratio,
            "sample_size": sample_size,
        }

        return total_loss, logging_output


"""copy from byprot.modules.cross_entropy.RDMCrossEntropyLoss
and modified the loss calculation to use domain weights
"""


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    flag = False
    if target.dim() == lprobs.dim() - 1:
        flag = True
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)

    if flag:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class RDMCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self, domain_weight: float = 1.0, non_domain_weight: float = 1.0, **kwargs
    ):
        self.domain_weight = domain_weight
        self.non_domain_weight = non_domain_weight
        super().__init__(**kwargs)

    def forward(
        self,
        scores: torch.Tensor,
        target: torch.Tensor,
        label_mask=None,
        weights=None,
        cal_constant_loss=False,
        watch_t1_t2_loss=False,
        domain_mask=None,
    ) -> torch.Tensor:
        """
        scores: [N, L, C], unnormalized scores
        target: [N, L]
        coord_mask: FloatTensor [N, L], where elements with `True` are allowed and `False` are masked-out
        """
        bsz = scores.shape[0]

        n_tokens = target.numel()
        if self.ignore_index is not None:
            sample_size = n_nonpad_tokens = target.ne(self.ignore_index).float().sum()
        else:
            sample_size = n_nonpad_tokens = n_tokens
        # [N, L]
        loss, _ = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),
            target=target,
            epsilon=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduce=False,
        )
        if domain_mask is not None:
            domain_loss = loss * domain_mask * self.domain_weight
            non_domain_loss = loss * (1 - domain_mask) * self.non_domain_weight
            reweighted_loss = domain_loss + non_domain_loss
        else:
            domain_loss = torch.tensor(
                0.0, device=loss.device, dtype=loss.dtype
            )  # for logging
            non_domain_loss = torch.tensor(
                0.0, device=loss.device, dtype=loss.dtype
            )  # for logging
            reweighted_loss = loss

        if weights is not None:
            loss = loss * weights
            reweighted_loss = reweighted_loss * weights
            domain_loss = domain_loss * weights
            non_domain_loss = non_domain_loss * weights

        fullseq_loss = loss.sum() / sample_size

        t1_loss, t2_loss = None, None
        if watch_t1_t2_loss:
            t1_loss, t2_loss = loss.chunk(2)
            t1_mask, t2_mask = label_mask.chunk(2)
            t1_loss = (t1_loss * t1_mask).sum() / (t1_mask.sum())
            t2_loss = (t2_loss * t2_mask).sum() / (t2_mask.sum())

        # use coord masked loss for model training,
        # ignoring those position with missing coords (as nan)
        if label_mask is not None:
            label_mask = label_mask.float()
            sample_size = (
                label_mask.sum()
            )  # sample size should be set to valid coordinates
            loss = (loss * label_mask).sum() / sample_size
            domain_loss = (domain_loss * label_mask).sum() / sample_size
            non_domain_loss = (non_domain_loss * label_mask).sum() / sample_size
            reweighted_loss = (reweighted_loss * label_mask).sum() / sample_size
        else:
            loss = fullseq_loss
            domain_loss = torch.tensor(
                0.0, device=loss.device, dtype=loss.dtype
            )  # for logging
            non_domain_loss = torch.tensor(
                0.0, device=loss.device, dtype=loss.dtype
            )  # for logging
            reweighted_loss = loss

        ppl = torch.exp(loss)

        logging_output = {
            "ppl": ppl.data,
            "fullseq_loss": fullseq_loss.data,
            "bsz": bsz,
            "sample_size": sample_size,
            "sample_ratio": sample_size / n_tokens,
            "nonpad_ratio": n_nonpad_tokens / n_tokens,
            "weight_diff_loss": reweighted_loss.data,
            "domain_loss": domain_loss.data,
            "non_domain_loss": non_domain_loss.data,
        }

        if cal_constant_loss:
            constant_weights = weights.new_ones(size=weights.size())
            constant_loss, _ = label_smoothed_nll_loss(
                lprobs=F.log_softmax(scores, dim=-1),
                target=target,
                epsilon=self.label_smoothing,
                ignore_index=self.ignore_index,
                reduce=False,
            )
            constant_loss = constant_loss * constant_weights
            constant_loss = (constant_loss * label_mask).sum() / sample_size
            logging_output["constant_diff_loss"] = constant_loss.data

        if watch_t1_t2_loss:
            logging_output["weight_diff_t1_loss"] = t1_loss.data
            logging_output["weight_diff_t2_loss"] = t2_loss.data

        return reweighted_loss, logging_output

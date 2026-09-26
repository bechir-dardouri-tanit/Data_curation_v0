"""DPO and SimPO loss implementations.

Implements the core loss functions for preference optimization:

- DPO: Direct Preference Optimization loss
- SimPO: Simple Preference Optimization loss
- Length-controlled variants
- SFT-loss component (RPO-style)

Reference implementations:
    DPO: https://arxiv.org/abs/2305.18290
    SimPO: https://arxiv.org/abs/2405.14734
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - PyTorch convention
from torch import Tensor

from medrl.train.pref.config import LengthNorm, PrefOptMethod

if TYPE_CHECKING:
    from medrl.train.pref.config import PrefOptConfig


class LengthNormStrategy(StrEnum):
    """Internal strategies for length normalization."""

    NONE = "none"
    MEAN = "mean"
    SIMPO = "simpo"


@dataclass(frozen=True)
class PrefLossOutput:
    """Output from preference loss computation.

    Attributes:
        loss: The total loss (backwarded).
        policy_loss: The preference loss component.
        sft_loss: The SFT (cross-entropy) loss component.
        chosen_logps: Log probabilities of chosen responses.
        rejected_logps: Log probabilities of rejected responses.
        chosen_lengths: Lengths of chosen responses (for logging).
        rejected_lengths: Lengths of rejected responses (for logging).
        accuracy: Fraction of pairs where chosen has higher reward.
    """

    loss: Tensor
    policy_loss: Tensor
    sft_loss: Tensor
    chosen_logps: Tensor
    rejected_logps: Tensor
    chosen_lengths: Tensor
    rejected_lengths: Tensor
    accuracy: Tensor


def compute_log_probs(
    logits: Tensor,  # [batch, seq_len, vocab_size]
    labels: Tensor,  # [batch, seq_len]
    mask: Tensor,  # [batch, seq_len]
) -> tuple[Tensor, Tensor]:
    """Compute log probabilities and mean length.

    Args:
        logits: Model output logits.
        labels: Target token IDs (with -100 for ignored positions).
        mask: Boolean mask for valid positions.

    Returns:
        (mean_log_prob, mean_length) where:
        - mean_log_prob: Average log probability per token
        - mean_length: Average sequence length
    """
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]

    # Gather log probs for target tokens
    vocab_size = log_probs.shape[-1]
    # Expand labels to match vocab dimension for gathering
    labels_expanded = labels.unsqueeze(-1).expand(-1, -1, vocab_size)
    target_log_probs = torch.gather(log_probs, dim=-1, index=labels_expanded)
    target_log_probs = target_log_probs.squeeze(-1)  # [batch, seq_len]

    # Apply mask and sum
    masked_log_probs = target_log_probs * mask.float()  # [batch, seq_len]
    sum_log_probs = masked_log_probs.sum(dim=-1)  # [batch]

    # Compute lengths
    lengths = mask.sum(dim=-1).float()  # [batch]
    mean_length = lengths.mean()

    # Mean log prob per sequence
    mean_log_prob = sum_log_probs / lengths.clamp(min=1.0)

    return mean_log_prob, mean_length


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Compute DPO loss.

    Args:
        policy_chosen_logps: Policy model log probs for chosen responses.
        policy_rejected_logps: Policy model log probs for rejected responses.
        reference_chosen_logps: Reference model log probs for chosen responses.
        reference_rejected_logps: Reference model log probs for rejected responses.
        beta: Temperature parameter.
        label_smoothing: Label smoothing factor (0 = no smoothing).

    Returns:
        DPO loss tensor.
    """
    # Compute log ratios
    policy_lograt = policy_chosen_logps - policy_rejected_logps
    reference_lograt = reference_chosen_logps - reference_rejected_logps

    # DPO loss: -log(sigmoid(beta * (policy_lograt - reference_lograt)))
    # With label smoothing: -log((1 - smoothing) * sigmoid(...) + smoothing / 2)
    logits = beta * (policy_lograt - reference_lograt)

    if label_smoothing > 0:
        # Label-smoothed DPO loss
        losses = -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
    else:
        losses = -F.logsigmoid(logits)

    return losses.mean()


def simpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    chosen_lengths: Tensor,
    rejected_lengths: Tensor,
    beta: float,
    gamma: float = 0.5,
) -> Tensor:
    """Compute SimPO loss.

    SimPO replaces the reference model with a length-based baseline:
        loss = -log(sigmoid(beta * (lograt_chosen - lograt_rejected - gamma * length_ratio)))

    Args:
        policy_chosen_logps: Policy model log probs for chosen responses.
        policy_rejected_logps: Tensor.
        policy_rejected_logps: Policy model log probs for rejected responses.
        chosen_lengths: Lengths of chosen responses.
        rejected_lengths: Lengths of rejected responses.
        beta: Temperature parameter.
        gamma: Length penalty coefficient (default 0.5).

    Returns:
        SimPO loss tensor.
    """
    # Compute log ratios
    policy_lograt = policy_chosen_logps - policy_rejected_logps

    # Length-based penalty
    length_ratio = chosen_lengths.float() / (rejected_lengths.float() + 1e-8)
    length_penalty = gamma * length_ratio

    # SimPO loss
    logits = beta * (policy_lograt - length_penalty)
    losses = -F.logsigmoid(logits)

    return losses.mean()


def sft_loss(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
) -> Tensor:
    """Compute standard SFT (cross-entropy) loss on chosen responses.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size].
        labels: Target token IDs [batch, seq_len].
        mask: Boolean mask for valid positions [batch, seq_len].

    Returns:
        Cross-entropy loss tensor.
    """
    # Shift for causal LM: predict next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()

    # Compute loss per position
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )

    # Mask out ignored positions
    loss = loss.view(shift_labels.shape) * shift_mask.float()

    # Return mean loss
    return loss.sum() / shift_mask.sum().clamp(min=1.0)


class PrefLoss(nn.Module):
    """Preference optimization loss module.

    Combines DPO/SimPO preference loss with optional SFT loss component.

    Example:
        >>> loss_fn = PrefLoss(config)
        >>> output = loss_fn(
        ...     policy_chosen_logits, policy_rejected_logits,
        ...     ref_chosen_logits, ref_rejected_logits,
        ...     chosen_labels, rejected_labels,
        ...     chosen_mask, rejected_mask
        ... )
        >>> output.loss.backward()
    """

    def __init__(
        self,
        config: PrefOptConfig,
        label_smoothing: float = 0.0,
        simpo_gamma: float = 0.5,
    ) -> None:
        """Initialize the loss module.

        Args:
            config: Preference optimization configuration.
            label_smoothing: Label smoothing factor for DPO.
            simpo_gamma: Length penalty coefficient for SimPO.
        """
        super().__init__()
        self.config = config
        self.label_smoothing = label_smoothing
        self.simpo_gamma = simpo_gamma

        # Determine normalization strategy
        self.norm_strategy = LengthNormStrategy.MEAN
        if config.length_norm == LengthNorm.NONE:
            self.norm_strategy = LengthNormStrategy.NONE
        elif config.length_norm == LengthNorm.SIMPO:
            self.norm_strategy = LengthNormStrategy.SIMPO

    def forward(
        self,
        policy_chosen_logits: Tensor,
        policy_rejected_logits: Tensor,
        ref_chosen_logits: Tensor,
        ref_rejected_logits: Tensor,
        chosen_labels: Tensor,
        rejected_labels: Tensor,
        chosen_mask: Tensor,
        rejected_mask: Tensor,
    ) -> PrefLossOutput:
        """Compute the full preference loss.

        Args:
            policy_chosen_logits: Policy model logits for chosen responses.
            policy_rejected_logits: Policy model logits for rejected responses.
            ref_chosen_logits: Reference model logits for chosen responses.
            ref_rejected_logits: Reference model logits for rejected responses.
            chosen_labels: Target token IDs for chosen responses.
            rejected_labels: Target token IDs for rejected responses.
            chosen_mask: Mask for valid positions in chosen responses.
            rejected_mask: Mask for valid positions in rejected responses.

        Returns:
            PrefLossOutput with all loss components and metrics.
        """
        # Compute log probabilities
        policy_chosen_logps, chosen_len = compute_log_probs(
            policy_chosen_logits, chosen_labels, chosen_mask
        )
        policy_rejected_logps, rejected_len = compute_log_probs(
            policy_rejected_logits, rejected_labels, rejected_mask
        )

        ref_chosen_logps, _ = compute_log_probs(
            ref_chosen_logits, chosen_labels, chosen_mask
        )
        ref_rejected_logps, _ = compute_log_probs(
            ref_rejected_logits, rejected_labels, rejected_mask
        )

        # Compute preference loss based on method
        if self.config.method == PrefOptMethod.SimPO:
            # SimPO doesn't use reference model
            policy_loss = simpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                chosen_len,
                rejected_len,
                self.config.beta,
                self.simpo_gamma,
            )
        else:
            # DPO uses reference model
            policy_loss = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                self.config.beta,
                self.label_smoothing,
            )

        # Optional SFT loss component (RPO-style)
        sft_loss_value = torch.tensor(0.0, device=policy_chosen_logits.device)
        if self.config.sft_loss_weight > 0:
            sft_loss_value = sft_loss(
                policy_chosen_logits,
                chosen_labels,
                chosen_mask,
            )

        # Combine losses
        total_loss = policy_loss + self.config.sft_loss_weight * sft_loss_value

        # Compute accuracy (chosen has higher reward than rejected)
        with torch.no_grad():
            if self.config.method == PrefOptMethod.SimPO:
                # SimPO reward: log prob - gamma * length penalty
                chosen_reward = policy_chosen_logps - self.simpo_gamma * chosen_len.float()
                rejected_reward = policy_rejected_logps - self.simpo_gamma * rejected_len.float()
            else:
                # DPO reward: policy log prob - reference log prob
                chosen_reward = policy_chosen_logps - ref_chosen_logps
                rejected_reward = policy_rejected_logps - ref_rejected_logps

            accuracy = (chosen_reward > rejected_reward).float().mean()

        return PrefLossOutput(
            loss=total_loss,
            policy_loss=policy_loss.detach(),
            sft_loss=sft_loss_value.detach(),
            chosen_logps=policy_chosen_logps.detach(),
            rejected_logps=policy_rejected_logps.detach(),
            chosen_lengths=chosen_len.detach(),
            rejected_lengths=rejected_len.detach(),
            accuracy=accuracy.detach(),
        )


def compute_grad_norm(
    model: nn.Module,
) -> float:
    """Compute gradient norm for logging.

    Args:
        model: The model to compute gradient norm for.

    Returns:
        The total gradient norm.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0
    return torch.nn.utils.clip_grad_norm_(parameters, float("inf")).item()

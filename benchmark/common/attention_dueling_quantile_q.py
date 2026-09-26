#!/usr/bin/env python3
"""Permutation-equivariant dueling quantile Q-network for well sequencing.

This is HyperActive's designated candidate-set scorer architecture
(Transformer-encoder input projection, masked-mean context pooling, dueling
advantage/value heads, QR-DQN-style quantile output). It is retrained from
scratch on each benchmark's own feature vector -- no weights are transferred
between benchmarks.

Requires PyTorch (imported lazily, only inside the functions below, so this
module can still be imported for documentation/tooling purposes in an
environment without PyTorch installed).
"""
from __future__ import annotations


def build_attention_q_network(
    feature_dim: int,
    *,
    d_model: int = 128,
    heads: int = 8,
    layers: int = 3,
    feedforward_dim: int = 256,
    quantiles: int = 32,
    dropout: float = 0.1,
):
    import torch
    from torch import nn

    if d_model % heads:
        raise ValueError("d_model must be divisible by heads")
    if layers < 1 or quantiles < 1:
        raise ValueError("layers and quantiles must be positive")

    class AttentionDuelingQuantileQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_dim = feature_dim
            self.d_model = d_model
            self.quantile_count = quantiles
            self.input_projection = nn.Sequential(
                nn.Linear(feature_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.value = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, quantiles),
            )
            self.advantage = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, quantiles),
            )

        def forward(self, observations, action_mask):
            if observations.ndim != 3 or action_mask.ndim != 2:
                raise ValueError("expected observations [B,N,F] and action_mask [B,N]")
            mask = action_mask.bool()
            encoded = self.input_projection(observations)
            safe_mask = mask.clone()
            empty = ~safe_mask.any(dim=1)
            if empty.any():
                safe_mask[empty, 0] = True
            encoded = self.encoder(encoded, src_key_padding_mask=~safe_mask)
            weights = mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            context = pooled.unsqueeze(1).expand(-1, encoded.shape[1], -1)
            advantage = self.advantage(torch.cat((encoded, context), dim=-1))
            denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
            centered = advantage - (
                advantage * mask.unsqueeze(-1).to(advantage.dtype)
            ).sum(dim=1, keepdim=True) / denominator
            quantile_values = self.value(pooled).unsqueeze(1) + centered
            return quantile_values.masked_fill(
                ~mask.unsqueeze(-1), torch.finfo(quantile_values.dtype).min
            )

        def q_values(self, observations, action_mask, *, anchor_strength: float = 0.0):
            values = self(observations, action_mask).mean(dim=-1)
            if anchor_strength:
                values = values + anchor_strength * observations[:, :, 2]
            return values.masked_fill(
                ~action_mask.bool(), torch.finfo(values.dtype).min
            )

    return AttentionDuelingQuantileQ()


def quantile_huber_loss(
    predicted, target, *, kappa: float = 1.0, reduction: str = "mean"
):
    """QR-DQN loss for tensors shaped [batch, quantiles]."""
    import torch

    if predicted.ndim != 2 or target.ndim != 2:
        raise ValueError("predicted and target must be [batch, quantiles]")
    td = target.unsqueeze(1) - predicted.unsqueeze(2)
    absolute = td.abs()
    huber = torch.where(
        absolute <= kappa,
        0.5 * td.square(),
        kappa * (absolute - 0.5 * kappa),
    )
    quantile_count = predicted.shape[1]
    taus = (
        (torch.arange(quantile_count, device=predicted.device, dtype=predicted.dtype) + 0.5)
        / quantile_count
    ).view(1, quantile_count, 1)
    weights = (taus - (td.detach() < 0).to(predicted.dtype)).abs()
    per_sample = (weights * huber / kappa).mean(dim=(1, 2))
    if reduction == "none":
        return per_sample
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'none'")
    return per_sample.mean()

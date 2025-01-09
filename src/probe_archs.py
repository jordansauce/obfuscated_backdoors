import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from accelerate import find_executable_batch_size
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import f1_score
from torch import nn
from tqdm.auto import tqdm

from .attacks import *
from .probe_training import *


class LinearProbe(Probe):
    # Linear probe for transformer activations

    def __init__(self, d_model):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class QuadraticProbe(Probe):
    # Quadratic probe for transformer activations

    def __init__(self, d_model):
        super(QuadraticProbe, self).__init__()
        self.M = nn.Parameter(torch.randn(d_model, d_model) / d_model**0.5)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        batch_dims = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])
        xM = torch.matmul(x_flat.unsqueeze(1), self.M)
        xMx = torch.matmul(xM, x_flat.unsqueeze(-1))
        linear_term = self.linear(x).squeeze(-1)
        return xMx.squeeze(-1).squeeze(-1).view(*batch_dims) + linear_term


class NonlinearProbe(Probe):
    # Nonlinear probe for transformer activations

    def __init__(self, d_model, d_mlp, dropout=0.1):
        super(NonlinearProbe, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class AttentionProbe(Probe):
    # Attention probe for transformer activations with lower dimensional projection

    def __init__(self, d_model, d_proj, nhead, max_length=8192, sliding_window=None):
        super(AttentionProbe, self).__init__()
        self.d_model = d_model
        self.d_proj = d_proj
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_proj * nhead)
        self.k_proj = nn.Linear(d_model, d_proj * nhead)
        self.v_proj = nn.Linear(d_model, d_proj * nhead)
        self.out_proj = nn.Linear(d_proj * nhead, 1)
        if sliding_window is not None:
            mask = self._construct_sliding_window_mask(max_length, sliding_window)
        else:
            mask = self._construct_causal_mask(max_length)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def _construct_causal_mask(self, seq_len):
        mask = torch.ones(seq_len, seq_len)
        mask = torch.tril(mask, diagonal=0)
        return mask.to(dtype=torch.bool)

    def _construct_sliding_window_mask(self, seq_len, window_size):
        q_idx = torch.arange(seq_len).unsqueeze(1)
        kv_idx = torch.arange(seq_len).unsqueeze(0)
        causal_mask = q_idx >= kv_idx
        windowed_mask = q_idx - kv_idx < window_size
        return causal_mask & windowed_mask

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.nhead, self.d_proj)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.nhead, self.d_proj)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.nhead, self.d_proj)
            .transpose(1, 2)
        )
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.mask[:seq_len, :seq_len]
        )
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        output = self.out_proj(attn_output).squeeze(-1)
        return output


class DirectionalProbe(Probe):
    # Directional probe for transformer activations

    def __init__(self, direction):
        super(DirectionalProbe, self).__init__()
        if direction.dim() == 1:
            direction = direction.unsqueeze(-1)

        # Normalize the direction vector
        direction = direction / torch.norm(direction, dim=0, keepdim=True)
        self.direction = nn.Parameter(direction, requires_grad=False)
        self.magnitude = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )  #  We can train this to calibrate the probe
        self.bias = nn.Parameter(
            torch.tensor(0.0), requires_grad=True
        )  #  We can train this to calibrate the probe

    def forward(self, x):
        return torch.matmul(x, self.direction * self.magnitude).squeeze(-1) + self.bias


class OrthogonalEnsembleProbe(Probe):
    # Multi-probe that trains several nearly orthogonal linear probes

    def __init__(self, d_model, n_probes):
        super(OrthogonalEnsembleProbe, self).__init__()
        self.linear = nn.Linear(d_model, n_probes)
        self.n_probes = n_probes

    def forward(self, x):
        return self.linear(x)

    def compute_loss(self, acts, labels, mask=None):
        # Compute scores for each probe
        scores = self.forward(acts)  # Shape: (..., n_probes)

        # Expand labels to match scores shape for parallel computation
        expanded_labels = labels.unsqueeze(-1).expand(*labels.shape, self.n_probes)
        expanded_labels = expanded_labels.to(device=scores.device, dtype=scores.dtype)

        # Handle masking
        if mask is not None:
            if mask.shape != labels.shape:
                # Check if view operation is possible by comparing total elements
                if mask.numel() != labels.numel():
                    raise ValueError(
                        f"Cannot reshape mask of size {mask.shape} ({mask.numel()} elements) to labels shape {labels.shape} ({labels.numel()} elements)"
                    )
                mask = mask.view(labels.shape)

            # Apply mask
            valid_scores = scores[mask]  # Shape: (n_valid, n_probes)
            valid_labels = expanded_labels[mask]  # Shape: (n_valid, n_probes)

            # Compute loss only on valid positions
            pred_loss = F.binary_cross_entropy_with_logits(
                valid_scores, valid_labels, reduction="mean"
            )
        else:
            pred_loss = F.binary_cross_entropy_with_logits(
                scores, expanded_labels, reduction="mean"
            )

        # Orthogonality regularization: ||M^T M - I||
        eye = torch.eye(
            self.n_probes,
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype,
        )
        weight_product = self.linear.weight @ self.linear.weight.t()
        ortho_loss = torch.norm(weight_product - eye)

        # Combine losses
        total_loss = pred_loss + 0.1 * ortho_loss
        return total_loss

    def predict(self, x):
        probe_probs = torch.sigmoid(self.forward(x))

        # Compute entropy-based weights
        entropy = -(
            probe_probs * torch.log(probe_probs + 1e-10)
            + (1 - probe_probs) * torch.log(1 - probe_probs + 1e-10)
        )
        weights = torch.softmax(-entropy, dim=-1)

        # Weighted geometric mean (in log space for stability)
        log_probs = torch.log(probe_probs + 1e-10)
        return torch.exp((log_probs * weights).sum(dim=-1))


class SubspaceProbe(Probe):
    # Subspace probe for transformer activations that finds an orthonormal basis
    # where positive examples have large projections and negative examples have small projections

    def __init__(self, d_model, subspace_dim, bias=True):
        super(SubspaceProbe, self).__init__()
        self.d_model = d_model
        self.subspace_dim = subspace_dim

        # Initialize random orthonormal basis vectors using SVD to ensure orthonormality from start
        basis = torch.randn(d_model, subspace_dim)
        u, _, _ = torch.linalg.svd(basis, full_matrices=False)
        self.basis = nn.Parameter(u[:, :subspace_dim])

        # Optional bias vector to offset activations before projection
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_model))

        # Temperature parameter for softmax in prediction
        self.temperature = nn.Parameter(torch.ones(1))

    def _ensure_orthonormal(self):
        # Keep basis vectors orthonormal using modified Gram-Schmidt process
        with torch.no_grad():
            u = self.basis.clone()
            for i in range(self.subspace_dim):
                # Normalize the i-th vector to unit length
                u[:, i] = u[:, i] / torch.norm(u[:, i])

                # Remove projection onto u_i from subsequent vectors
                if i < self.subspace_dim - 1:
                    proj = u[:, i : i + 1] @ (u[:, i : i + 1].t() @ u[:, i + 1 :])
                    u[:, i + 1 :] = u[:, i + 1 :] - proj

            self.basis.copy_(u)

    def _compute_projections(self, x):
        # Compute magnitudes of projections onto the subspace and its orthogonal complement

        # Apply optional bias shift to input
        if self.use_bias:
            x = x + self.bias

        # Normalize input vectors to unit length
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / (x_norm + 1e-8)  # Add small epsilon to prevent division by zero

        # Project onto basis vectors
        # Shape: (..., subspace_dim)
        projections = torch.matmul(x, self.basis)

        # Compute squared magnitude of projection onto subspace
        subspace_magnitude = (projections**2).sum(dim=-1)

        # Compute squared magnitude of projection onto orthogonal complement
        orthogonal_magnitude = 1.0 - subspace_magnitude

        return subspace_magnitude, orthogonal_magnitude

    def forward(self, x):
        # Forward pass computes logits for both subspace and orthogonal complement

        # Get projection magnitudes
        subspace_mag, orthogonal_mag = self._compute_projections(x)

        # Stack magnitudes and apply temperature scaling
        logits = torch.stack([subspace_mag, orthogonal_mag], dim=-1)
        logits = logits / self.temperature

        return logits

    def compute_loss(self, acts, labels, mask=None):
        # Loss directly optimizes projection magnitudes:
        # - For positive examples (label=1): maximize projection onto subspace
        # - For negative examples (label=0): minimize projection onto subspace

        # Ensure basis stays orthonormal
        self._ensure_orthonormal()

        # Get projection magnitude onto subspace
        subspace_mag, _ = self._compute_projections(acts)

        # For positive examples: loss = -proj_magnitude (to maximize projection)
        # For negative examples: loss = proj_magnitude (to minimize projection)
        # Convert labels to float and scale to {-1, 1}
        labels = labels.to(dtype=subspace_mag.dtype)
        scale = 2 * labels - 1  # Maps 0->-1, 1->1

        # Compute loss: negative for positives (to maximize), positive for negatives (to minimize)
        loss = -scale * subspace_mag

        # Handle masking consistently with other probe classes
        if mask is not None:
            # Ensure mask shape matches loss shape
            if mask.shape != loss.shape:
                # If mask is flattened, reshape it to match loss
                mask = mask.view(loss.shape)

            # Apply mask
            loss = loss[mask]

            # Take mean of masked elements
            loss = loss.mean()
        else:
            loss = loss.mean()

        return loss

    def predict(self, x):
        # Predict probability of positive class using softmax
        # Higher subspace projection -> higher probability
        logits = self.forward(x)  # Shape: (..., 2)
        return logits[..., 0]  # Return probability for subspace projection


def train_linear_probe(encoder, positive_examples, negative_examples, layers, **kwargs):
    # Train a linear probe for each specified layer
    def create_linear_probe():
        return LinearProbe(encoder.model.config.hidden_size)

    return train_probe(
        encoder,
        positive_examples,
        negative_examples,
        create_linear_probe,
        layers,
        **kwargs,
    )


def train_quadratic_probe(
    encoder, positive_examples, negative_examples, layers, **kwargs
):
    # Train a quadratic probe for each specified layer
    def create_quadratic_probe():
        return QuadraticProbe(encoder.model.config.hidden_size)

    return train_probe(
        encoder,
        positive_examples,
        negative_examples,
        create_quadratic_probe,
        layers,
        **kwargs,
    )


def train_nonlinear_probe(
    encoder, positive_examples, negative_examples, d_mlp, layers, **kwargs
):
    # Train a nonlinear probe for each specified layer
    def create_nonlinear_probe():
        return NonlinearProbe(encoder.model.config.hidden_size, d_mlp)

    return train_probe(
        encoder,
        positive_examples,
        negative_examples,
        create_nonlinear_probe,
        layers,
        **kwargs,
    )


def train_attention_probe(
    encoder,
    positive_examples,
    negative_examples,
    d_proj,
    nhead,
    sliding_window,
    layers,
    **kwargs,
):
    # Train an attention probe for each specified layer
    def create_attention_probe():
        return AttentionProbe(
            encoder.model.config.hidden_size,
            d_proj,
            nhead,
            sliding_window=sliding_window,
        )

    return train_probe(
        encoder,
        positive_examples,
        negative_examples,
        create_attention_probe,
        layers,
        **kwargs,
    )


def train_orthogonal_ensemble_probe(
    encoder, positive_examples, negative_examples, n_probes, layers, **kwargs
):
    # Train an orthogonal ensemble probe for each specified layer
    def create_orthogonal_ensemble_probe():
        return OrthogonalEnsembleProbe(encoder.model.config.hidden_size, n_probes)

    return train_probe(
        encoder,
        positive_examples,
        negative_examples,
        create_orthogonal_ensemble_probe,
        layers,
        **kwargs,
    )


def train_subspace_probe(
    encoder, positive_examples, negative_examples, subspace_dim, layers, **kwargs
):
    # Train a subspace probe for each specified layer
    def create_subspace_probe():
        return SubspaceProbe(encoder.model.config.hidden_size, subspace_dim)

    return train_probe(
        encoder,
        positive_examples,
        negative_examples,
        create_subspace_probe,
        layers,
        **kwargs,
    )

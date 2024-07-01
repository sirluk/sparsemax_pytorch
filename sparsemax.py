import torch
from torch import nn
from functools import partial

from typing import Optional

from utils import (
    convert_target_indices_to_label_vector,
    sparsemax,
    _sparsemax_intermediate_calc
)


class Sparsemax(nn.Module):
    
        def __init__(self, dim: int = -1):
            """Sparsemax activation function.
    
            Args:
                dim: Dimension along which to apply sparsemax.
            """
            super().__init__()
            self.dim = dim
    
        def forward(self, x) -> torch.Tensor:
            """Sparsemax activation function.

            Args:
                x: A `torch.Tensor` of logits.

            Returns:
                `torch.Tensor` of sparse probability values.
            """
            return sparsemax(x, dim=self.dim)


class SparsemaxLoss(nn.Module):

    def __init__(self, reduce: bool = True, target_indices: bool = False, fixed_n_labels: Optional[int] = None):
        """Sparsemax loss function.

        Args:
            reduce: If true returns mean loss per batch item.
            target_indices: Set to true if targets contains indices for true labels.
            fixed_n_labels: If not None, the number of positive labels for each example equal to this constant.
                This can speed up the conversion step from indices to label probability vectors.
        """
        super().__init__()
        self.reduce = reduce
        self.target_indices = target_indices
        self.fixed_n_labels = fixed_n_labels
        
        if reduce:
            self._reduce_fn = lambda x: x.mean()
        else:
            self._reduce_fn = lambda x: x
        
        if target_indices:
            self._target_transform = partial(
                convert_target_indices_to_label_vector,
                make_prob=True,
                fixed_n_labels=fixed_n_labels)
        else:
            self._target_transform = lambda x, y: y
    
    def forward_with_sparsemax(self, output, target, sparsemax_output = None) -> torch.Tensor:
        """Sparsemax loss function.

        Args:
            output: A `torch.Tensor` of logits.
            target: A `torch.Tensor` with label indices.
            sparsemax: A `torch.Tensor` of sparsemax probabilities. If None, it will be computed from `output`.

        Returns:
            `torch.Tensor` of loss values.
        """
        target = self._target_transform(output, target)
        output = self._handle_neginf(output)

        if sparsemax_output is None:
            sparsemax_output = sparsemax(output)

        sum_s = torch.where(
            sparsemax_output > 0,
            sparsemax_output * (output - 0.5 * sparsemax_output),
            torch.zeros_like(sparsemax_output)
        ).sum(dim=-1)

        q_part = self._q_part(target, output)

        return self._reduce_fn(sum_s + q_part)
    
    def forward(self, output, target) -> torch.Tensor:
        """Sparsemax loss function formulation that does not use the sparsemax transformation.

        Args:
            output: A `torch.Tensor` of logits.
            target: A `torch.Tensor` with label indices.

        Returns:
            `torch.Tensor` of loss values.
        """
        target = self._target_transform(output, target)
        output = self._handle_neginf(output)

        tau, k, z_j = _sparsemax_intermediate_calc(output, dim=-1)
        sum_s = 0.5 * (z_j.pow(2) - tau.pow(2)).cumsum(-1).gather(-1, k-1).squeeze(-1)

        q_part = self._q_part(target, output)

        return self._reduce_fn(sum_s + q_part)
    
    @staticmethod
    def _handle_neginf(x):
        """Handle -inf in logits gracefully (avoids potential NaN issues)"""
        return x.masked_fill(torch.isneginf(x), torch.finfo(x.dtype).min)#
    
    @staticmethod
    def _q_part(target, output):
        q_part = target * (0.5 * target - output)
        # Safely compute q_part (avoiding NaN when labels=0 and z=-inf)
        q_part = torch.where((target == 0) & torch.isinf(output), 0, q_part)
        return q_part.sum(dim=-1)
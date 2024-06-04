import torch


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Sparsemax activation function.

    For each batch $i$, and class $j$,
    compute sparsemax activation function:

    $$
    \mathrm{sparsemax}(x)[i, j] = \max(\mathrm{logits}[i, j] - \tau(\mathrm{logits}[i, :]), 0).
    $$

    See [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068).

    Usage:

    >>> x = torch.tensor([[-1.0, 0.0, 1.0], [-5.0, 1.0, 2.0]])
    >>> sparsemax(x, dim = -1)
    <torch.Tensor: shape=(2, 3), dtype=torch.float32, data=
    tensor([[0., 0., 1.],
           [0., 0., 1.]])>

    Args:
        logits: A `torch.Tensor`.
        axis: `int`, axis along which the sparsemax operation is applied.
    Returns:
        `torch.Tensor`, output of sparsemax transformation. Has the same type and
        shape as `logits`.
    Raises:
        ValueError: In  case `logits.ndim == 1`.
    """
    logits_dtype = logits.dtype
    
    logits, idx = logits.float().sort(dim=dim, descending=True)

    cs = logits.cumsum(dim=dim)

    k = 1 + torch.arange(1, logits.shape[dim]+1, device=logits.device, dtype=logits.dtype) * logits > cs.detach()
    k = k.sum(dim=dim, keepdim=True)

    s = torch.gather(cs, dim, k - 1)
    tau = torch.exp((s - 1).log() - k.to(s.dtype).log())

    return (logits - tau).clamp(min=0).gather(dim, idx.argsort(dim=-1)).to(logits_dtype)


def sparsemax_loss_with_logits(logits: torch.Tensor, sparsemax: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Sparsemax loss function.

    Args:
        logits: A `torch.Tensor` of logits.
        sparsemax: A `torch.Tensor` of sparsemax probabilities.
        targets: A `torch.Tensor` with label indices.

    Returns:
        `torch.Tensor` of loss values, one per example.
    """
    # Handle -inf in logits gracefully (avoids potential NaN issues)
    logits = logits.masked_fill(torch.isneginf(logits), torch.finfo(logits.dtype).min)

    sum_s = torch.where(sparsemax > 0, sparsemax * (logits - 0.5 * sparsemax), torch.zeros_like(sparsemax)).sum(dim=1)

    # convert target indices to label vector
    labels = torch.zeros_like(logits).scatter_(-1, targets, logits.new_ones(targets.shape))

    # Safely compute q_part (avoiding NaN when labels=0 and z=-inf)
    q_part = labels * (0.5 * labels - logits)
    q_part = torch.where((labels == 0) & torch.isinf(logits), torch.zeros_like(logits), q_part).sum(dim=1)

    return (sum_s + q_part).mean()

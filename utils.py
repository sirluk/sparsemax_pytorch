import torch

from typing import Optional, Union


def convert_target_indices_to_label_vector(
    output: torch.tensor,
    target: torch.tensor,
    make_prob: bool = True,
    fixed_n_labels: Optional[int] = None
) -> torch.tensor:
    target = torch.zeros_like(output).scatter(-1, target.long(), 1)
    if make_prob and fixed_n_labels is None:
        target /= target.sum(dim=-1, keepdim=True)
    elif make_prob:
        target /= fixed_n_labels
    return target


def prepare_tensor_for_broadcasting(x: torch.tensor, other: torch.tensor, dims: Union[int, tuple, list]) -> torch.tensor:
    if isinstance(dims, int):
        dims = [dims]
    m = [1] * other.dim()
    for i, dim in enumerate(dims):
        m[dim] = x.shape[i]
    return x.view(*m) 


def _sparsemax_intermediate_calc(z, dim):
    z, _ = z.sort(dim=dim, descending=True)
    cs = z.cumsum(dim=dim)
    k = torch.arange(1, z.shape[dim]+1, device=z.device, dtype=z.dtype)
    k = prepare_tensor_for_broadcasting(k, z, dim)
    k = (1 + k * z) > cs.detach()
    k = k.sum(dim=dim, keepdim=True)
    s = torch.gather(cs, dim, k - 1)
    tau = (s - 1) / k
    return tau, k, z[...,:k.max()]


def sparsemax(z, dim=-1):
    tau, _, _ = _sparsemax_intermediate_calc(z, dim)
    return (z - tau).clamp(min=0)
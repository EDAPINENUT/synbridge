import torch

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def expand_batch(batch, num_samples):
    batch_expanded = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            B = v.shape[0]
            dim = v.ndim
            batch_expanded[k] = (v.unsqueeze(0)
                                .repeat(num_samples, *([1] * dim))
                                .reshape(num_samples * B, *v.shape[1:]))
        else:
            batch_expanded[k] = [v] * num_samples
    return batch_expanded

def shape_back(x, B):
    return x.reshape(-1, B, *x.shape[1:])
import torch


def dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute element-wise batched dot product.

    For inputs of shape ``(bs, V)`` this returns a tensor of shape ``(bs, 1)``
    where each element contains the dot-product between the corresponding row
    vectors of ``x`` and ``y``.

    Args:
        x (torch.Tensor): Input tensor of shape (bs, V)
        y (torch.Tensor): Input tensor of shape (bs, V)

    Returns:
        torch.Tensor: Scores tensor of shape (bs, 1) where
            scores[i][0] = dot_product(x[i], y[i])
    """
    scores = torch.sum(x * y, dim=1, keepdim=True)
    return scores


def dot_product_cross(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute cross-batch dot-product between rows of ``x`` and rows of ``y``.

    Given ``x`` and ``y`` with shapes ``(bs, V)``, returns a ``(bs, bs)``
    matrix where element ``(i, j)`` equals the dot product between ``x[i]`` and
    ``y[j]``. This is useful for in-batch negative sampling schemes.

    Args:
        x (torch.Tensor): Input tensor of shape (bs, V)
        y (torch.Tensor): Input tensor of shape (bs, V)

    Returns:
        torch.Tensor: Scores tensor of shape (bs, bs) where
            scores[i][j] = dot_product(x[i], y[j])
    """
    scores = torch.matmul(x, y.T)
    return scores

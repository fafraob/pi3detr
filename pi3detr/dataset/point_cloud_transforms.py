import math
import random
import torch
import numpy as np
from typing import Optional
import torch_geometric.transforms as T
from torch_geometric.data import Data


def subsample(
    data: Data,
    upper_bound: float = 1.0,
    lower_bound: float = 0.5,
    max_points: Optional[int] = None,
    extra_fields: list[str] = [],
) -> Data:
    r"""Subsamples the point cloud to a random number of points within the
    range :obj:`[lower_bound, upper_bound]` (functional name: :obj:`subsample`).
    """
    if data.pos.size(0) == 0:
        return data
    num_points = int(random.uniform(lower_bound, upper_bound) * data.pos.size(0))
    if max_points is not None:
        num_points = min(num_points, max_points)
    idx = torch.randperm(data.pos.size(0))[:num_points]
    data.pos = data.pos[idx]
    for field in extra_fields:
        if hasattr(data, field):
            setattr(data, field, getattr(data, field)[idx])
    return data


def numpy_normalize_and_scale(xyz: np.ndarray) -> tuple[np.ndarray, float, float]:
    r"""Normalizes the point cloud in such a way that the points are centered
    around the origin and are within the interval :math:`[-1, 1]` (functional
    name: :obj:`normalize`).
    """
    center = xyz.mean(0)
    scale = (1 / np.max(np.abs(xyz - center))) * 0.999999
    xyz = numpy_normalize_and_scale_with_params(xyz, center, scale)
    return xyz, center, scale


def numpy_normalize_and_scale_with_params(
    xyz: np.ndarray, center: np.ndarray, scale: float
) -> np.ndarray:
    r"""Normalizes the point cloud in such a way that the points are centered
    around the origin and are within the interval :math:`[-1, 1]` (functional
    name: :obj:`normalize`).
    """
    if xyz.size == 0:
        return xyz
    shape = xyz.shape
    return ((xyz.reshape(-1, shape[-1]) - center) * scale).reshape(shape)


def normalize_and_scale(data: Data, extra_fields: list[str] = []) -> Data:
    r"""Centers and normalizes the given fields to the interval :math:`[-1, 1]`
    (functional name: :obj:`normalize_scale`).
    """
    if data.pos.size(0) == 0:
        data.center = torch.empty(0)
        data.scale = torch.empty(0)
        return data
    # center the pos points
    center = data.pos.mean(dim=-2, keepdim=True)
    # scale the pos points
    scale = (1 / (data.pos - center).abs().max()) * 0.999999

    return normalize_and_scale_with_params(data, center, scale, extra_fields)


def reverse_normalize_and_scale(data: Data, extra_fields: list[str] = []) -> Data:
    r"""Reverses the centering and normalization of the given fields
    (functional name: :obj:`reverse_normalize_scale`).
    """
    assert hasattr(data, "center") and hasattr(
        data, "scale"
    ), "Data object does not contain the center and scale attributes."
    return reverse_normalize_and_scale_with_params(
        data, data.center, data.scale, extra_fields
    )


def normalize_and_scale_with_params(
    data: Data, center: torch.Tensor, scale: torch.Tensor, extra_fields: list[str] = []
) -> Data:
    if data.pos.size(0) == 0:
        data.center = torch.empty(0)
        data.scale = torch.empty(0)
        return data
    data.pos = (data.pos - center) * scale
    for field in extra_fields:
        if hasattr(data, field):
            shape = getattr(data, field).size()
            setattr(
                data,
                field,
                (getattr(data, field).reshape(-1, shape[-1]) - center) * scale,
            )
            setattr(data, field, getattr(data, field).reshape(shape))
    data.center = center
    data.scale = scale
    return data


def reverse_normalize_and_scale_with_params(
    data: Data, center: torch.Tensor, scale: torch.Tensor, extra_fields: list[str] = []
) -> Data:
    r"""Reverses the centering and normalization of the given fields
    (functional name: :obj:`reverse_normalize_scale`).
    """
    # Reverse the scaling and centering of the pos points
    data.pos = data.pos / scale + center

    for field in extra_fields:
        if hasattr(data, field):
            shape = getattr(data, field).size()
            setattr(
                data,
                field,
                (getattr(data, field).reshape(-1, shape[-1]) / scale) + center,
            )
            setattr(data, field, getattr(data, field).reshape(shape))
    data.center = torch.empty(0)
    data.scale = torch.empty(0)
    return data


def reverse_normalize_and_scale_with_params(
    data: Data, center: torch.Tensor, scale: torch.Tensor, extra_fields: list[str] = []
) -> Data:
    r"""Reverses the centering and normalization of the given fields
    (functional name: :obj:`reverse_normalize_scale`).
    """
    # Reverse the scaling and centering of the pos points
    data.pos = data.pos / scale + center

    for field in extra_fields:
        if hasattr(data, field):
            shape = getattr(data, field).size()
            setattr(
                data,
                field,
                (getattr(data, field).reshape(-1, shape[-1]) / scale) + center,
            )
            setattr(data, field, getattr(data, field).reshape(shape))
    data.center = torch.empty(0)
    data.scale = torch.empty(0)
    return data


def random_rotate(
    data: Data, degrees: float, axis: int, extra_fields: list[str] = []
) -> Data:
    r"""Rotates the object around the origin by a random angle within the
    range :obj:`[-degrees, degrees]` (functional name: :obj:`random_rotate
    `).
    """
    if data.pos.size(0) == 0:
        return data
    return rotate_with_params(
        data, random.uniform(-degrees, degrees), axis, extra_fields
    )


def rotate_with_params(
    data: Data, degrees: float, axis: int = 0, extra_fields: list[str] = []
) -> Data:
    r"""Rotates the object around the origin by a given angle
    (functional name: :obj:`rotate`).
    """
    angle = math.pi * degrees / 180.0
    if data.pos.size(0) == 0:
        return data
    sin, cos = math.sin(angle), math.cos(angle)
    if data.pos.size(-1) == 2:
        matrix = torch.tensor([[cos, sin], [-sin, cos]])
    else:
        if axis == 0:
            matrix = torch.tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
        elif axis == 1:
            matrix = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
        else:
            matrix = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])

    matrix_dtype = matrix.to(data.pos.dtype)
    matrix = matrix.to(matrix_dtype)

    data.pos = data.pos @ matrix.t()
    for field in extra_fields:
        if hasattr(data, field):
            shape = getattr(data, field).size()
            # get dtype of field
            dtype = getattr(data, field).dtype

            matrix_dtype = matrix.to(dtype)
            setattr(data, field, getattr(data, field) @ matrix_dtype.t())
            setattr(data, field, getattr(data, field).reshape(shape))
    setattr(data, f"rotated_{axis}", degrees)
    return data


def reverse_rotate(data: Data, axis: int = 0, extra_fields: list[str] = []) -> Data:
    r"""Reverses the rotation of the object around the origin
    (functional name: :obj:`reverse_rotate`).
    """
    if not hasattr(data, f"rotated_{axis}"):
        return data
    return rotate_with_params(
        data, -getattr(data, f"rotated_{axis}"), axis, extra_fields
    )


def add_noise(data: Data, std: float) -> Data:
    r"""Adds Gaussian noise to the node features (functional name:
    :obj:`add_noise`).
    """
    if data.pos.size(0) == 0:
        return data
    noise = torch.randn_like(data.pos) * std
    data.pos = data.pos + noise
    data.noise = noise
    return data


def remove_noise(data: Data) -> Data:
    r"""Removes the noise from the node features (functional name:
    :obj:`remove_noise`).
    """
    assert hasattr(data, "noise"), "Data object does not contain the noise attribute."
    data.pos = data.pos - data.noise
    del data.noise
    return data


def custom_normalize_and_scale(
    data: Data, p1: torch.Tensor, p2: torch.Tensor, extra_fields: list[str] = []
) -> Data:
    r"""Normalizes the point cloud in such a way that after the transformation
    p1 is at (0,0,0) and p2 at (1,1,1)` (functional name:
    :obj:`normalize`).
    """
    assert p1.size() == p2.size() == (3,), "Invalid interval."
    if data.pos.size(0) == 0:
        return data
    data.pos = (data.pos - p1) / (p2 - p1)
    for field in extra_fields:
        if hasattr(data, field):
            shape = getattr(data, field).size()
            setattr(
                data,
                field,
                (getattr(data, field).reshape(-1, shape[-1]) - p1) / (p2 - p1),
            )
            setattr(data, field, getattr(data, field).reshape(shape))
    data.p1 = p1
    data.p2 = p2
    return data

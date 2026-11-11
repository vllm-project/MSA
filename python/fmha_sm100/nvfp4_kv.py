# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""The NVFP4 KV cache page layout that fmha_sm100 reads."""

from typing import Optional, Tuple

import torch


def nvfp4_head_slot_views(
    k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the per-head K/V slots of an NVFP4 cache into data and block-scale views.

    The NVFP4 cache is uint8 ``[pages, 2 * Hkv, P, 72]`` with P=128 or 256: slot ``2 * h``
    is head ``h``'s K and slot ``2 * h + 1`` its V. A slot holds the head's packed E2M1
    data (P * 64 bytes) followed by its E4M3 block scales (P * 8 bytes).
    ``k`` and ``v`` are the K and V slots, ``cache[:, 0::2]`` and
    ``cache[:, 1::2]``; any other layout is rejected.

    Returns ``(k_data, k_scale, v_data, v_scale)``, ``[pages, Hkv, P, 64]`` and
    ``[pages, Hkv, P, 8]`` uint8 views.
    """
    if k.dtype != torch.uint8 or v.dtype != torch.uint8:
        raise TypeError(f"NVFP4 k/v must be torch.uint8, got {k.dtype} and {v.dtype}")
    if (k.ndim != 4 or k.shape[2] not in (128, 256) or k.shape[3] != 72
            or k.shape != v.shape):
        raise ValueError("NVFP4 k/v must be [pages, Hkv, P, 72] slot views with P=128/256, got "
                         f"{tuple(k.shape)} and {tuple(v.shape)}")
    pages, heads = k.shape[:2]
    page_size = k.shape[2]
    data_bytes, slot_bytes = page_size * 64, page_size * 72
    head_stride, page_stride = 2 * slot_bytes, k.stride(0)
    if (any(t.stride(2) != 72 or t.stride(3) != 1 for t in (k, v))
            or v.stride(0) != page_stride
            or v.data_ptr() != k.data_ptr() + slot_bytes
            or (heads > 1 and not k.stride(1) == v.stride(1) == head_stride)
            or page_stride < heads * head_stride):
        raise ValueError(
            "NVFP4 k/v must be the per-head K/V slots cache[:, 0::2] and cache[:, 1::2] of a "
            "[pages, 2 * Hkv, P, 72] cache, each head's K slot followed by its V slot")
    if page_stride % 16 or k.data_ptr() % 16:
        raise ValueError("NVFP4 cache pages must be 16-byte aligned")
    views = []
    for slots in (k, v):
        for width, offset in ((64, 0), (8, data_bytes)):
            views.append(slots.as_strided((pages, heads, page_size, width),
                                          (page_stride, head_stride, width, 1),
                                          slots.storage_offset() + offset))
    k_data, k_scale, v_data, v_scale = views
    return k_data, k_scale, v_data, v_scale


def _check_nvfp4_kv_views(
    k: torch.Tensor, v: torch.Tensor, k_sf: torch.Tensor, v_sf: torch.Tensor
) -> None:
    """Validate caller-built NVFP4 data and block-scale views.

    Data views are uint8 ``[pages, Hkv, 128, 64]``, scale views uint8
    ``[pages, Hkv, 128, 8]``, all with contiguous rows. The kernels take one
    page stride and one data / one scale head stride from the K views, so V
    must match K stride for stride.
    """
    pages, heads = k.shape[:2]
    for name, t, row in (("k", k, 64), ("v", v, 64), ("k_sf", k_sf, 8), ("v_sf", v_sf, 8)):
        if (t.dtype != torch.uint8 or tuple(t.shape) != (pages, heads, 128, row)
                or t.stride(3) != 1 or t.stride(2) != row):
            raise ValueError(
                f"NVFP4 {name} view must be uint8 [pages, Hkv, 128, {row}] with "
                f"contiguous rows; got shape {tuple(t.shape)} strides {tuple(t.stride())}")
    if not (k.stride(0) == v.stride(0) == k_sf.stride(0) == v_sf.stride(0)):
        raise ValueError("NVFP4 K/V data and scale views must share one page stride")
    if k.stride(1) != v.stride(1) or k_sf.stride(1) != v_sf.stride(1):
        raise ValueError("NVFP4 K and V views must share the data and scale head strides")


def nvfp4_kv_views(
    k: torch.Tensor,
    v: torch.Tensor,
    k_sf: Optional[torch.Tensor] = None,
    v_sf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ``(k_data, k_scale, v_data, v_scale)`` views the NVFP4 decode and prefill kernels read.

    Without ``k_sf`` / ``v_sf``, ``k`` and ``v`` are the per-head K/V slots and the views come
    from :func:`nvfp4_head_slot_views`. With them, ``k`` / ``v`` are caller-built uint8
    ``[pages, Hkv, 128, 64]`` data views and ``k_sf`` / ``v_sf`` ``[pages, Hkv, 128, 8]``
    block-scale views, returned as they are once validated; the kernels take the page stride and
    the data and scale head strides from them.
    """
    if (k_sf is None) != (v_sf is None):
        raise ValueError("NVFP4 k_sf and v_sf must be passed together")
    if k_sf is None:
        return nvfp4_head_slot_views(k, v)
    _check_nvfp4_kv_views(k, v, k_sf, v_sf)
    return k, k_sf, v, v_sf

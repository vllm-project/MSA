# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Caller-built NVFP4 K/V views for ``fmha_sm100``: shape and stride contract."""

import pytest
import torch

from fmha_sm100.api import _check_nvfp4_kv_views

PAGES, HEADS, PAGE_SIZE, SLOT_BYTES = 3, 4, 128, 128 * 72


def _head_slot_views(kv_cache: torch.Tensor):
    """Views of a ``[pages, 2*Hkv, 128, 72]`` cache whose slot ``2h`` is head
    ``h``'s K data then K scales and slot ``2h+1`` the same for V."""
    pages, slots = kv_cache.shape[:2]
    heads = slots // 2
    page_stride, slot_stride = kv_cache.stride(0), kv_cache.stride(1)
    views = []
    for side in range(2):
        base = kv_cache.storage_offset() + side * slot_stride
        views.append(kv_cache.as_strided(
            (pages, heads, PAGE_SIZE, 64), (page_stride, 2 * slot_stride, 64, 1), base))
        views.append(kv_cache.as_strided(
            (pages, heads, PAGE_SIZE, 8), (page_stride, 2 * slot_stride, 8, 1),
            base + PAGE_SIZE * 64))
    return views


def test_head_slot_views_are_accepted():
    kv_cache = torch.zeros(PAGES, 2 * HEADS, PAGE_SIZE, 72, dtype=torch.uint8)
    k, k_sf, v, v_sf = _head_slot_views(kv_cache)
    _check_nvfp4_kv_views(k, v, k_sf, v_sf)
    assert k.stride(1) == k_sf.stride(1) == 2 * SLOT_BYTES
    assert v.storage_offset() == SLOT_BYTES and k_sf.storage_offset() == PAGE_SIZE * 64


def test_legacy_side_views_are_accepted():
    """The strides ``_fmha_sm100`` derives itself for packed sides pass too."""
    kv_cache = torch.zeros(PAGES, 2 * HEADS, PAGE_SIZE, 72, dtype=torch.uint8)
    k_side, v_side = kv_cache[:, :HEADS], kv_cache[:, HEADS:]
    views = []
    for side in (k_side, v_side):
        views.append(side.as_strided((PAGES, HEADS, PAGE_SIZE, 64), (side.stride(0), 8192, 64, 1)))
        views.append(side.as_strided((PAGES, HEADS, PAGE_SIZE, 8), (side.stride(0), 1024, 8, 1),
                                     side.storage_offset() + HEADS * 8192))
    _check_nvfp4_kv_views(views[0], views[2], views[1], views[3])


@pytest.mark.parametrize("bad", ["shape", "row", "page_stride", "head_stride", "dtype"])
def test_mismatched_views_are_rejected(bad):
    kv_cache = torch.zeros(PAGES, 2 * HEADS, PAGE_SIZE, 72, dtype=torch.uint8)
    k, k_sf, v, v_sf = _head_slot_views(kv_cache)
    if bad == "shape":
        v_sf = v_sf[:, :-1]
    elif bad == "row":
        k = kv_cache[:, 0::2, :, :64]
    elif bad == "page_stride":
        v = torch.zeros(PAGES, HEADS, PAGE_SIZE, 64, dtype=torch.uint8)
    elif bad == "head_stride":
        v_sf = torch.zeros_like(kv_cache).as_strided(
            v_sf.shape, (k_sf.stride(0), 8 * PAGE_SIZE, 8, 1))
    elif bad == "dtype":
        k_sf = k_sf.view(torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        _check_nvfp4_kv_views(k, v, k_sf, v_sf)

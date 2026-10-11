// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once

#include "cutlass/cutlass.h"

namespace q8kv4_indexer_sm107 {

struct IndexerGemmTraits {
  static constexpr int kQueryLength = 8;
  static constexpr int kHeadDim = 128;
  static constexpr int kPageTokens = 128;
  static constexpr int kScaleGroupSize = 16;
  static constexpr int kScaleGroups = kHeadDim / kScaleGroupSize;
  static constexpr int kPackedKBytes = kPageTokens * kHeadDim / 2;
  static constexpr int kScaleBytes = kPageTokens * kScaleGroups;
  static constexpr int kPageBytes = kPackedKBytes + kScaleBytes;
  static constexpr int kScoreWarps = 8;
  static constexpr int kTransportWarps = 1;
  static constexpr int kThreads = (kScoreWarps + kTransportWarps) * cutlass::NumThreadsPerWarp;
  static constexpr int kMaxPagesPerCta = 6;
  // Largest work tile the plan may choose (the partials buffer holds one tile).
  static constexpr int kPagesPerWorkTile = 24;
  static constexpr int kPartialPageCapacity = kPagesPerWorkTile;
  // Tile size once a launch needs more than one wave of CTAs (HBM-bound): long
  // enough to amortize a tile's claim and fold, short enough to keep the last
  // wave's tail small. Measured best within ~3% from 16k to 131k pages.
  static constexpr int kBandwidthTilePages = 16;
  // Per page the score warps store a partial per (warp, query slot, group of 4
  // tokens rows): 8 * 8 * 4 floats, padded to a 264-float stride so the fold's
  // vectorized loads and the warps' stores stay bank-conflict free.
  static constexpr int kPartialGroups = 4;
  static constexpr int kPartialPageStride =
      kScoreWarps * kQueryLength * kPartialGroups + 8;
  static constexpr int kSchedulerStages = 2;
  // Keep the inline scan small enough that scheduler setup remains negligible
  // for the latency-sensitive decode range. Larger batches use CUB scan.
  static constexpr int kPrepareThreads = 128;
  static constexpr int kPrepareWarps = kPrepareThreads / 32;
  static constexpr int kMaximumPages = 8192;
  // Inline-plan workspace word holding the launch's tile size, one 128-byte
  // line past the work counter (kPrepareThreads + 1) so its loads do not
  // queue behind the counter's atomics.
  static constexpr int kTilePagesWord = kPrepareThreads + 1 + 32;
};

} // namespace q8kv4_indexer_sm107

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_S_TIR_META_SCHEDULE_DRAFT_MODEL_H_
#define TVM_S_TIR_META_SCHEDULE_DRAFT_MODEL_H_

#include <tvm/ir/module.h>

#include <cstdint>
#include <vector>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*!
 * \brief The explicit hardware profile used by the Pruner-style draft model.
 *
 * Peak performance is in GFLOP/s and global-memory bandwidth is in GB/s, matching
 * the units used by Pruner's public PSA implementation.
 */
struct PrunerDraftHardwareProfile {
  double peak_gflops{-1.0};
  double global_memory_gbps{-1.0};
  int64_t sm_count{-1};
  int64_t active_blocks_per_sm{1};
  int64_t sm_partitions{4};
  int64_t warp_size{-1};
  double register_capacity{128.0};

  bool IsValid() const;
};

/*! \brief One 11-field PSA feature row before serialization. */
struct PrunerDraftStatementFeatures {
  double float_workload_per_thread{0.0};
  double compute_workload{0.0};
  double register_usage{0.0};
  double memory_workload{0.0};
  double shared_memory_footprint{0.0};
  double dram_workload{0.0};
  double data_reuse_penalty{1.0};
  double is_gpu{0.0};
  double grid_size{1.0};
  double block_size{1.0};
  double virtual_thread_size{1.0};
  bool known{true};
};

/*! \brief Result of deterministic Pruner-style draft scoring. */
struct PrunerDraftScore {
  bool known{false};
  double score{1.0};
  int64_t statement_count{0};
};

/*!
 * \brief Apply the public Pruner PSA latency formula to already extracted rows.
 * \return A positive score when all required fields are known, otherwise neutral 1.0.
 */
PrunerDraftScore ScorePrunerDraftFeatures(const std::vector<PrunerDraftStatementFeatures>& features,
                                          const PrunerDraftHardwareProfile& profile);

/*!
 * \brief Extract operator-independent PSA features from postprocessed S-TIR and score them.
 *
 * Dynamic loop extents and unsupported GPU bindings produce an unknown, neutral result.
 */
PrunerDraftScore AnalyzePrunerDraftScore(const IRModule& mod,
                                         const PrunerDraftHardwareProfile& profile);

/*!
 * \brief Normalize known draft scores by the best score in the current population.
 *
 * Unknown scores receive the neutral multiplier 1.0.
 */
std::vector<double> NormalizePrunerDraftScores(const std::vector<PrunerDraftScore>& scores);

/*!
 * \brief Combine learned-model scores with normalized draft scores.
 *
 * Mode 0 (off) and mode 1 (shadow) are identity.  Mode 2 applies
 * `learned_score * draft_score^beta` without removing candidates.
 */
std::vector<double> ApplyPrunerDraftScores(const std::vector<double>& learned_scores,
                                           const std::vector<double>& normalized_draft_scores,
                                           int mode, double beta);

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_DRAFT_MODEL_H_

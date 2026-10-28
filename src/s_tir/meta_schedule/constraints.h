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
#ifndef TVM_S_TIR_META_SCHEDULE_CONSTRAINTS_H_
#define TVM_S_TIR_META_SCHEDULE_CONSTRAINTS_H_

#include <tvm/ffi/any.h>
#include <tvm/ir/module.h>
#include <tvm/s_tir/schedule/schedule.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*! \brief Result of the cheap pre-lowering GPU feasibility analysis. */
struct ThreadFeasibilityResult {
  /*! \brief Whether all physical thread extents were statically understood. */
  bool all_extents_known{true};
  /*! \brief Whether all encountered thread-extent contracts could be checked. */
  bool all_contracts_known{true};
  /*! \brief Whether a statically proven thread constraint is violated. */
  bool reject{false};
  /*! \brief Whether a block is below its annotated inclusive lower bound. */
  bool reject_thread_extent_low{false};
  /*! \brief Whether a block is above its annotated inclusive upper bound. */
  bool reject_thread_extent_high{false};
  /*! \brief Whether a launch exceeds the target threads-per-block limit. */
  bool reject_threads_per_block{false};
  /*! \brief Maximum threads on any launch path, or -1 when some extent is unknown. */
  int64_t max_threads{-1};
  /*! \brief Whether every declared shared allocation had a static shape. */
  bool all_shared_allocations_known{true};
  /*!
   * \brief Whether declared pre-compaction shared bytes exceed the target limit.
   *
   * This is an observation only.  It is deliberately excluded from `reject`,
   * because later lowering may compact or merge schedule-level buffers.
   */
  bool shared_memory_proxy_reject{false};
  /*! \brief Maximum simultaneously scoped, declared shared bytes before compaction. */
  int64_t max_declared_shared_bytes{0};
};

/*! \brief Three-valued result used by the trace-level constraint evaluator. */
enum class TraceConstraintTruth : int {
  kProvenFeasible = 0,
  kProvenInfeasible,
  kUnknown,
};

/*! \brief Reasons why a trace constraint could not be proved. */
enum class TraceConstraintUnknownReason : int {
  kNone = 0,
  kMissingDecision,
  kDynamicExtent,
  kPartialContract,
  kUnresolvedBlockScope,
  kConflictingAxisExtent,
  kUnsupportedInstruction,
  kOverlappingConstraints,
  kWarpSizeMissing,
  kNoContractAnnotation,
  kEnumerationLimit,
  kCount,
};

const char* TraceConstraintUnknownReasonName(TraceConstraintUnknownReason reason);

/*! \brief A reference to one entry in a sampling instruction's decision. */
struct TraceFactorRef {
  int instruction_index{-1};
  int decision_index{-1};
};

/*! \brief A product of constants and trace sampling decisions. */
struct TraceSymbolicProduct {
  bool known{false};
  int64_t constant{1};
  std::vector<TraceFactorRef> factors;
};

/*! \brief Sampling metadata needed by the constrained decision provider. */
struct TraceSampleSpec {
  enum class Kind : int {
    kPerfectTile = 0,
    kCategorical,
  };

  Kind kind{Kind::kPerfectTile};
  int instruction_index{-1};
  int n_splits{0};
  int max_innermost_factor{-1};
  std::vector<int64_t> categorical_candidates;
  std::vector<double> categorical_probabilities;
};

/*! \brief Thread-product constraint attached to one schedule block. */
struct TraceBlockThreadConstraint {
  std::string block_key;
  std::array<TraceSymbolicProduct, 3> axis_products;
  std::array<bool, 3> axis_present{{false, false, false}};
  bool scope_known{true};
  bool contract_present{false};
  bool contract_complete{false};
  bool warp_execution{false};
  int64_t low_inclusive{-1};
  int64_t high_inclusive{-1};
  int64_t hardware_limit{-1};
  int conditional_instruction_index{-1};
  TraceConstraintUnknownReason unknown_reason{TraceConstraintUnknownReason::kNone};
};

/*! \brief Reusable constraint plan derived from a trace instruction skeleton. */
struct TraceConstraintPlan {
  std::vector<TraceSampleSpec> samples;
  std::vector<TraceBlockThreadConstraint> constraints;
  std::array<int64_t, static_cast<size_t>(TraceConstraintUnknownReason::kCount)>
      unknown_reason_counts{};
  int64_t contract_blocks{0};
  int64_t enforceable_constraints{0};
};

/*! \brief Evaluation of a materialized trace against a TraceConstraintPlan. */
struct TraceConstraintEvaluation {
  TraceConstraintTruth truth{TraceConstraintTruth::kUnknown};
  bool reject_thread_extent_low{false};
  bool reject_thread_extent_high{false};
  bool reject_threads_per_block{false};
  int64_t max_threads{-1};
  TraceConstraintUnknownReason unknown_reason{TraceConstraintUnknownReason::kNone};
};

/*! \brief Result returned by the constrained decision provider helper. */
struct TraceConstraintDecisionResult {
  bool handled{false};
  bool empty_domain{false};
  bool unknown{false};
  ffi::Any decision{nullptr};
  int64_t feasible_domain_size{0};
  TraceConstraintUnknownReason unknown_reason{TraceConstraintUnknownReason::kNone};
};

/*!
 * \brief Derive thread-product constraints from a trace instruction skeleton.
 *
 * The analysis is structural.  It does not use workload names or fixed tile indices.
 */
TraceConstraintPlan BuildTraceConstraintPlan(const s_tir::Trace& trace,
                                             int64_t max_threads_per_block,
                                             int64_t thread_warp_size);

/*! \brief Evaluate all provable constraints using decisions stored in a trace. */
TraceConstraintEvaluation EvaluateTraceConstraints(const TraceConstraintPlan& plan,
                                                    const s_tir::Trace& trace);

/*!
 * \brief Sample a decision for a conditional instruction from its feasible domain.
 * \param plan The template-level constraint plan.
 * \param instruction_index The instruction currently being replayed.
 * \param prefix_trace The already replayed prefix, containing earlier decisions.
 * \param loop_extent Current concrete loop extent, or -1 for non-perfect-tile samples.
 * \param rand_state Random state private to the decision provider.
 * \param enumeration_limit Maximum complete decision vectors to enumerate.
 */
TraceConstraintDecisionResult SampleConstrainedTraceDecision(
    const TraceConstraintPlan& plan, int instruction_index, const s_tir::Trace& prefix_trace,
    int64_t loop_extent, LinearCongruentialEngine::TRandState* rand_state,
    int64_t enumeration_limit = 65536);

/*!
 * \brief Analyze physical threads per block without lowering the module.
 * \param mod The module to inspect.
 * \param max_threads_per_block The target limit.
 * \param thread_warp_size The target warp size used by warp-execution blocks.
 * \param max_shared_memory_per_block The target shared-memory limit.
 * \return The feasibility result.  Unknown structure is never itself a rejection.
 */
ThreadFeasibilityResult AnalyzeThreadFeasibility(const IRModule& mod,
                                                 int64_t max_threads_per_block,
                                                 int64_t thread_warp_size,
                                                 int64_t max_shared_memory_per_block);

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_CONSTRAINTS_H_

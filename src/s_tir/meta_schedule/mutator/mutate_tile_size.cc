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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using s_tir::Instruction;
using s_tir::InstructionKind;
using s_tir::Trace;

/*!
 * \brief Downcast the decision of Sample-Perfect-Tile to an array of integers
 * \param decision The decision of Sample-Perfect-Tile
 * \return The result of downcast
 */
std::vector<int64_t> DowncastTilingDecision(const ffi::ObjectRef& decision) {
  const auto* arr = TVM_TYPE_AS(decision, ffi::ArrayObj);
  return support::AsVector<ffi::ObjectRef, int64_t>(ffi::GetRef<ffi::Array<ffi::ObjectRef>>(arr));
}

/*!
 * \brief Calculate the product of elements in an array
 * \param array The array
 * \return The product of elements in the array
 */
int64_t Product(const std::vector<int64_t>& array) {
  int64_t result = 1;
  for (int64_t x : array) {
    result *= x;
  }
  return result;
}

namespace {

struct ThreadExtentEstimate {
  bool known{true};
  int64_t threads{1};
};

struct MutateTileThreadStats {
  std::atomic<int64_t> proposal_total{0};
  std::atomic<int64_t> proposal_known{0};
  std::atomic<int64_t> proposal_unknown{0};
  std::atomic<int64_t> proposal_initial_legal{0};
  std::atomic<int64_t> checked{0};
  std::atomic<int64_t> unknown{0};
  std::atomic<int64_t> initial_reject{0};
  std::atomic<int64_t> projected{0};
  std::atomic<int64_t> projection_failed{0};
  std::atomic<int64_t> direction_candidates{0};
  std::atomic<int64_t> directed_choices{0};
  std::atomic<int64_t> direction_fallback{0};
  std::atomic<int64_t> slack_expand{0};
  std::atomic<int64_t> pressure_nonincrease{0};
  std::atomic<int64_t> divisor_steps{0};

  void Reset() {
    proposal_total.store(0, std::memory_order_relaxed);
    proposal_known.store(0, std::memory_order_relaxed);
    proposal_unknown.store(0, std::memory_order_relaxed);
    proposal_initial_legal.store(0, std::memory_order_relaxed);
    checked.store(0, std::memory_order_relaxed);
    unknown.store(0, std::memory_order_relaxed);
    initial_reject.store(0, std::memory_order_relaxed);
    projected.store(0, std::memory_order_relaxed);
    projection_failed.store(0, std::memory_order_relaxed);
    direction_candidates.store(0, std::memory_order_relaxed);
    directed_choices.store(0, std::memory_order_relaxed);
    direction_fallback.store(0, std::memory_order_relaxed);
    slack_expand.store(0, std::memory_order_relaxed);
    pressure_nonincrease.store(0, std::memory_order_relaxed);
    divisor_steps.store(0, std::memory_order_relaxed);
  }

  static MutateTileThreadStats* Global() {
    static MutateTileThreadStats stats;
    return &stats;
  }
};

bool SafeMultiply(int64_t lhs, int64_t rhs, int64_t* result) {
  if (lhs <= 0 || rhs <= 0) {
    return false;
  }
  if (lhs > std::numeric_limits<int64_t>::max() / rhs) {
    *result = std::numeric_limits<int64_t>::max();
  } else {
    *result = lhs * rhs;
  }
  return true;
}

int ThreadAxis(const ffi::String& tag) {
  if (tag == "threadIdx.x") return 0;
  if (tag == "threadIdx.y") return 1;
  if (tag == "threadIdx.z") return 2;
  return -1;
}

/*!
 * \brief Infer physical thread extents from trace data-flow only.
 *
 * The inference follows sampled expressions through Split and loop extents
 * through Fuse until they reach Bind(threadIdx.*).  It deliberately returns
 * unknown for unsupported provenance instead of guessing a tile position.
 */
ThreadExtentEstimate EstimateThreadsFromTrace(const Trace& trace) {
  static const InstructionKind& inst_sample_perfect_tile =
      InstructionKind::Get("SamplePerfectTile");
  static const InstructionKind& inst_sample_categorical = InstructionKind::Get("SampleCategorical");
  static const InstructionKind& inst_split = InstructionKind::Get("Split");
  static const InstructionKind& inst_fuse = InstructionKind::Get("Fuse");
  static const InstructionKind& inst_bind = InstructionKind::Get("Bind");
  static const InstructionKind& inst_annotate = InstructionKind::Get("Annotate");

  std::unordered_map<const ffi::Object*, int64_t> rv_extent;
  std::array<int64_t, 3> axis_extent{{1, 1, 1}};
  std::array<bool, 3> axis_seen{{false, false, false}};
  bool known = true;

  auto resolve_extent = [&rv_extent](const Any& value, int64_t* extent) -> bool {
    if (value == nullptr) return false;
    if (const auto* imm = value.as<IntImmNode>()) {
      if (imm->value <= 0) return false;
      *extent = imm->value;
      return true;
    }
    if (const ffi::Object* object = value.as<ffi::Object>()) {
      auto it = rv_extent.find(object);
      if (it != rv_extent.end()) {
        *extent = it->second;
        return true;
      }
    }
    return false;
  };

  for (const Instruction& inst : trace->insts) {
    if (inst->kind->IsPostproc()) break;

    if (inst->kind.same_as(inst_sample_perfect_tile)) {
      Any decision = trace->GetDecision(inst);
      if (decision == nullptr) continue;
      std::vector<int64_t> tiles = DowncastTilingDecision(decision.cast<ffi::ObjectRef>());
      if (tiles.size() != inst->outputs.size()) continue;
      for (int i = 0, n = static_cast<int>(tiles.size()); i < n; ++i) {
        if (tiles[i] > 0) {
          rv_extent[inst->outputs[i].as<ffi::Object>()] = tiles[i];
        }
      }
      continue;
    }

    if (inst->kind.same_as(inst_sample_categorical)) {
      Any decision = trace->GetDecision(inst);
      if (decision == nullptr || inst->attrs.empty() || inst->outputs.size() != 1) continue;
      int64_t index = decision.cast<int64_t>();
      ffi::Array<int64_t> candidates = Downcast<ffi::Array<int64_t>>(inst->attrs[0]);
      if (0 <= index && index < static_cast<int64_t>(candidates.size()) && candidates[index] > 0) {
        rv_extent[inst->outputs[0].as<ffi::Object>()] = candidates[index];
      }
      continue;
    }

    if (inst->kind.same_as(inst_split)) {
      int n = inst->outputs.size();
      if (inst->inputs.size() != static_cast<size_t>(n + 1)) continue;
      std::vector<int64_t> extents(n, -1);
      int missing = -1;
      int num_missing = 0;
      int64_t known_product = 1;
      for (int i = 0; i < n; ++i) {
        if (!resolve_extent(inst->inputs[i + 1], &extents[i])) {
          missing = i;
          ++num_missing;
        } else if (!SafeMultiply(known_product, extents[i], &known_product)) {
          ++num_missing;
        }
      }
      if (num_missing == 1) {
        int64_t input_extent = -1;
        if (resolve_extent(inst->inputs[0], &input_extent) && known_product > 0 &&
            input_extent % known_product == 0) {
          extents[missing] = input_extent / known_product;
        }
      }
      for (int i = 0; i < n; ++i) {
        if (extents[i] > 0) {
          rv_extent[inst->outputs[i].as<ffi::Object>()] = extents[i];
        }
      }
      continue;
    }

    if (inst->kind.same_as(inst_fuse)) {
      if (inst->outputs.size() != 1) continue;
      int64_t extent = 1;
      bool fuse_known = true;
      for (const Any& input : inst->inputs) {
        int64_t input_extent = -1;
        if (!resolve_extent(input, &input_extent) || !SafeMultiply(extent, input_extent, &extent)) {
          fuse_known = false;
          break;
        }
      }
      if (fuse_known) {
        rv_extent[inst->outputs[0].as<ffi::Object>()] = extent;
      }
      continue;
    }

    if (inst->kind.same_as(inst_bind)) {
      if (inst->inputs.size() != 1 || inst->attrs.size() != 1) {
        known = false;
        continue;
      }
      ffi::String tag = Downcast<ffi::String>(inst->attrs[0]);
      int axis = ThreadAxis(tag);
      if (axis == -1) {
        if (std::string(tag).rfind("threadIdx.", 0) == 0) known = false;
        continue;
      }
      int64_t extent = -1;
      if (!resolve_extent(inst->inputs[0], &extent)) {
        known = false;
      } else if (axis_seen[axis] && axis_extent[axis] != extent) {
        known = false;
      } else {
        axis_seen[axis] = true;
        axis_extent[axis] = extent;
      }
      continue;
    }

    if (inst->kind.same_as(inst_annotate) && inst->attrs.size() == 1 &&
        Downcast<ffi::String>(inst->attrs[0]) == s_tir::attr::warp_execution) {
      // warp_execution implicitly supplies threadIdx.x, but the warp size is
      // target-dependent and is not encoded in this trace.
      known = false;
    }
  }

  int64_t threads = 1;
  int num_axes_seen = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (axis_seen[axis]) {
      ++num_axes_seen;
      if (!SafeMultiply(threads, axis_extent[axis], &threads)) {
        known = false;
      }
    }
  }
  // A flat trace records Bind instructions but not enough lexical scope to
  // prove that different physical axes are active on the same launch path.
  // Defer zero-axis and multi-axis cases to the IR-based hard analyzer.
  if (num_axes_seen != 1) known = false;
  return ThreadExtentEstimate{/*known=*/known, /*threads=*/known ? threads : -1};
}

}  // namespace

/*! \brief A mutator that mutates the tile size */
class MutateTileSizeNode : public MutatorNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MutateTileSizeNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.MutateTileSize", MutateTileSizeNode,
                                    MutatorNode);

 public:
  // Inherit from `MutatorNode`
  void InitializeWithTuneContext(const TuneContext& context) final {
    const char* value = std::getenv("TVM_GA_T2_MODE");
    this->ga_t2_mode_ = value == nullptr ? 0 : std::atoi(value);
    if (this->ga_t2_mode_ < 0 || this->ga_t2_mode_ > 3) {
      this->ga_t2_mode_ = 0;
    }
    this->max_threads_per_block_ = -1;
    if (this->ga_t2_mode_ != 0 && context->target.defined()) {
      const Target& target = context->target.value();
      this->max_threads_per_block_ =
          target->GetAttr<int64_t>("max_threads_per_block")
              .value_or(target->GetAttr<int64_t>("max_num_threads").value_or(-1));
    }
  }
  // Inherit from `MutatorNode`
  ffi::Optional<Trace> Apply(const Trace& trace, TRandState* rand_state) final;
  // Inherit from `MutatorNode`
  Mutator Clone() const final {
    ffi::ObjectPtr<MutateTileSizeNode> n = ffi::make_object<MutateTileSizeNode>(*this);
    return Mutator(n);
  }

 private:
  /*! \brief 0: off, 1: shadow, 2: projection, 3: projection plus resource direction. */
  int ga_t2_mode_{0};
  /*! \brief Hardware limit used by the trace-level thread estimator. */
  int64_t max_threads_per_block_{-1};
};

/*!
 * \brief Find a sample-perfect-tile decision in the trace
 * \param trace The trace
 * \param rand_state The random state
 * \param inst The instruction selected
 * \param decision The decision selected
 * \return Whether a decision is found
 */
void FindSamplePerfectTile(const Trace& trace, std::vector<Instruction>* inst,
                           std::vector<std::vector<int64_t>>* decision) {
  static const InstructionKind& inst_sample_perfect_tile =
      InstructionKind::Get("SamplePerfectTile");
  std::vector<Instruction>& instructions = *inst;
  std::vector<std::vector<int64_t>>& decisions = *decision;
  instructions.reserve(trace->decisions.size());
  decisions.reserve(trace->decisions.size());
  for (const auto& kv : trace->decisions) {
    const Instruction& inst = kv.first;
    if (inst->kind.same_as(inst_sample_perfect_tile)) {
      std::vector<int64_t> tiles = DowncastTilingDecision(kv.second.cast<ffi::ObjectRef>());
      if (tiles.size() >= 2 && Product(tiles) >= 2) {
        instructions.push_back(inst);
        decisions.push_back(tiles);
      }
    }
  }
}

void FindSampleVectorize(const Trace& trace, std::vector<Instruction>* inst,
                         std::vector<int64_t>* decision) {
  static const InstructionKind& inst_sample_categorical = InstructionKind::Get("SampleCategorical");
  static const InstructionKind& inst_annotate = InstructionKind::Get("Annotate");
  std::vector<Instruction>& instructions = *inst;
  std::vector<int64_t>& decisions = *decision;
  std::unordered_set<const ffi::Object*> annotated;
  instructions.reserve(trace->decisions.size());
  decisions.reserve(trace->decisions.size());
  annotated.reserve(trace->decisions.size());
  // Find annotation with `meta_schedule_cooperative_fetch`
  for (const Instruction& inst : trace->insts) {
    if (inst->kind.same_as(inst_annotate)) {
      TVM_FFI_ICHECK_EQ(inst->attrs.size(), 1);
      TVM_FFI_ICHECK_EQ(inst->inputs.size(), 2);
      if (Downcast<ffi::String>(inst->attrs[0]) == s_tir::attr::meta_schedule_cooperative_fetch) {
        const auto* ann_val = inst->inputs[1].as<s_tir::ExprRVNode>();
        TVM_FFI_ICHECK(ann_val);
        annotated.insert(ann_val);
      }
    }
  }
  // Find sampling instruction that generates the annotation
  for (const auto& kv : trace->decisions) {
    const Instruction& inst = kv.first;
    if (inst->kind.same_as(inst_sample_categorical)) {
      TVM_FFI_ICHECK_EQ(inst->outputs.size(), 1);
      if (annotated.count(inst->outputs[0].as<ffi::Object>())) {
        TVM_FFI_ICHECK_EQ(inst->attrs.size(), 2);
        std::vector<double> probs =
            support::AsVector<FloatImm, double>(Downcast<ffi::Array<FloatImm>>(inst->attrs[1]));
        if (probs.size() == 1) {
          // Skip mutating the sampling instructions who have only single candidate.
          continue;
        }
        // SampleCategorical decision is Optional<int64_t> after the
        // Integer phase-out, so the Any holds a bare POD int.
        instructions.push_back(inst);
        decisions.push_back(kv.second.cast<int64_t>());
      }
    }
  }
}

struct FactorMemo {
  static std::vector<int> Factorize(int n) {
    if (const std::vector<int>* result = Global()->Query(n)) {
      return *result;
    }
    std::vector<int> result;
    for (int64_t i = 1; i * i <= n; ++i) {
      if (n % i == 0) {
        result.push_back(i);
        if (i * i != n) {
          result.push_back(n / i);
        }
      }
    }
    std::sort(result.begin(), result.end());
    Global()->Add(n, result);
    return result;
  }

 private:
  const std::vector<int>* Query(int n) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = memo_.find(n);
    if (it != memo_.end()) {
      return &it->second;
    }
    return nullptr;
  }

  void Add(int n, std::vector<int> result) {
    std::unique_lock<std::mutex> lock(mutex_);
    memo_.emplace(n, std::move(result));
  }

  static FactorMemo* Global() {
    static FactorMemo singleton;
    return &singleton;
  }

  std::unordered_map<int, std::vector<int>> memo_;
  std::mutex mutex_;
};

ffi::Optional<Trace> MutateSampleTileSize(const Trace& trace, Instruction inst,
                                          std::vector<int64_t> tiles, TRandState* rand_state,
                                          int ga_t2_mode, int64_t max_threads_per_block) {
  int n_splits = tiles.size();
  // Step 1. Choose two loops, `x` and `y`
  int x, y;
  // select source
  while (true) {
    x = s_tir::SampleInt(rand_state, 0, n_splits);
    if (tiles[x] <= 1) {
      continue;
    }
    y = s_tir::SampleInt(rand_state, 0, n_splits - 1);
    if (y >= x) {
      ++y;
    }
    std::vector<int> factors = FactorMemo::Factorize(tiles[x]);
    // Step 2. Choose the divide factor
    int max_factor_index = static_cast<int>(factors.size()) - 1;
    if (y == n_splits - 1) {
      int64_t limit = Downcast<IntImm>(inst->attrs[1])->value;
      for (; max_factor_index >= 1; max_factor_index--) {
        if (factors[max_factor_index] * tiles[y] <= limit) {
          break;
        }
      }
      if (max_factor_index == 0) {
        if (n_splits <= 2) {
          return std::nullopt;
        }
        // Failed on this dst_idx, try next one.
        continue;
      }
    }
    int factor_index = s_tir::SampleInt(rand_state, 1, max_factor_index + 1);
    auto make_trace_for = [&](int src, int dst, int64_t divide_factor) {
      std::vector<int64_t> new_tiles = tiles;
      new_tiles[src] /= divide_factor;
      new_tiles[dst] *= divide_factor;
      return trace->WithDecision(inst, ffi::Array<int64_t>(new_tiles.begin(), new_tiles.end()),
                                 /*remove_postproc=*/true);
    };
    auto make_trace = [&](int index) { return make_trace_for(x, y, factors[index]); };

    Trace proposal = make_trace(factor_index);
    if (ga_t2_mode == 0) {
      return proposal;
    }

    MutateTileThreadStats* stats = MutateTileThreadStats::Global();
    stats->proposal_total.fetch_add(1, std::memory_order_relaxed);
    if (max_threads_per_block <= 0) {
      stats->unknown.fetch_add(1, std::memory_order_relaxed);
      stats->proposal_unknown.fetch_add(1, std::memory_order_relaxed);
      return proposal;
    }
    ThreadExtentEstimate estimate = EstimateThreadsFromTrace(proposal);
    if (!estimate.known) {
      stats->unknown.fetch_add(1, std::memory_order_relaxed);
      stats->proposal_unknown.fetch_add(1, std::memory_order_relaxed);
      return proposal;
    }
    stats->checked.fetch_add(1, std::memory_order_relaxed);
    stats->proposal_known.fetch_add(1, std::memory_order_relaxed);
    bool initial_legal = estimate.threads <= max_threads_per_block;
    if (initial_legal) {
      stats->proposal_initial_legal.fetch_add(1, std::memory_order_relaxed);
    } else {
      stats->initial_reject.fetch_add(1, std::memory_order_relaxed);
    }
    if (ga_t2_mode == 1) {
      // Shadow mode is observational only.  In particular, it performs no
      // extra sampling and returns the exact proposal produced by mode 0.
      return proposal;
    }

    if (ga_t2_mode == 3) {
      ThreadExtentEstimate parent_estimate = EstimateThreadsFromTrace(trace);
      if (parent_estimate.known) {
        struct DirectionCandidate {
          Trace trace;
          int64_t threads;
          bool preferred;
        };
        std::vector<DirectionCandidate> direction_candidates;

        // Above 75% utilization, do not grow the physical thread extent.  Below
        // it, spend available thread margin.  We inspect divisor choices using
        // trace arithmetic only and return exactly one trace; EvolutionarySearch
        // therefore performs one replay/postprocess for this mutation.
        int64_t target_threads = max_threads_per_block - max_threads_per_block / 4;
        bool under_pressure = parent_estimate.threads >= target_threads;
        int64_t innermost_limit = Downcast<IntImm>(inst->attrs[1])->value;

        auto consider = [&](int src, int dst, int64_t divide_factor) {
          if (divide_factor <= 1 || tiles[src] % divide_factor != 0) return;
          if (dst == n_splits - 1 &&
              (tiles[dst] > innermost_limit || divide_factor > innermost_limit / tiles[dst])) {
            return;
          }
          Trace candidate = make_trace_for(src, dst, divide_factor);
          ThreadExtentEstimate candidate_estimate = EstimateThreadsFromTrace(candidate);
          stats->divisor_steps.fetch_add(1, std::memory_order_relaxed);
          if (!candidate_estimate.known || candidate_estimate.threads > max_threads_per_block) {
            return;
          }
          bool preferred = under_pressure ? candidate_estimate.threads <= parent_estimate.threads
                                          : candidate_estimate.threads >= parent_estimate.threads;
          direction_candidates.push_back(
              DirectionCandidate{candidate, candidate_estimate.threads, preferred});
        };

        // Keep the random proposal first so a resource-score tie retains the
        // baseline's stochastic choice.  Also inspect the reverse transfer:
        // this is what lets a high-pressure thread tile move factors outward
        // instead of merely rejecting an inward-growing proposal.
        consider(x, y, factors[factor_index]);
        for (int index = 1; index <= max_factor_index; ++index) {
          if (index != factor_index) consider(x, y, factors[index]);
        }
        if (tiles[y] > 1) {
          std::vector<int> reverse_factors = FactorMemo::Factorize(tiles[y]);
          for (int index = 1, n = static_cast<int>(reverse_factors.size()); index < n; ++index) {
            consider(y, x, reverse_factors[index]);
          }
        }
        stats->direction_candidates.fetch_add(direction_candidates.size(),
                                              std::memory_order_relaxed);

        bool has_preferred = false;
        int best_index = -1;
        int64_t best_distance = std::numeric_limits<int64_t>::max();
        for (int index = 0, n = static_cast<int>(direction_candidates.size()); index < n; ++index) {
          const DirectionCandidate& candidate = direction_candidates[index];
          if (has_preferred && !candidate.preferred) continue;
          int64_t distance = candidate.threads >= target_threads
                                 ? candidate.threads - target_threads
                                 : target_threads - candidate.threads;
          if ((candidate.preferred && !has_preferred) || distance < best_distance) {
            has_preferred = candidate.preferred;
            best_index = index;
            best_distance = distance;
          }
        }
        if (best_index != -1) {
          const DirectionCandidate& selected = direction_candidates[best_index];
          stats->directed_choices.fetch_add(1, std::memory_order_relaxed);
          if (selected.threads > parent_estimate.threads) {
            stats->slack_expand.fetch_add(1, std::memory_order_relaxed);
          }
          if (under_pressure && selected.threads <= parent_estimate.threads) {
            stats->pressure_nonincrease.fetch_add(1, std::memory_order_relaxed);
          }
          if (!initial_legal) {
            stats->projected.fetch_add(1, std::memory_order_relaxed);
          }
          return selected.trace;
        }
      }
      stats->direction_fallback.fetch_add(1, std::memory_order_relaxed);
      if (initial_legal) return proposal;
      // If provenance of the parent is unknown, retain the mode-2 repair for
      // a proposal whose own resource use is known to overflow.
    }

    if (initial_legal) return proposal;

    // Project along the sampled divisor lattice.  The first passing divisor
    // is the closest non-trivial mutation below the original proposal.
    for (int index = factor_index - 1; index >= 1; --index) {
      stats->divisor_steps.fetch_add(1, std::memory_order_relaxed);
      Trace projected = make_trace(index);
      ThreadExtentEstimate projected_estimate = EstimateThreadsFromTrace(projected);
      if (!projected_estimate.known) {
        return projected;
      }
      if (projected_estimate.threads <= max_threads_per_block) {
        stats->projected.fetch_add(1, std::memory_order_relaxed);
        return projected;
      }
    }
    stats->projection_failed.fetch_add(1, std::memory_order_relaxed);
    return std::nullopt;
  }
}

ffi::Optional<Trace> MutateSampleVectorize(const Trace& trace, Instruction inst,
                                           int64_t original_decision, TRandState* rand_state) {
  TVM_FFI_ICHECK_EQ(inst->attrs.size(), 2);
  std::vector<double> probs =
      support::AsVector<FloatImm, double>(Downcast<ffi::Array<FloatImm>>(inst->attrs[1]));
  probs.erase(probs.begin() + original_decision);
  int result = s_tir::MakeMultinomialSampler(rand_state, probs)();
  if (result >= original_decision) {
    result += 1;
  }
  return trace->WithDecision(inst, static_cast<int64_t>(result), /*remove_postproc=*/true);
}

ffi::Optional<Trace> MutateTileSizeNode::Apply(const Trace& trace, TRandState* rand_state) {
  std::vector<Instruction> sample_perfect_tile_insts;
  std::vector<Instruction> sample_vectorize_insts;
  std::vector<std::vector<int64_t>> sample_perfect_tile_tiles;
  std::vector<int64_t> sample_vectorize_decisions;
  FindSamplePerfectTile(trace, &sample_perfect_tile_insts, &sample_perfect_tile_tiles);
  FindSampleVectorize(trace, &sample_vectorize_insts, &sample_vectorize_decisions);
  int size_a = sample_perfect_tile_insts.size();
  int size_b = sample_vectorize_insts.size();
  if (size_a == 0 && size_b == 0) {
    return std::nullopt;
  }
  int n = s_tir::SampleInt(rand_state, 0, size_a + size_b);
  if (n < size_a) {
    return MutateSampleTileSize(trace, sample_perfect_tile_insts[n], sample_perfect_tile_tiles[n],
                                rand_state, this->ga_t2_mode_, this->max_threads_per_block_);
  } else {
    n -= size_a;
    return MutateSampleVectorize(trace, sample_vectorize_insts[n], sample_vectorize_decisions[n],
                                 rand_state);
  }
}

Mutator Mutator::MutateTileSize() { return Mutator(ffi::make_object<MutateTileSizeNode>()); }

TVM_FFI_STATIC_INIT_BLOCK() { MutateTileSizeNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.MutatorMutateTileSize", Mutator::MutateTileSize);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.meta_schedule.testing.GetMutateTileSizeThreadStats",
           []() {
             MutateTileThreadStats* stats = MutateTileThreadStats::Global();
             return ffi::Map<ffi::String, int64_t>{
                 {"proposal_total", stats->proposal_total.load(std::memory_order_relaxed)},
                 {"proposal_known", stats->proposal_known.load(std::memory_order_relaxed)},
                 {"proposal_unknown", stats->proposal_unknown.load(std::memory_order_relaxed)},
                 {"proposal_initial_legal",
                  stats->proposal_initial_legal.load(std::memory_order_relaxed)},
                 {"checked", stats->checked.load(std::memory_order_relaxed)},
                 {"unknown", stats->unknown.load(std::memory_order_relaxed)},
                 {"initial_reject", stats->initial_reject.load(std::memory_order_relaxed)},
                 {"projected", stats->projected.load(std::memory_order_relaxed)},
                 {"projection_failed", stats->projection_failed.load(std::memory_order_relaxed)},
                 {"direction_candidates",
                  stats->direction_candidates.load(std::memory_order_relaxed)},
                 {"directed_choices", stats->directed_choices.load(std::memory_order_relaxed)},
                 {"direction_fallback", stats->direction_fallback.load(std::memory_order_relaxed)},
                 {"slack_expand", stats->slack_expand.load(std::memory_order_relaxed)},
                 {"pressure_nonincrease",
                  stats->pressure_nonincrease.load(std::memory_order_relaxed)},
                 {"divisor_steps", stats->divisor_steps.load(std::memory_order_relaxed)},
             };
           })
      .def("s_tir.meta_schedule.testing.ResetMutateTileSizeThreadStats",
           []() { MutateTileThreadStats::Global()->Reset(); });
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

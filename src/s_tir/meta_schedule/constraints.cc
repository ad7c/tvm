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

#include "constraints.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/schedule/instruction.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../runtime/thread_storage_scope.h"
#include "../schedule/primitive.h"
#include "../schedule/utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using namespace tvm::tirx;

namespace {

using s_tir::Instruction;
using s_tir::InstructionKind;
using s_tir::Trace;

struct TraceLoopInfo {
  int block_id{-1};
  TraceSymbolicProduct extent;
};

struct TraceBlockInfo {
  std::string key;
  std::array<TraceSymbolicProduct, 3> axis_products;
  std::array<bool, 3> axis_present{{false, false, false}};
  bool scope_known{true};
  bool contract_present{false};
  bool has_low{false};
  bool has_high{false};
  bool warp_execution{false};
  int64_t low{-1};
  int64_t high{-1};
  TraceConstraintUnknownReason unknown_reason{TraceConstraintUnknownReason::kNone};
};

bool SafeMultiplyPositive(int64_t lhs, int64_t rhs, int64_t* result) {
  if (lhs <= 0 || rhs <= 0) return false;
  if (lhs > std::numeric_limits<int64_t>::max() / rhs) return false;
  *result = lhs * rhs;
  return true;
}

bool TryGetInt(const Any& value, int64_t* result) {
  if (value == nullptr) return false;
  if (auto integer = value.try_cast<int64_t>()) {
    *result = integer.value();
    return true;
  }
  if (auto integer = value.try_cast<IntImm>()) {
    *result = integer.value()->value;
    return true;
  }
  return false;
}

int TraceThreadAxis(const ffi::String& tag) {
  if (tag == "threadIdx.x") return 0;
  if (tag == "threadIdx.y") return 1;
  if (tag == "threadIdx.z") return 2;
  return -1;
}

void RecordPlanUnknown(TraceConstraintPlan* plan, TraceConstraintUnknownReason reason) {
  if (reason == TraceConstraintUnknownReason::kNone) return;
  plan->unknown_reason_counts[static_cast<size_t>(reason)]++;
}

TraceSymbolicProduct ConstantProduct(int64_t value) {
  TraceSymbolicProduct result;
  result.known = value > 0;
  result.constant = value;
  return result;
}

TraceSymbolicProduct FactorProduct(int instruction_index, int decision_index) {
  TraceSymbolicProduct result;
  result.known = true;
  result.constant = 1;
  result.factors.push_back(TraceFactorRef{instruction_index, decision_index});
  return result;
}

bool AppendProduct(const TraceSymbolicProduct& value, TraceSymbolicProduct* result) {
  if (!value.known || !result->known) {
    result->known = false;
    return false;
  }
  int64_t product = -1;
  if (!SafeMultiplyPositive(result->constant, value.constant, &product)) {
    result->known = false;
    return false;
  }
  result->constant = product;
  result->factors.insert(result->factors.end(), value.factors.begin(), value.factors.end());
  return true;
}

TraceSymbolicProduct ResolveSymbolicProduct(
    const Any& value,
    const std::unordered_map<const ffi::Object*, TraceSymbolicProduct>& expression_products) {
  int64_t integer = -1;
  if (TryGetInt(value, &integer)) return ConstantProduct(integer);
  if (const ffi::Object* object = value.as<ffi::Object>()) {
    auto it = expression_products.find(object);
    if (it != expression_products.end()) return it->second;
  }
  return TraceSymbolicProduct{};
}

const TraceSampleSpec* FindSampleSpec(const TraceConstraintPlan& plan, int instruction_index) {
  for (const TraceSampleSpec& sample : plan.samples) {
    if (sample.instruction_index == instruction_index) return &sample;
  }
  return nullptr;
}

bool GetPerfectTileDecision(const Any& decision, std::vector<int64_t>* values) {
  if (decision == nullptr) return false;
  if (auto array = decision.try_cast<ffi::Array<int64_t>>()) {
    values->assign(array.value().begin(), array.value().end());
    return true;
  }
  return false;
}

bool ResolveFactor(const TraceConstraintPlan& plan, const TraceFactorRef& factor,
                   const Trace& trace, int overlay_instruction_index, const Any& overlay_decision,
                   int64_t* value) {
  const TraceSampleSpec* sample = FindSampleSpec(plan, factor.instruction_index);
  if (sample == nullptr) return false;

  Any decision{nullptr};
  if (factor.instruction_index == overlay_instruction_index) {
    decision = overlay_decision;
  } else {
    if (factor.instruction_index < 0 ||
        factor.instruction_index >= static_cast<int>(trace->insts.size())) {
      return false;
    }
    decision = trace->GetDecision(trace->insts[factor.instruction_index]);
  }
  if (decision == nullptr) return false;

  if (sample->kind == TraceSampleSpec::Kind::kPerfectTile) {
    std::vector<int64_t> tiles;
    if (!GetPerfectTileDecision(decision, &tiles) || factor.decision_index < 0 ||
        factor.decision_index >= static_cast<int>(tiles.size()) ||
        tiles[factor.decision_index] <= 0) {
      return false;
    }
    *value = tiles[factor.decision_index];
    return true;
  }

  int64_t index = -1;
  if (!TryGetInt(decision, &index) || factor.decision_index != 0 || index < 0 ||
      index >= static_cast<int64_t>(sample->categorical_candidates.size())) {
    return false;
  }
  *value = sample->categorical_candidates[index];
  return *value > 0;
}

bool ResolveProduct(const TraceConstraintPlan& plan, const TraceSymbolicProduct& product,
                    const Trace& trace, int overlay_instruction_index,
                    const Any& overlay_decision, int64_t* value) {
  if (!product.known || product.constant <= 0) return false;
  int64_t result = product.constant;
  for (const TraceFactorRef& factor : product.factors) {
    int64_t factor_value = -1;
    if (!ResolveFactor(plan, factor, trace, overlay_instruction_index, overlay_decision,
                       &factor_value) ||
        !SafeMultiplyPositive(result, factor_value, &result)) {
      return false;
    }
  }
  *value = result;
  return true;
}

bool ResolveThreadProduct(const TraceConstraintPlan& plan,
                          const TraceBlockThreadConstraint& constraint, const Trace& trace,
                          int overlay_instruction_index, const Any& overlay_decision,
                          int64_t* threads) {
  if (!constraint.scope_known) return false;
  int64_t result = 1;
  for (int axis = 0; axis < 3; ++axis) {
    if (!constraint.axis_present[axis]) continue;
    int64_t extent = -1;
    if (!ResolveProduct(plan, constraint.axis_products[axis], trace, overlay_instruction_index,
                        overlay_decision, &extent) ||
        !SafeMultiplyPositive(result, extent, &result)) {
      return false;
    }
  }
  *threads = result;
  return true;
}

enum class ConstraintCheck : int {
  kAllowed = 0,
  kRejected,
  kUnknown,
};

ConstraintCheck CheckConstraintWithOverlay(const TraceConstraintPlan& plan,
                                           const TraceBlockThreadConstraint& constraint,
                                           const Trace& prefix_trace,
                                           int overlay_instruction_index,
                                           const Any& overlay_decision) {
  int64_t threads = -1;
  if (!ResolveThreadProduct(plan, constraint, prefix_trace, overlay_instruction_index,
                            overlay_decision, &threads)) {
    return ConstraintCheck::kUnknown;
  }
  if (constraint.hardware_limit > 0 && threads > constraint.hardware_limit) {
    return ConstraintCheck::kRejected;
  }
  if (constraint.contract_complete &&
      (threads < constraint.low_inclusive || threads > constraint.high_inclusive)) {
    return ConstraintCheck::kRejected;
  }
  return ConstraintCheck::kAllowed;
}

void GetDivisors(int64_t value, std::vector<int64_t>* divisors) {
  divisors->clear();
  for (int64_t factor = 1; factor <= value / factor; ++factor) {
    if (value % factor != 0) continue;
    divisors->push_back(factor);
    if (factor != value / factor) divisors->push_back(value / factor);
  }
  std::sort(divisors->begin(), divisors->end());
}

void EnumeratePerfectTiles(int64_t remaining, int position, int n_splits,
                           int max_innermost_factor, int64_t enumeration_limit,
                           std::vector<int64_t>* current,
                           std::vector<std::vector<int64_t>>* output, int64_t* enumerated,
                           bool* overflow) {
  if (*overflow) return;
  if (position == n_splits - 1) {
    if (remaining <= 0 ||
        (max_innermost_factor != -1 && remaining > max_innermost_factor)) {
      return;
    }
    (*enumerated)++;
    if (*enumerated > enumeration_limit) {
      *overflow = true;
      output->clear();
      return;
    }
    (*current)[position] = remaining;
    output->push_back(*current);
    return;
  }

  std::vector<int64_t> divisors;
  GetDivisors(remaining, &divisors);
  for (int64_t factor : divisors) {
    (*current)[position] = factor;
    EnumeratePerfectTiles(remaining / factor, position + 1, n_splits,
                          max_innermost_factor, enumeration_limit, current, output, enumerated,
                          overflow);
    if (*overflow) return;
  }
}

class PhysicalThreadCollector : private StmtVisitor {
 public:
  static ThreadFeasibilityResult Collect(const PrimFunc& func, int64_t limit,
                                         int64_t thread_warp_size,
                                         int64_t shared_memory_limit) {
    PhysicalThreadCollector collector(limit, thread_warp_size, shared_memory_limit);
    collector(func->body);
    return ThreadFeasibilityResult{
        /*all_extents_known=*/collector.all_extents_known_,
        /*all_contracts_known=*/collector.all_contracts_known_,
        /*reject=*/collector.reject_thread_extent_low_ ||
            collector.reject_thread_extent_high_ || collector.reject_threads_per_block_,
        /*reject_thread_extent_low=*/collector.reject_thread_extent_low_,
        /*reject_thread_extent_high=*/collector.reject_thread_extent_high_,
        /*reject_threads_per_block=*/collector.reject_threads_per_block_,
        /*max_threads=*/collector.all_extents_known_ ? collector.max_threads_
                                                     : static_cast<int64_t>(-1),
        /*all_shared_allocations_known=*/collector.all_shared_allocations_known_,
        /*shared_memory_proxy_reject=*/collector.shared_memory_proxy_reject_,
        /*max_declared_shared_bytes=*/collector.max_declared_shared_bytes_};
  }

 private:
  PhysicalThreadCollector(int64_t limit, int64_t thread_warp_size,
                          int64_t shared_memory_limit)
      : limit_(limit),
        thread_warp_size_(thread_warp_size),
        shared_memory_limit_(shared_memory_limit) {}

  static int AxisIndex(const ffi::String& tag) {
    if (tag == "threadIdx.x") return 0;
    if (tag == "threadIdx.y") return 1;
    if (tag == "threadIdx.z") return 2;
    return -1;
  }

  static bool IsUnsupportedPhysicalAxis(const ffi::String& tag) {
    return std::string(tag).rfind("threadIdx.", 0) == 0;
  }

  void MarkUnknownAndVisit(const Stmt& body) {
    all_extents_known_ = false;
    ++unknown_depth_;
    VisitStmt(body);
    --unknown_depth_;
  }

  void VisitPhysicalScope(const ffi::String& tag, const PrimExpr& extent, const Stmt& body) {
    int axis = AxisIndex(tag);
    if (axis == -1) {
      if (IsUnsupportedPhysicalAxis(tag)) {
        MarkUnknownAndVisit(body);
      } else {
        VisitStmt(body);
      }
      return;
    }

    const int64_t* value = as_const_int(extent);
    if (value == nullptr || *value <= 0) {
      MarkUnknownAndVisit(body);
      return;
    }

    if (axis_active_[axis]) {
      if (axis_extent_[axis] != *value) {
        MarkUnknownAndVisit(body);
      } else {
        VisitStmt(body);
      }
      return;
    }

    axis_active_[axis] = true;
    axis_extent_[axis] = *value;
    UpdateProduct();
    VisitStmt(body);
    axis_active_[axis] = false;
    axis_extent_[axis] = 1;
  }

  int64_t CurrentProduct() const {
    int64_t product = 1;
    for (int axis = 0; axis < 3; ++axis) {
      if (!axis_active_[axis]) continue;
      int64_t extent = axis_extent_[axis];
      if (product > std::numeric_limits<int64_t>::max() / extent) {
        product = std::numeric_limits<int64_t>::max();
        break;
      }
      product *= extent;
    }
    return product;
  }

  void UpdateProduct() {
    if (unknown_depth_ != 0) return;
    int64_t product = CurrentProduct();
    if (product > max_threads_) max_threads_ = product;
    if (limit_ > 0 && product > limit_) reject_threads_per_block_ = true;
  }

  void CheckThreadContract(const SBlockNode* block) {
    ffi::Optional<int64_t> low =
        GetAnn<int64_t>(block, s_tir::attr::meta_schedule_thread_extent_low_inclusive);
    ffi::Optional<int64_t> high =
        GetAnn<int64_t>(block, s_tir::attr::meta_schedule_thread_extent_high_inclusive);
    if (!low.has_value() && !high.has_value()) return;
    if (!low.has_value() || !high.has_value() || unknown_depth_ != 0) {
      all_contracts_known_ = false;
      return;
    }
    int64_t product = CurrentProduct();
    if (product < low.value()) reject_thread_extent_low_ = true;
    if (product > high.value()) reject_thread_extent_high_ = true;
  }

  static int64_t SaturatingAdd(int64_t lhs, int64_t rhs) {
    if (rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) {
      return std::numeric_limits<int64_t>::max();
    }
    return lhs + rhs;
  }

  int64_t DeclaredSharedBytes(const SBlockNode* block) {
    int64_t bytes = 0;
    for (const Buffer& buffer : block->alloc_buffers) {
      runtime::StorageScope scope = runtime::StorageScope::Create(buffer.scope());
      if (scope.rank != runtime::StorageRank::kShared) continue;
      int64_t elements = 1;
      for (const PrimExpr& extent : buffer->shape) {
        const int64_t* value = as_const_int(extent);
        if (value == nullptr || *value < 0) {
          all_shared_allocations_known_ = false;
          elements = 0;
          break;
        }
        if (*value != 0 && elements > std::numeric_limits<int64_t>::max() / *value) {
          elements = std::numeric_limits<int64_t>::max();
          break;
        }
        elements *= *value;
      }
      int64_t element_bytes = buffer->dtype.bytes() * buffer->dtype.lanes();
      if (elements != 0 && element_bytes > 0 &&
          elements > std::numeric_limits<int64_t>::max() / element_bytes) {
        bytes = std::numeric_limits<int64_t>::max();
      } else {
        bytes = SaturatingAdd(bytes, elements * element_bytes);
      }
    }
    return bytes;
  }

  void VisitStmt_(const ForNode* loop) final {
    if (loop->kind == ForKind::kThreadBinding) {
      if (!loop->thread_binding.has_value()) {
        MarkUnknownAndVisit(loop->body);
        return;
      }
      VisitPhysicalScope(loop->thread_binding.value()->thread_tag, loop->extent, loop->body);
      return;
    }
    StmtVisitor::VisitStmt_(loop);
  }

  void VisitStmt_(const AttrStmtNode* attr_stmt) final {
    if (attr_stmt->attr_key != tirx::attr::thread_extent) {
      StmtVisitor::VisitStmt_(attr_stmt);
      return;
    }

    ffi::String tag;
    if (const auto* iter_var = attr_stmt->node.as<IterVarNode>()) {
      tag = iter_var->thread_tag.empty() ? iter_var->var->name_hint : iter_var->thread_tag;
    } else if (const auto* var = attr_stmt->node.as<VarNode>()) {
      tag = var->name_hint;
    } else {
      MarkUnknownAndVisit(attr_stmt->body);
      return;
    }
    VisitPhysicalScope(tag, attr_stmt->value, attr_stmt->body);
  }

  void VisitStmt_(const SBlockNode* block) final {
    int64_t old_shared_bytes = current_declared_shared_bytes_;
    current_declared_shared_bytes_ =
        SaturatingAdd(current_declared_shared_bytes_, DeclaredSharedBytes(block));
    max_declared_shared_bytes_ =
        std::max(max_declared_shared_bytes_, current_declared_shared_bytes_);
    if (shared_memory_limit_ > 0 &&
        current_declared_shared_bytes_ > shared_memory_limit_) {
      shared_memory_proxy_reject_ = true;
    }

    bool has_warp_execution = block->annotations.count(s_tir::attr::warp_execution);
    if (has_warp_execution &&
        (thread_warp_size_ <= 0 ||
         (axis_active_[0] && axis_extent_[0] != thread_warp_size_))) {
      all_extents_known_ = false;
      all_contracts_known_ = false;
      ++unknown_depth_;
      StmtVisitor::VisitStmt_(block);
      --unknown_depth_;
      current_declared_shared_bytes_ = old_shared_bytes;
      return;
    }

    bool pushed_warp_axis = has_warp_execution && !axis_active_[0];
    if (pushed_warp_axis) {
      axis_active_[0] = true;
      axis_extent_[0] = thread_warp_size_;
      UpdateProduct();
    }
    CheckThreadContract(block);
    StmtVisitor::VisitStmt_(block);
    if (pushed_warp_axis) {
      axis_active_[0] = false;
      axis_extent_[0] = 1;
    }
    current_declared_shared_bytes_ = old_shared_bytes;
  }

  int64_t limit_;
  int64_t thread_warp_size_;
  int64_t shared_memory_limit_;
  std::array<bool, 3> axis_active_{{false, false, false}};
  std::array<int64_t, 3> axis_extent_{{1, 1, 1}};
  int unknown_depth_{0};
  bool all_extents_known_{true};
  bool all_contracts_known_{true};
  bool reject_thread_extent_low_{false};
  bool reject_thread_extent_high_{false};
  bool reject_threads_per_block_{false};
  int64_t max_threads_{1};
  bool all_shared_allocations_known_{true};
  bool shared_memory_proxy_reject_{false};
  int64_t current_declared_shared_bytes_{0};
  int64_t max_declared_shared_bytes_{0};
};

}  // namespace

const char* TraceConstraintUnknownReasonName(TraceConstraintUnknownReason reason) {
  switch (reason) {
    case TraceConstraintUnknownReason::kNone:
      return "none";
    case TraceConstraintUnknownReason::kMissingDecision:
      return "missing_decision";
    case TraceConstraintUnknownReason::kDynamicExtent:
      return "dynamic_extent";
    case TraceConstraintUnknownReason::kPartialContract:
      return "partial_contract";
    case TraceConstraintUnknownReason::kUnresolvedBlockScope:
      return "unresolved_block_scope";
    case TraceConstraintUnknownReason::kConflictingAxisExtent:
      return "conflicting_axis_extent";
    case TraceConstraintUnknownReason::kUnsupportedInstruction:
      return "unsupported_instruction";
    case TraceConstraintUnknownReason::kOverlappingConstraints:
      return "overlapping_constraints";
    case TraceConstraintUnknownReason::kWarpSizeMissing:
      return "warp_size_missing";
    case TraceConstraintUnknownReason::kNoContractAnnotation:
      return "no_contract_annotation";
    case TraceConstraintUnknownReason::kEnumerationLimit:
      return "enumeration_limit";
    case TraceConstraintUnknownReason::kCount:
      break;
  }
  return "unknown";
}

TraceConstraintPlan BuildTraceConstraintPlan(const Trace& trace, int64_t max_threads_per_block,
                                             int64_t thread_warp_size) {
  static const InstructionKind& inst_get_sblock = InstructionKind::Get("GetSBlock");
  static const InstructionKind& inst_get_loops = InstructionKind::Get("GetLoops");
  static const InstructionKind& inst_sample_perfect_tile =
      InstructionKind::Get("SamplePerfectTile");
  static const InstructionKind& inst_sample_categorical =
      InstructionKind::Get("SampleCategorical");
  static const InstructionKind& inst_split = InstructionKind::Get("Split");
  static const InstructionKind& inst_fuse = InstructionKind::Get("Fuse");
  static const InstructionKind& inst_bind = InstructionKind::Get("Bind");
  static const InstructionKind& inst_annotate = InstructionKind::Get("Annotate");
  static const InstructionKind& inst_unannotate = InstructionKind::Get("Unannotate");

  TraceConstraintPlan plan;
  std::map<std::string, int> block_key_to_id;
  std::vector<TraceBlockInfo> blocks;
  std::unordered_map<const ffi::Object*, int> block_rv_to_id;
  std::unordered_map<const ffi::Object*, TraceLoopInfo> loops;
  std::unordered_map<const ffi::Object*, TraceSymbolicProduct> expression_products;

  auto block_id_for_key = [&](const std::string& key) -> int {
    auto it = block_key_to_id.find(key);
    if (it != block_key_to_id.end()) return it->second;
    int block_id = static_cast<int>(blocks.size());
    block_key_to_id.emplace(key, block_id);
    TraceBlockInfo info;
    info.key = key;
    blocks.push_back(std::move(info));
    return block_id;
  };

  int instruction_index = 0;
  for (const Instruction& inst : trace->insts) {
    if (inst->kind->IsPostproc()) break;

    if (inst->kind.same_as(inst_get_sblock)) {
      if (inst->attrs.size() == 2 && inst->outputs.size() == 1) {
        ffi::String name = inst->attrs[0].cast<ffi::String>();
        ffi::String func_name = inst->attrs[1].cast<ffi::String>();
        std::string key = std::string(func_name) + "::" + std::string(name);
        int block_id = block_id_for_key(key);
        if (const ffi::Object* output = inst->outputs[0].as<ffi::Object>()) {
          block_rv_to_id[output] = block_id;
        }
      }
      ++instruction_index;
      continue;
    }

    if (inst->kind.same_as(inst_get_loops)) {
      if (inst->inputs.size() == 1) {
        const ffi::Object* block_rv = inst->inputs[0].as<ffi::Object>();
        auto block_it = block_rv_to_id.find(block_rv);
        if (block_it != block_rv_to_id.end()) {
          for (const Any& output_value : inst->outputs) {
            if (const ffi::Object* output = output_value.as<ffi::Object>()) {
              loops[output] = TraceLoopInfo{/*block_id=*/block_it->second,
                                            /*extent=*/TraceSymbolicProduct{}};
            }
          }
        } else {
          RecordPlanUnknown(&plan, TraceConstraintUnknownReason::kUnresolvedBlockScope);
        }
      }
      ++instruction_index;
      continue;
    }

    if (inst->kind.same_as(inst_sample_perfect_tile)) {
      TraceSampleSpec sample;
      sample.kind = TraceSampleSpec::Kind::kPerfectTile;
      sample.instruction_index = instruction_index;
      int64_t n_splits = -1;
      int64_t max_innermost_factor = -1;
      if (inst->attrs.size() == 2 && TryGetInt(inst->attrs[0], &n_splits) &&
          TryGetInt(inst->attrs[1], &max_innermost_factor) && n_splits > 0) {
        sample.n_splits = static_cast<int>(n_splits);
        sample.max_innermost_factor = static_cast<int>(max_innermost_factor);
      } else {
        RecordPlanUnknown(&plan, TraceConstraintUnknownReason::kUnsupportedInstruction);
      }
      plan.samples.push_back(std::move(sample));
      for (int i = 0, n = static_cast<int>(inst->outputs.size()); i < n; ++i) {
        if (const ffi::Object* output = inst->outputs[i].as<ffi::Object>()) {
          expression_products[output] = FactorProduct(instruction_index, i);
        }
      }
      ++instruction_index;
      continue;
    }

    if (inst->kind.same_as(inst_sample_categorical)) {
      TraceSampleSpec sample;
      sample.kind = TraceSampleSpec::Kind::kCategorical;
      sample.instruction_index = instruction_index;
      if (inst->attrs.size() == 2) {
        ffi::Array<int64_t> candidates = inst->attrs[0].cast<ffi::Array<int64_t>>();
        ffi::Array<FloatImm> probabilities = inst->attrs[1].cast<ffi::Array<FloatImm>>();
        sample.categorical_candidates.assign(candidates.begin(), candidates.end());
        sample.categorical_probabilities.reserve(probabilities.size());
        for (const FloatImm& probability : probabilities) {
          sample.categorical_probabilities.push_back(probability->value);
        }
      }
      plan.samples.push_back(std::move(sample));
      if (inst->outputs.size() == 1) {
        if (const ffi::Object* output = inst->outputs[0].as<ffi::Object>()) {
          expression_products[output] = FactorProduct(instruction_index, 0);
        }
      }
      ++instruction_index;
      continue;
    }

    if (inst->kind.same_as(inst_split)) {
      int n = static_cast<int>(inst->outputs.size());
      if (inst->inputs.size() == static_cast<size_t>(n + 1)) {
        const ffi::Object* input_loop = inst->inputs[0].as<ffi::Object>();
        auto loop_it = loops.find(input_loop);
        if (loop_it != loops.end()) {
          for (int i = 0; i < n; ++i) {
            if (const ffi::Object* output = inst->outputs[i].as<ffi::Object>()) {
              loops[output] = TraceLoopInfo{
                  /*block_id=*/loop_it->second.block_id,
                  /*extent=*/ResolveSymbolicProduct(inst->inputs[i + 1], expression_products)};
            }
          }
        }
      }
      ++instruction_index;
      continue;
    }

    if (inst->kind.same_as(inst_fuse)) {
      TraceLoopInfo fused;
      fused.extent.known = true;
      fused.extent.constant = 1;
      bool known = !inst->inputs.empty() && inst->outputs.size() == 1;
      for (const Any& input_value : inst->inputs) {
        const ffi::Object* input = input_value.as<ffi::Object>();
        auto loop_it = loops.find(input);
        if (loop_it == loops.end()) {
          known = false;
          break;
        }
        if (fused.block_id == -1) {
          fused.block_id = loop_it->second.block_id;
        } else if (fused.block_id != loop_it->second.block_id) {
          known = false;
          RecordPlanUnknown(&plan, TraceConstraintUnknownReason::kOverlappingConstraints);
          break;
        }
        if (!AppendProduct(loop_it->second.extent, &fused.extent)) known = false;
      }
      if (const ffi::Object* output =
              inst->outputs.size() == 1 ? inst->outputs[0].as<ffi::Object>() : nullptr) {
        if (known) {
          loops[output] = std::move(fused);
        } else {
          loops[output] = TraceLoopInfo{};
        }
      }
      ++instruction_index;
      continue;
    }

    if (inst->kind.same_as(inst_bind)) {
      if (inst->inputs.size() == 1 && inst->attrs.size() == 1) {
        ffi::String tag = inst->attrs[0].cast<ffi::String>();
        int axis = TraceThreadAxis(tag);
        if (axis != -1) {
          const ffi::Object* loop = inst->inputs[0].as<ffi::Object>();
          auto loop_it = loops.find(loop);
          if (loop_it == loops.end() || loop_it->second.block_id < 0) {
            RecordPlanUnknown(&plan, TraceConstraintUnknownReason::kUnresolvedBlockScope);
          } else {
            TraceBlockInfo& block = blocks[loop_it->second.block_id];
            if (block.axis_present[axis]) {
              block.scope_known = false;
              block.unknown_reason = TraceConstraintUnknownReason::kConflictingAxisExtent;
            } else {
              block.axis_present[axis] = true;
              block.axis_products[axis] = loop_it->second.extent;
              if (!loop_it->second.extent.known) {
                block.scope_known = false;
                block.unknown_reason = TraceConstraintUnknownReason::kDynamicExtent;
              }
            }
          }
        }
      }
      ++instruction_index;
      continue;
    }

    if (inst->kind.same_as(inst_annotate)) {
      if (inst->inputs.size() == 2 && inst->attrs.size() == 1) {
        ffi::String key = inst->attrs[0].cast<ffi::String>();
        bool relevant = key == s_tir::attr::meta_schedule_thread_extent_low_inclusive ||
                        key == s_tir::attr::meta_schedule_thread_extent_high_inclusive ||
                        key == s_tir::attr::warp_execution;
        if (relevant) {
          const ffi::Object* target = inst->inputs[0].as<ffi::Object>();
          auto block_it = block_rv_to_id.find(target);
          if (block_it == block_rv_to_id.end()) {
            RecordPlanUnknown(&plan, TraceConstraintUnknownReason::kUnresolvedBlockScope);
          } else {
            TraceBlockInfo& block = blocks[block_it->second];
            if (key == s_tir::attr::meta_schedule_thread_extent_low_inclusive) {
              block.contract_present = true;
              block.has_low = TryGetInt(inst->inputs[1], &block.low);
            } else if (key == s_tir::attr::meta_schedule_thread_extent_high_inclusive) {
              block.contract_present = true;
              block.has_high = TryGetInt(inst->inputs[1], &block.high);
            } else {
              block.warp_execution = true;
            }
          }
        }
      }
      ++instruction_index;
      continue;
    }

    if (inst->kind.same_as(inst_unannotate)) {
      if (inst->inputs.size() == 1 && inst->attrs.size() == 1) {
        const ffi::Object* target = inst->inputs[0].as<ffi::Object>();
        auto block_it = block_rv_to_id.find(target);
        if (block_it != block_rv_to_id.end()) {
          TraceBlockInfo& block = blocks[block_it->second];
          ffi::String key = inst->attrs[0].cast<ffi::String>();
          if (key == s_tir::attr::meta_schedule_thread_extent_low_inclusive) {
            block.contract_present = block.has_high;
            block.has_low = false;
          } else if (key == s_tir::attr::meta_schedule_thread_extent_high_inclusive) {
            block.contract_present = block.has_low;
            block.has_high = false;
          } else if (key == s_tir::attr::warp_execution) {
            block.warp_execution = false;
          }
        }
      }
      ++instruction_index;
      continue;
    }

    ++instruction_index;
  }

  for (TraceBlockInfo& block : blocks) {
    bool has_axis = block.axis_present[0] || block.axis_present[1] || block.axis_present[2];
    if (!has_axis && !block.contract_present && !block.warp_execution) continue;

    TraceBlockThreadConstraint constraint;
    constraint.block_key = block.key;
    constraint.axis_products = block.axis_products;
    constraint.axis_present = block.axis_present;
    constraint.scope_known = block.scope_known;
    constraint.contract_present = block.contract_present;
    constraint.contract_complete = block.has_low && block.has_high;
    constraint.low_inclusive = block.low;
    constraint.high_inclusive = block.high;
    constraint.hardware_limit = max_threads_per_block;
    constraint.warp_execution = block.warp_execution;
    constraint.unknown_reason = block.unknown_reason;

    if (block.contract_present) {
      plan.contract_blocks++;
      if (!constraint.contract_complete) {
        constraint.unknown_reason = TraceConstraintUnknownReason::kPartialContract;
        RecordPlanUnknown(&plan, TraceConstraintUnknownReason::kPartialContract);
      }
    } else {
      RecordPlanUnknown(&plan, TraceConstraintUnknownReason::kNoContractAnnotation);
    }

    if (constraint.warp_execution) {
      if (thread_warp_size <= 0) {
        constraint.scope_known = false;
        constraint.unknown_reason = TraceConstraintUnknownReason::kWarpSizeMissing;
        RecordPlanUnknown(&plan, TraceConstraintUnknownReason::kWarpSizeMissing);
      } else if (constraint.axis_present[0]) {
        constraint.scope_known = false;
        constraint.unknown_reason = TraceConstraintUnknownReason::kConflictingAxisExtent;
        RecordPlanUnknown(&plan, TraceConstraintUnknownReason::kConflictingAxisExtent);
      } else {
        constraint.axis_present[0] = true;
        constraint.axis_products[0] = ConstantProduct(thread_warp_size);
      }
    }

    if (!constraint.scope_known) {
      TraceConstraintUnknownReason reason = constraint.unknown_reason;
      if (reason == TraceConstraintUnknownReason::kNone) {
        reason = TraceConstraintUnknownReason::kUnresolvedBlockScope;
        constraint.unknown_reason = reason;
      }
      RecordPlanUnknown(&plan, reason);
    }

    int conditional_instruction_index = -1;
    for (int axis = 0; axis < 3; ++axis) {
      if (!constraint.axis_present[axis]) continue;
      for (const TraceFactorRef& factor : constraint.axis_products[axis].factors) {
        conditional_instruction_index =
            std::max(conditional_instruction_index, factor.instruction_index);
      }
    }
    constraint.conditional_instruction_index = conditional_instruction_index;

    if (constraint.scope_known &&
        (constraint.contract_complete || constraint.hardware_limit > 0)) {
      plan.enforceable_constraints++;
    }
    plan.constraints.push_back(std::move(constraint));
  }
  return plan;
}

TraceConstraintEvaluation EvaluateTraceConstraints(const TraceConstraintPlan& plan,
                                                    const Trace& trace) {
  TraceConstraintEvaluation result;
  bool any_check = false;
  bool any_unknown = false;
  bool any_product = false;
  result.max_threads = 1;

  for (const TraceBlockThreadConstraint& constraint : plan.constraints) {
    bool checkable = constraint.contract_complete || constraint.hardware_limit > 0;
    if (!checkable) {
      any_unknown = true;
      if (result.unknown_reason == TraceConstraintUnknownReason::kNone) {
        result.unknown_reason = constraint.contract_present
                                    ? TraceConstraintUnknownReason::kPartialContract
                                    : TraceConstraintUnknownReason::kNoContractAnnotation;
      }
      continue;
    }
    int64_t threads = -1;
    if (!ResolveThreadProduct(plan, constraint, trace, /*overlay_instruction_index=*/-1,
                              /*overlay_decision=*/Any{nullptr}, &threads)) {
      any_unknown = true;
      if (result.unknown_reason == TraceConstraintUnknownReason::kNone) {
        result.unknown_reason = constraint.unknown_reason != TraceConstraintUnknownReason::kNone
                                    ? constraint.unknown_reason
                                    : TraceConstraintUnknownReason::kMissingDecision;
      }
      continue;
    }

    any_product = true;
    any_check = true;
    result.max_threads = std::max(result.max_threads, threads);
    if (constraint.hardware_limit > 0 && threads > constraint.hardware_limit) {
      result.reject_threads_per_block = true;
    }
    if (constraint.contract_complete) {
      if (threads < constraint.low_inclusive) result.reject_thread_extent_low = true;
      if (threads > constraint.high_inclusive) result.reject_thread_extent_high = true;
    }
  }

  if (result.reject_thread_extent_low || result.reject_thread_extent_high ||
      result.reject_threads_per_block) {
    result.truth = TraceConstraintTruth::kProvenInfeasible;
  } else if (!any_check || any_unknown) {
    result.truth = TraceConstraintTruth::kUnknown;
    if (result.unknown_reason == TraceConstraintUnknownReason::kNone) {
      result.unknown_reason = TraceConstraintUnknownReason::kNoContractAnnotation;
    }
  } else {
    result.truth = TraceConstraintTruth::kProvenFeasible;
  }
  if (!any_product) result.max_threads = -1;
  return result;
}

TraceConstraintDecisionResult SampleConstrainedTraceDecision(
    const TraceConstraintPlan& plan, int instruction_index, const Trace& prefix_trace,
    int64_t loop_extent, LinearCongruentialEngine::TRandState* rand_state,
    int64_t enumeration_limit) {
  TraceConstraintDecisionResult result;
  std::vector<const TraceBlockThreadConstraint*> active_constraints;
  for (const TraceBlockThreadConstraint& constraint : plan.constraints) {
    if (constraint.conditional_instruction_index == instruction_index && constraint.scope_known &&
        (constraint.contract_complete || constraint.hardware_limit > 0)) {
      active_constraints.push_back(&constraint);
    }
  }
  if (active_constraints.empty()) return result;

  result.handled = true;
  const TraceSampleSpec* sample = FindSampleSpec(plan, instruction_index);
  if (sample == nullptr || enumeration_limit <= 0) {
    result.unknown = true;
    result.unknown_reason = TraceConstraintUnknownReason::kUnsupportedInstruction;
    return result;
  }

  auto decision_allowed = [&](const Any& decision, bool* unknown) -> bool {
    for (const TraceBlockThreadConstraint* constraint : active_constraints) {
      ConstraintCheck check = CheckConstraintWithOverlay(plan, *constraint, prefix_trace,
                                                         instruction_index, decision);
      if (check == ConstraintCheck::kUnknown) {
        *unknown = true;
        return false;
      }
      if (check == ConstraintCheck::kRejected) return false;
    }
    return true;
  };

  if (sample->kind == TraceSampleSpec::Kind::kPerfectTile) {
    if (loop_extent <= 0 || sample->n_splits <= 0 ||
        loop_extent > std::numeric_limits<int32_t>::max()) {
      result.unknown = true;
      result.unknown_reason = TraceConstraintUnknownReason::kDynamicExtent;
      return result;
    }
    std::vector<std::vector<int64_t>> decisions;
    std::vector<int64_t> current(sample->n_splits, 1);
    int64_t enumerated = 0;
    bool overflow = false;
    EnumeratePerfectTiles(loop_extent, 0, sample->n_splits, sample->max_innermost_factor,
                          enumeration_limit, &current, &decisions, &enumerated, &overflow);
    if (overflow) {
      result.unknown = true;
      result.unknown_reason = TraceConstraintUnknownReason::kEnumerationLimit;
      return result;
    }

    std::vector<ffi::Array<int64_t>> feasible;
    feasible.reserve(decisions.size());
    for (const std::vector<int64_t>& values : decisions) {
      ffi::Array<int64_t> decision(values.begin(), values.end());
      bool unknown = false;
      if (decision_allowed(decision, &unknown)) feasible.push_back(std::move(decision));
      if (unknown) {
        result.unknown = true;
        result.unknown_reason = TraceConstraintUnknownReason::kMissingDecision;
        return result;
      }
    }
    result.feasible_domain_size = feasible.size();
    if (feasible.empty()) {
      result.empty_domain = true;
      return result;
    }
    int index = s_tir::SampleInt(rand_state, 0, feasible.size());
    result.decision = feasible[index];
    return result;
  }

  if (sample->categorical_candidates.size() != sample->categorical_probabilities.size() ||
      sample->categorical_candidates.empty()) {
    result.unknown = true;
    result.unknown_reason = TraceConstraintUnknownReason::kUnsupportedInstruction;
    return result;
  }
  std::vector<int64_t> feasible_indices;
  std::vector<double> feasible_probabilities;
  for (int64_t index = 0; index < static_cast<int64_t>(sample->categorical_candidates.size());
       ++index) {
    double probability = sample->categorical_probabilities[index];
    if (!(probability > 0.0) || !std::isfinite(probability)) continue;
    bool unknown = false;
    Any decision = index;
    if (decision_allowed(decision, &unknown)) {
      feasible_indices.push_back(index);
      feasible_probabilities.push_back(probability);
    }
    if (unknown) {
      result.unknown = true;
      result.unknown_reason = TraceConstraintUnknownReason::kMissingDecision;
      return result;
    }
  }
  result.feasible_domain_size = feasible_indices.size();
  if (feasible_indices.empty()) {
    result.empty_domain = true;
    return result;
  }
  int sampled = s_tir::MakeMultinomialSampler(rand_state, feasible_probabilities)();
  result.decision = feasible_indices[sampled];
  return result;
}

ThreadFeasibilityResult AnalyzeThreadFeasibility(const IRModule& mod,
                                                 int64_t max_threads_per_block,
                                                 int64_t thread_warp_size,
                                                 int64_t max_shared_memory_per_block) {
  ThreadFeasibilityResult result;
  result.max_threads = 1;
  for (const auto& kv : mod->functions) {
    if (ffi::Optional<PrimFunc> prim_func = kv.second.as<PrimFunc>()) {
      ThreadFeasibilityResult current =
          PhysicalThreadCollector::Collect(prim_func.value(), max_threads_per_block,
                                           thread_warp_size, max_shared_memory_per_block);
      result.reject = result.reject || current.reject;
      result.reject_thread_extent_low =
          result.reject_thread_extent_low || current.reject_thread_extent_low;
      result.reject_thread_extent_high =
          result.reject_thread_extent_high || current.reject_thread_extent_high;
      result.reject_threads_per_block =
          result.reject_threads_per_block || current.reject_threads_per_block;
      if (!current.all_extents_known) {
        result.all_extents_known = false;
      } else if (current.max_threads > result.max_threads) {
        result.max_threads = current.max_threads;
      }
      result.all_contracts_known = result.all_contracts_known && current.all_contracts_known;
      result.all_shared_allocations_known =
          result.all_shared_allocations_known && current.all_shared_allocations_known;
      result.shared_memory_proxy_reject =
          result.shared_memory_proxy_reject || current.shared_memory_proxy_reject;
      result.max_declared_shared_bytes =
          std::max(result.max_declared_shared_bytes, current.max_declared_shared_bytes);
    }
  }
  if (!result.all_extents_known) result.max_threads = -1;
  return result;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "s_tir.meta_schedule.testing.AnalyzeThreadFeasibility",
      [](IRModule mod, int64_t max_threads_per_block, int64_t thread_warp_size,
         int64_t max_shared_memory_per_block) {
        ThreadFeasibilityResult result =
            AnalyzeThreadFeasibility(mod, max_threads_per_block, thread_warp_size,
                                     max_shared_memory_per_block);
        return ffi::Array<int64_t>{static_cast<int64_t>(result.all_extents_known),
                                   static_cast<int64_t>(result.reject), result.max_threads,
                                   static_cast<int64_t>(result.all_contracts_known),
                                   static_cast<int64_t>(result.reject_thread_extent_low),
                                   static_cast<int64_t>(result.reject_thread_extent_high),
                                   static_cast<int64_t>(result.reject_threads_per_block),
                                   static_cast<int64_t>(result.all_shared_allocations_known),
                                   static_cast<int64_t>(result.shared_memory_proxy_reject),
                                   result.max_declared_shared_bytes};
      })
      .def("s_tir.meta_schedule.testing.EvaluateTraceConstraints",
           [](s_tir::Trace trace, int64_t max_threads_per_block, int64_t thread_warp_size) {
             TraceConstraintPlan plan = BuildTraceConstraintPlan(
                 trace, max_threads_per_block, thread_warp_size);
             TraceConstraintEvaluation result = EvaluateTraceConstraints(plan, trace);
             ffi::Map<ffi::String, int64_t> values;
             values.Set("truth", static_cast<int64_t>(result.truth));
             values.Set("reject_low", static_cast<int64_t>(result.reject_thread_extent_low));
             values.Set("reject_high", static_cast<int64_t>(result.reject_thread_extent_high));
             values.Set("reject_hardware",
                        static_cast<int64_t>(result.reject_threads_per_block));
             values.Set("max_threads", result.max_threads);
             values.Set("unknown_reason", static_cast<int64_t>(result.unknown_reason));
             values.Set("contract_blocks", plan.contract_blocks);
             values.Set("constraints", static_cast<int64_t>(plan.constraints.size()));
             values.Set("enforceable_constraints", plan.enforceable_constraints);
             for (int i = 1; i < static_cast<int>(TraceConstraintUnknownReason::kCount); ++i) {
               auto reason = static_cast<TraceConstraintUnknownReason>(i);
               values.Set(TraceConstraintUnknownReasonName(reason), plan.unknown_reason_counts[i]);
             }
             return values;
           });
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

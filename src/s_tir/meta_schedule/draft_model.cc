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

#include "draft_model.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using namespace tvm::tirx;

namespace {

enum class AccessType : int { kUnknown = 0, kRead = 1, kWrite = 2, kReadWrite = 3 };

struct BufferAccess {
  AccessType type{AccessType::kUnknown};
  std::vector<std::vector<PrimExpr>> indices;
};

struct LoopInfo {
  Var var;
  int64_t extent{1};
  bool extent_known{true};
  bool is_physical_gpu_axis{false};
};

class FloatWorkloadCounter : public ExprVisitor {
 public:
#define TVM_DRAFT_VISIT_BINARY(Type)       \
  void VisitExpr_(const Type* op) final {  \
    if (op->dtype.is_float()) ++float_ops; \
    ExprVisitor::VisitExpr_(op);           \
  }

  TVM_DRAFT_VISIT_BINARY(AddNode);
  TVM_DRAFT_VISIT_BINARY(SubNode);
  TVM_DRAFT_VISIT_BINARY(MulNode);
  TVM_DRAFT_VISIT_BINARY(DivNode);
  TVM_DRAFT_VISIT_BINARY(ModNode);
  TVM_DRAFT_VISIT_BINARY(FloorDivNode);
  TVM_DRAFT_VISIT_BINARY(FloorModNode);

#undef TVM_DRAFT_VISIT_BINARY

  int64_t float_ops{0};
};

class BufferAccessCollector : public ExprVisitor {
 public:
  void InsertWrite(const Buffer& buffer, const ffi::Array<PrimExpr>& indices) {
    BufferAccess& access = accesses[buffer.get()];
    access.type = access.type == AccessType::kRead ? AccessType::kReadWrite : AccessType::kWrite;
    access.indices.emplace_back(indices.begin(), indices.end());
  }

  void VisitExpr_(const BufferLoadNode* load) final {
    BufferAccess& access = accesses[load->buffer.get()];
    if (access.type == AccessType::kWrite) {
      access.type = AccessType::kReadWrite;
    } else if (access.type == AccessType::kUnknown) {
      access.type = AccessType::kRead;
    }
    // Pruner assumes that an update reads and writes the same indices, so the
    // write-side indices already represent a read-write access.
    if (access.type != AccessType::kReadWrite) {
      access.indices.emplace_back(load->indices.begin(), load->indices.end());
    }
    ExprVisitor::VisitExpr_(load);
  }

  std::unordered_map<const BufferNode*, BufferAccess> accesses;
};

bool StartsWith(const ffi::String& value, const char* prefix) {
  return std::string(value).rfind(prefix, 0) == 0;
}

int PhysicalAxisIndex(const ffi::String& tag) {
  if (tag == "blockIdx.x") return 0;
  if (tag == "blockIdx.y") return 1;
  if (tag == "blockIdx.z") return 2;
  if (tag == "threadIdx.x") return 3;
  if (tag == "threadIdx.y") return 4;
  if (tag == "threadIdx.z") return 5;
  return -1;
}

bool VarInIndices(const Var& var, const std::vector<std::vector<PrimExpr>>& indices) {
  bool found = false;
  for (const std::vector<PrimExpr>& access : indices) {
    for (const PrimExpr& index : access) {
      PostOrderVisit(index, [&found, &var](const ffi::ObjectRef& node) {
        if (!found) {
          if (const auto* variable = node.as<VarNode>()) found = variable == var.get();
        }
      });
      if (found) return true;
    }
  }
  return false;
}

class PrunerDraftFeatureExtractor : private StmtExprVisitor {
 public:
  static std::vector<PrunerDraftStatementFeatures> Extract(const IRModule& mod, int64_t warp_size) {
    PrunerDraftFeatureExtractor extractor(warp_size);
    for (const auto& kv : mod->functions) {
      if (ffi::Optional<PrimFunc> func = kv.second.as<PrimFunc>()) {
        extractor(func.value()->body);
      }
    }
    std::vector<PrunerDraftStatementFeatures> result;
    result.reserve(extractor.features_.size());
    for (const auto& entry : extractor.features_) result.push_back(entry.second);
    return result;
  }

 private:
  explicit PrunerDraftFeatureExtractor(int64_t warp_size) : warp_size_(warp_size) {}
  // Extract one lexical/thread binding and propagate conservative feature state.
  void VisitBinding(const ffi::String& tag, const Var& var, const PrimExpr& extent,
                    const Stmt& body) {
    const int64_t* static_extent = as_const_int(extent);
    bool extent_known = static_extent != nullptr && *static_extent > 0;
    int64_t extent_value = extent_known ? *static_extent : 1;

    int axis = PhysicalAxisIndex(tag);
    bool is_vthread = StartsWith(tag, "vthread");
    if (axis == -1 && !is_vthread) {
      bool old_known = scope_known_;
      scope_known_ = false;
      VisitStmt(body);
      scope_known_ = old_known;
      return;
    }

    bool old_known = scope_known_;
    double old_inner_product = inner_loop_product_;
    bool old_axis_active = false;
    int64_t old_axis_extent = 1;
    if (axis != -1) {
      old_axis_active = axis_active_[axis];
      old_axis_extent = axis_extents_[axis];
      if (old_axis_active && (!extent_known || old_axis_extent != extent_value)) {
        scope_known_ = false;
      }
      axis_active_[axis] = true;
      axis_extents_[axis] = extent_value;
    } else {
      virtual_thread_size_ *= extent_value;
      inner_loop_product_ *= extent_value;
    }
    if (!extent_known) scope_known_ = false;

    loops_.push_back(LoopInfo{var, extent_value, extent_known, axis != -1});
    VisitStmt(body);
    loops_.pop_back();

    if (axis != -1) {
      axis_active_[axis] = old_axis_active;
      axis_extents_[axis] = old_axis_extent;
    } else {
      virtual_thread_size_ /= extent_value;
    }
    inner_loop_product_ = old_inner_product;
    scope_known_ = old_known;
  }

  void VisitStmt_(const ForNode* loop) final {
    if (loop->kind == ForKind::kThreadBinding) {
      if (!loop->thread_binding.has_value()) {
        bool old_known = scope_known_;
        scope_known_ = false;
        StmtExprVisitor::VisitStmt_(loop);
        scope_known_ = old_known;
        return;
      }
      VisitBinding(loop->thread_binding.value()->thread_tag, loop->loop_var, loop->extent,
                   loop->body);
      return;
    }

    const int64_t* static_extent = as_const_int(loop->extent);
    bool extent_known = static_extent != nullptr && *static_extent > 0;
    int64_t extent_value = extent_known ? *static_extent : 1;
    bool old_known = scope_known_;
    double old_inner_product = inner_loop_product_;
    if (!extent_known) scope_known_ = false;
    inner_loop_product_ *= extent_value;
    loops_.push_back(LoopInfo{loop->loop_var, extent_value, extent_known, false});
    VisitStmt(loop->body);
    loops_.pop_back();
    inner_loop_product_ = old_inner_product;
    scope_known_ = old_known;
  }

  void VisitStmt_(const AttrStmtNode* attr) final {
    if (attr->attr_key != tirx::attr::thread_extent && attr->attr_key != "virtual_thread") {
      StmtExprVisitor::VisitStmt_(attr);
      return;
    }
    const auto* iter_var = attr->node.as<IterVarNode>();
    if (iter_var == nullptr) {
      bool old_known = scope_known_;
      scope_known_ = false;
      VisitStmt(attr->body);
      scope_known_ = old_known;
      return;
    }
    ffi::String tag =
        attr->attr_key == "virtual_thread" ? ffi::String("vthread") : iter_var->thread_tag;
    if (tag.empty()) tag = iter_var->var->name_hint;
    VisitBinding(tag, iter_var->var, attr->value, attr->body);
  }

  void VisitStmt_(const SBlockNode* block) final {
    bool warp_execution = block->annotations.count(s_tir::attr::warp_execution);
    if (!warp_execution) {
      StmtExprVisitor::VisitStmt_(block);
      return;
    }

    bool old_known = scope_known_;
    bool old_active = axis_active_[3];
    int64_t old_extent = axis_extents_[3];
    if (warp_size_ <= 0 || (old_active && old_extent != warp_size_)) {
      scope_known_ = false;
    } else if (!old_active) {
      axis_active_[3] = true;
      axis_extents_[3] = warp_size_;
    }
    StmtExprVisitor::VisitStmt_(block);
    axis_active_[3] = old_active;
    axis_extents_[3] = old_extent;
    scope_known_ = old_known;
  }

  std::pair<double, bool> MemoryWorkload(const std::vector<std::vector<PrimExpr>>& indices,
                                         const std::vector<const LoopInfo*>& selected_loops) const {
    double workload = 1.0;
    for (const LoopInfo* loop : selected_loops) {
      if (!VarInIndices(loop->var, indices)) continue;
      if (!loop->extent_known) return {1.0, false};
      workload *= loop->extent;
    }
    return {workload, std::isfinite(workload)};
  }
  // Extract one feature record for each buffer store.
  void VisitStmt_(const BufferStoreNode* store) final {
    FloatWorkloadCounter counter;
    counter(store->value);

    double grid_size = 1.0;
    double block_size = 1.0;
    for (int i = 0; i < 3; ++i) {
      if (axis_active_[i]) grid_size *= axis_extents_[i];
      if (axis_active_[i + 3]) block_size *= axis_extents_[i + 3];
    }

    PrunerDraftStatementFeatures feature;
    feature.known = scope_known_ && std::isfinite(inner_loop_product_) &&
                    std::isfinite(grid_size) && std::isfinite(block_size);
    feature.is_gpu =
        std::any_of(axis_active_.begin(), axis_active_.end(), [](bool value) { return value; });
    feature.grid_size = grid_size;
    feature.block_size = block_size;
    feature.virtual_thread_size = virtual_thread_size_;
    feature.float_workload_per_thread = inner_loop_product_ * counter.float_ops;
    feature.compute_workload =
        feature.float_workload_per_thread * feature.grid_size * feature.block_size;

    BufferAccessCollector accesses;
    accesses.InsertWrite(store->buffer, store->indices);
    accesses(store->value);

    std::vector<const LoopInfo*> all_nonphysical;
    std::vector<const LoopInfo*> store_axes;
    all_nonphysical.reserve(loops_.size());
    store_axes.reserve(loops_.size());
    for (const LoopInfo& loop : loops_) {
      if (loop.is_physical_gpu_axis) continue;
      all_nonphysical.push_back(&loop);
      if (VarInIndices(loop.var, accesses.accesses.at(store->buffer.get()).indices)) {
        store_axes.push_back(&loop);
      }
    }

    for (const auto& entry : accesses.accesses) {
      const BufferAccess& access = entry.second;
      if (feature.float_workload_per_thread != 0.0) {
        auto [register_workload, register_known] = MemoryWorkload(access.indices, store_axes);
        auto [memory_workload, memory_known] = MemoryWorkload(access.indices, all_nonphysical);
        feature.register_usage += register_workload;
        feature.memory_workload += memory_workload;
        feature.known = feature.known && register_known && memory_known;
      } else if (StartsWith(store->buffer.scope(), "shared")) {
        auto [shared_workload, shared_known] = MemoryWorkload(access.indices, store_axes);
        feature.shared_memory_footprint += shared_workload * feature.block_size;
        feature.shared_memory_bytes_proxy += shared_workload * feature.block_size *
                                             store->buffer->dtype.bytes();
        feature.dram_workload += feature.shared_memory_footprint * feature.grid_size;
        feature.known = feature.known && shared_known;
        break;
      } else {
        feature.dram_workload += inner_loop_product_ * feature.grid_size * feature.block_size;
        break;
      }
    }
    if (feature.float_workload_per_thread != 0.0 && feature.memory_workload != 0.0) {
      feature.data_reuse_penalty += feature.float_workload_per_thread / feature.memory_workload;
    }
    feature.known = feature.known && std::isfinite(feature.compute_workload) &&
                    std::isfinite(feature.register_usage) && std::isfinite(feature.dram_workload) &&
                    std::isfinite(feature.data_reuse_penalty);
    features_[store->buffer.get()] = feature;
  }

  int64_t warp_size_;
  std::vector<LoopInfo> loops_;
  std::array<int64_t, 6> axis_extents_{{1, 1, 1, 1, 1, 1}};
  std::array<bool, 6> axis_active_{{false, false, false, false, false, false}};
  double inner_loop_product_{1.0};
  double virtual_thread_size_{1.0};
  bool scope_known_{true};
  std::unordered_map<const BufferNode*, PrunerDraftStatementFeatures> features_;
};

}  // namespace

bool PrunerDraftHardwareProfile::IsValid() const {
  return peak_gflops > 0.0 && global_memory_gbps > 0.0 && sm_count > 0 &&
         active_blocks_per_sm > 0 && sm_partitions > 0 && warp_size > 0 && register_capacity > 0.0;
}

PrunerDraftScore ScorePrunerDraftFeatures(const std::vector<PrunerDraftStatementFeatures>& features,
                                          const PrunerDraftHardwareProfile& profile) {
  if (!profile.IsValid() || features.empty()) return PrunerDraftScore{};

  double total_latency = 0.0;
  int64_t gpu_statements = 0;
  for (const PrunerDraftStatementFeatures& feature : features) {
    if (feature.is_gpu == 0.0) continue;
    ++gpu_statements;
    if (!feature.known || feature.grid_size <= 0.0 || feature.block_size <= 0.0 ||
        feature.data_reuse_penalty <= 0.0 || feature.compute_workload < 0.0 ||
        feature.dram_workload < 0.0) {
      return PrunerDraftScore{/*known=*/false, /*score=*/1.0, gpu_statements};
    }

    double raw_warps = feature.block_size / profile.warp_size;
    double scheduled_warps = std::ceil(raw_warps);
    double warp_penalty = raw_warps / scheduled_warps;

    double scheduling_units = feature.grid_size;
    double scheduling_capacity =
        static_cast<double>(profile.sm_count * profile.active_blocks_per_sm);
    if (feature.grid_size < scheduling_capacity) {
      double active_warps =
          scheduled_warps < profile.sm_partitions ? scheduled_warps : profile.sm_partitions;
      scheduling_units *= active_warps;
      scheduling_capacity = static_cast<double>(profile.sm_partitions * profile.sm_count);
    }
    double waves = std::ceil(scheduling_units / scheduling_capacity);
    double peak_penalty = scheduling_units / (waves * scheduling_capacity);
    double register_penalty = std::max(feature.register_usage / profile.register_capacity, 1.0);
    double thread_reuse_penalty = 1.0 + 1.0 / feature.data_reuse_penalty;

    double compute_throughput =
        profile.peak_gflops * peak_penalty * warp_penalty / register_penalty;
    double compute_time =
        feature.compute_workload * thread_reuse_penalty / compute_throughput / 1.0e6;
    double memory_time =
        feature.dram_workload / profile.global_memory_gbps * 1000.0 / (1024.0 * 1024.0 * 1024.0);
    total_latency += std::max(compute_time, memory_time);
  }

  if (gpu_statements == 0 || !std::isfinite(total_latency) || total_latency <= 0.0) {
    return PrunerDraftScore{/*known=*/false, /*score=*/1.0, gpu_statements};
  }
  return PrunerDraftScore{/*known=*/true, /*score=*/1.0 / total_latency, gpu_statements};
}
//算分
PrunerDraftScore AnalyzePrunerDraftScore(const IRModule& mod,
                                         const PrunerDraftHardwareProfile& profile) {
  if (!profile.IsValid()) return PrunerDraftScore{};
  return ScorePrunerDraftFeatures(PrunerDraftFeatureExtractor::Extract(mod, profile.warp_size),
                                  profile);
}

std::vector<double> NormalizePrunerDraftScores(const std::vector<PrunerDraftScore>& scores) {
  double best = 0.0;
  for (const PrunerDraftScore& score : scores) {
    if (score.known && std::isfinite(score.score) && score.score > best) best = score.score;
  }

  std::vector<double> normalized(scores.size(), 1.0);
  if (best <= 0.0) return normalized;
  for (int i = 0, n = scores.size(); i < n; ++i) {
    if (scores[i].known && std::isfinite(scores[i].score) && scores[i].score > 0.0) {
      normalized[i] = std::max(0.0, std::min(1.0, scores[i].score / best));
    }
  }
  return normalized;
}
// Normalize draft scores and combine them with learned scores.
std::vector<double> ApplyPrunerDraftScores(const std::vector<double>& learned_scores,
                                           const std::vector<double>& normalized_draft_scores,
                                           int mode, double beta) {
  TVM_FFI_ICHECK_EQ(learned_scores.size(), normalized_draft_scores.size());
  std::vector<double> combined = learned_scores;
  if (mode != 2 || beta == 0.0) return combined;
  constexpr double kLogFloor = 1.0e-12;
  for (int i = 0, n = combined.size(); i < n; ++i) {
    double draft = std::max(kLogFloor, std::min(1.0, normalized_draft_scores[i]));
    combined[i] *= std::pow(draft, beta);
  }
  return combined;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.meta_schedule.testing.ScoutDraftFeatures", [](IRModule mod) {
        // Explicit profile for this experimental FP32/sm_86 study.
        PrunerDraftHardwareProfile profile{10752.0, 192.0, 20, 1, 4, 32, 128.0};
        auto rows = PrunerDraftFeatureExtractor::Extract(mod, 32);
        auto score = ScorePrunerDraftFeatures(rows, profile);
        double threads = 0, grid = 0, registers = 0, shared = 0, reuse = 0, work = 0;
        bool known = score.known;
        for (const auto& row : rows) {
          if (!row.is_gpu) continue;
          known = known && row.known;
          threads = std::max(threads, row.block_size);
          grid = std::max(grid, row.grid_size);
          registers = std::max(registers, row.register_usage);
          shared = std::max(shared, row.shared_memory_bytes_proxy);
          double weight = std::max(0.0, row.compute_workload);
          reuse += weight * row.data_reuse_penalty;
          work += weight;
        }
        // 1/score is the legacy nominal millisecond proxy, not calibrated latency.
        // The existing per-buffer overwrite aggregation is intentionally retained.
        return ffi::Array<double>{static_cast<double>(known), score.score,
                                  std::log1p(threads), std::log1p(grid / 20.0),
                                  std::log1p(registers / 128.0),
                                  std::log1p(shared / 49152.0),
                                  std::log1p(work > 0 ? reuse / work : 1.0)};
      })
      .def("s_tir.meta_schedule.testing.ScorePrunerDraftStatement",
           [](double compute_workload, double register_usage, double dram_workload,
              double data_reuse_penalty, double grid_size, double block_size, double peak_gflops,
              double global_memory_gbps, int64_t sm_count, int64_t active_blocks_per_sm,
              int64_t sm_partitions, int64_t warp_size, double register_capacity) {
             PrunerDraftStatementFeatures feature;
             feature.compute_workload = compute_workload;
             feature.register_usage = register_usage;
             feature.dram_workload = dram_workload;
             feature.data_reuse_penalty = data_reuse_penalty;
             feature.grid_size = grid_size;
             feature.block_size = block_size;
             feature.is_gpu = 1.0;
             PrunerDraftHardwareProfile profile{peak_gflops,          global_memory_gbps, sm_count,
                                                active_blocks_per_sm, sm_partitions,      warp_size,
                                                register_capacity};
             return ScorePrunerDraftFeatures({feature}, profile).score;
           })
      .def("s_tir.meta_schedule.testing.AnalyzePrunerDraftScore",
           [](IRModule mod, double peak_gflops, double global_memory_gbps, int64_t sm_count,
              int64_t active_blocks_per_sm, int64_t sm_partitions, int64_t warp_size,
              double register_capacity) {
             PrunerDraftHardwareProfile profile{peak_gflops,          global_memory_gbps, sm_count,
                                                active_blocks_per_sm, sm_partitions,      warp_size,
                                                register_capacity};
             PrunerDraftScore score = AnalyzePrunerDraftScore(mod, profile);
             return ffi::Array<double>{static_cast<double>(score.known), score.score,
                                       static_cast<double>(score.statement_count)};
           })
      .def("s_tir.meta_schedule.testing.NormalizePrunerDraftScores",
           [](ffi::Array<double> raw_scores, ffi::Array<int64_t> known) {
             TVM_FFI_ICHECK_EQ(raw_scores.size(), known.size());
             std::vector<PrunerDraftScore> scores;
             scores.reserve(raw_scores.size());
             for (int i = 0, n = raw_scores.size(); i < n; ++i) {
               scores.push_back(PrunerDraftScore{known[i] != 0, raw_scores[i], 1});
             }
             std::vector<double> normalized = NormalizePrunerDraftScores(scores);
             return ffi::Array<double>(normalized.begin(), normalized.end());
           })
      .def("s_tir.meta_schedule.testing.ApplyPrunerDraftScores",
           [](ffi::Array<double> learned_scores, ffi::Array<double> draft_scores, int mode,
              double beta) {
             std::vector<double> learned(learned_scores.begin(), learned_scores.end());
             std::vector<double> draft(draft_scores.begin(), draft_scores.end());
             std::vector<double> combined = ApplyPrunerDraftScores(learned, draft, mode, beta);
             return ffi::Array<double>(combined.begin(), combined.end());
           });
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

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
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/registry.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include "../draft_guided_ga.h"
#include "../draft_model.h"
#include "../draft_mutator_credit.h"
#include "../module_equality.h"
#include "../utils.h"

#define TVM_META_SCHEDULE_CHECK_PROB_RANGE(p, name)   \
  TVM_FFI_CHECK(0.0 <= (p) && (p) <= 1.0, ValueError) \
      << "name should be within [0, 1], "             \
      << "but get `" << #p << " = " << (p) << '\'';

namespace tvm {
namespace s_tir {
namespace meta_schedule {

using s_tir::Schedule;

/**************** Data Structure ****************/

/*! \brief The direct mutation edge that produced a candidate. */
struct MutationLineage {
  IRModule parent_mod;
  int mutator_index;
  std::optional<double> measured_parent_latency_sec;
  int generation;
};

/*! \brief A strict structural map containing only latencies observed by RunnerResult. */
class MeasuredLatencyTable {
 public:
  std::optional<double> Get(const IRModule& mod) const {
    auto it = table_.find(mod);
    return it == table_.end() ? std::nullopt : std::optional<double>(it->second);
  }

  void Set(const IRModule& mod, double latency_sec) { table_.insert_or_assign(mod, latency_sec); }

  size_t Size() const { return table_.size(); }

 private:
  struct Hash {
    size_t operator()(const IRModule& mod) const { return ffi::StructuralHash()(mod); }
  };
  struct Equal {
    bool operator()(const IRModule& lhs, const IRModule& rhs) const {
      return ffi::StructuralEqual()(lhs, rhs);
    }
  };
  std::unordered_map<IRModule, double, Hash, Equal> table_;
};

/*! \brief An auxiliary data structure to help deduplicate IRModules */
class IRModuleSet {
 public:
  explicit IRModuleSet(const ModuleEquality& mod_eq)
      : tab_(/*bucket_count*/ 0, ItemHash(), ItemEqual(mod_eq)) {}

  /*! \brief Add an IRModule to the set */
  void Add(const IRModule& mod, size_t shash) { tab_.insert(Item{mod, shash}); }
  /*! \brief Check if the IRModule is in the set */
  bool Has(const IRModule& mod, size_t shash) const { return tab_.count(Item{mod, shash}); }

 private:
  struct Item {
    IRModule mod;
    size_t shash;
  };
  struct ItemHash {
    size_t operator()(const Item& hash) const { return hash.shash; }
  };
  struct ItemEqual {
    explicit ItemEqual(const ModuleEquality& mod_eq) : mod_eq_(mod_eq) {}
    ItemEqual& operator=(const ItemEqual& other) { return *this; }

    bool operator()(const Item& lhs, const Item& rhs) const {
      return lhs.shash == rhs.shash && mod_eq_.Equal(lhs.mod, rhs.mod);
    }

    const ModuleEquality& mod_eq_;
  };

  std::unordered_set<Item, ItemHash, ItemEqual> tab_;
};

/*!
 * \brief A heap with a size up-limit. If overflow happens, it evicted the worst items.
 * \note It maintains the worst item at the top, using priority class, score,
 * and a deterministic tie-breaker in that order.
 */
class SizedHeap {
 public:
  struct Item {
    Schedule sch;
    double score;
    std::optional<MutationLineage> lineage;
    int priority_class;
    uint64_t tie_breaker;

    bool operator<(const Item& other) const {
      if (priority_class != other.priority_class) {
        return priority_class > other.priority_class;
      }
      if (score != other.score) {
        return score > other.score;
      }
      return tie_breaker > other.tie_breaker;
    }
  };

  /*!
   * \brief Constructor
   * \param size_limit The up-limit of the heap size
   */
  explicit SizedHeap(int size_limit) : size_limit(size_limit) { heap.reserve(size_limit); }

  /*!
   * \brief Push the specific item to the heap if its key did not appears in the heap
   * \param item The item to be pushed
   */
  void Push(Schedule sch, double score, std::optional<MutationLineage> lineage = std::nullopt,
            int priority_class = 0, uint64_t tie_breaker = 0) {
    Item item{sch, score, std::move(lineage), priority_class, tie_breaker};
    int size = heap.size();
    if (size < size_limit) {
      // Heap is not full, just push
      heap.push_back(std::move(item));
      std::push_heap(heap.begin(), heap.end());
    } else if (item < heap.front()) {
      // if the item is better than the worst one in the heap, we can safely kick it out
      std::pop_heap(heap.begin(), heap.end());
      heap.back() = std::move(item);
      std::push_heap(heap.begin(), heap.end());
    }
    // Otherwise, the item is worse than any other element in the heap
  }

  /*! \brief Up-limit of the heap size */
  int size_limit;
  /*! \brief The heap, the worse the topper */
  std::vector<Item> heap;
};

struct PerThreadData {
  IRModule mod{ffi::UnsafeInit()};
  TRandState rand_state{-1};
  std::function<int32_t()> trace_sampler = nullptr;
  std::function<ffi::Optional<Mutator>()> mutator_sampler = nullptr;

  /*!
   * \brief Set the value for the trace and mutator samplers per thread.
   * \param scores The predicted score for the given samples.
   * \param genetic_mutate_prob The probability of mutation.
   * \param mutator_probs The probability of each mutator as a dict.
   */
  void Set(const std::vector<double>& scores, double genetic_mutate_prob,
           const ffi::Map<Mutator, FloatImm>& mutator_probs) {
    trace_sampler = s_tir::MakeMultinomialSampler(&rand_state, scores);
    mutator_sampler = MakeMutatorSampler(genetic_mutate_prob, mutator_probs, &rand_state);
  }

 private:
  /*!
   * \brief Create a sampler function that picks mutators according to the mass function
   * \param rand_state The random state for sampling
   * \return The sampler created
   */
  static std::function<ffi::Optional<Mutator>()> MakeMutatorSampler(
      double genetic_mutate_prob,                        //
      const ffi::Map<Mutator, FloatImm>& mutator_probs,  //
      TRandState* rand_state) {
    std::vector<ffi::Optional<Mutator>> mutators;
    std::vector<double> masses;
    mutators.push_back(std::nullopt);
    masses.push_back(1.0 - genetic_mutate_prob);
    double total_mass_mutator = 0.0;
    if (genetic_mutate_prob > 0) {
      for (const auto& kv : mutator_probs) {
        Mutator mutator = kv.first;
        double mass = kv.second->value;
        total_mass_mutator += mass;
        mutators.push_back(mutator);
        masses.push_back(mass * genetic_mutate_prob);
      }
    }
    // Normalize the sum to 1.0
    if (total_mass_mutator == 0.0) {
      masses[0] = 1.0;
      for (int i = 1, n = masses.size(); i < n; ++i) {
        masses[i] = 0.0;
      }
    } else if (total_mass_mutator != 1.0) {
      for (int i = 1, n = masses.size(); i < n; ++i) {
        masses[i] /= total_mass_mutator;
      }
    }
    return [idx_sampler = s_tir::MakeMultinomialSampler(rand_state, masses),
            mutators = std::move(mutators)]() -> ffi::Optional<Mutator> {
      int i = idx_sampler();
      return mutators[i];
    };
  }
};

struct ConcurrentBitmask {
  /*! The bit width. */
  static constexpr const int kBitWidth = 64;
  /*! \brief The size of the concurrent bitmask. */
  int size;
  /*! \brief The bitmasks. */
  std::vector<uint64_t> bitmask;
  /*! \brief The mutexes, one per kBitWidth(64 here) bitmasks. */
  std::vector<std::mutex> mutexes;

  /*!
   * \brief Constructor
   * \param n The total slots managed by the concurrent bitmask.
   */
  explicit ConcurrentBitmask(int n)
      : size((n + kBitWidth - 1) / kBitWidth), bitmask(size, 0), mutexes(size) {}
  /*!
   * \brief Query and mark the given index if not visited before.
   * \param x The index to concurrently check if used. If not, mark as used.
   * \return Whether the index has been used before.
   */
  bool QueryAndMark(int x) {
    constexpr uint64_t one = 1;
    std::unique_lock<std::mutex> lock(mutexes[x / kBitWidth]);
    if (bitmask[x / kBitWidth] & (one << (x % kBitWidth))) {
      return false;
    } else {
      bitmask[x / kBitWidth] |= one << (x % kBitWidth);
      return true;
    }
  }
};

struct DraftAnnealStats {
  std::atomic<int64_t> compared{0};
  std::atomic<int64_t> improved_or_equal{0};
  std::atomic<int64_t> worse{0};
  std::atomic<int64_t> accepted_worse{0};
  std::atomic<int64_t> rejected_worse{0};
  std::atomic<int64_t> unknown{0};
  std::atomic<int64_t> draft_time_us{0};
};

struct DraftG3FeedbackStats {
  int child_measured{0};
  int with_lineage{0};
  int with_measured_parent{0};
  int credited{0};
  int improved{0};
  int skipped_child_error{0};
  int skipped_no_lineage{0};
  int skipped_parent_unmeasured{0};
};

/**************** Util Functions ****************/

namespace {

constexpr double kDefaultSoftFilterWeight = 0.2;
constexpr double kDefaultDraftBeta = 0.2;
constexpr double kDefaultDraftEpsilon = 0.10;
constexpr double kDefaultDraftGAVerifyRatio = 0.25;

// Thread-only profile for the sm_86 development GPU.  These are deliberately
// kept out of the hard filter: they describe performance, not legality.
constexpr int64_t kSm86WarpSchedulersPerSm = 4;
constexpr int64_t kSm86ThreadsPerSm = 1536;
constexpr int64_t kSm86MaxResidentBlocksPerSm = 16;

int ReadSoftFilterMode() {
  const char* env = std::getenv("TVM_SOFT_FILTER_MODE");
  if (env == nullptr) return 0;
  int mode = std::atoi(env);
  return 0 <= mode && mode <= 2 ? mode : 0;
}

double ReadSoftFilterWeight() {
  const char* env = std::getenv("TVM_SOFT_FILTER_WEIGHT");
  if (env == nullptr) return kDefaultSoftFilterWeight;
  char* end = nullptr;
  double weight = std::strtod(env, &end);
  if (end == env || *end != '\0' || !std::isfinite(weight)) {
    return kDefaultSoftFilterWeight;
  }
  return std::min(1.0, std::max(0.0, weight));
}

int ReadDraftMode() {
  const char* env = std::getenv("TVM_DRAFT_MODE");
  if (env == nullptr) return 0;
  int mode = std::atoi(env);
  return 0 <= mode && mode <= 2 ? mode : 0;
}

int ReadDraftGAMode() {
  const char* env = std::getenv("TVM_DRAFT_GA_MODE");
  if (env == nullptr) return 0;
  int mode = std::atoi(env);
  return 0 <= mode && mode <= 2 ? mode : 0;
}

double ReadDraftGAVerifyRatio() {
  const char* env = std::getenv("TVM_DRAFT_GA_VERIFY_RATIO");
  if (env == nullptr) return kDefaultDraftGAVerifyRatio;
  char* end = nullptr;
  double ratio = std::strtod(env, &end);
  return end != env && *end == '\0' && std::isfinite(ratio) && ratio > 0.0 && ratio <= 1.0
             ? ratio
             : kDefaultDraftGAVerifyRatio;
}

double ReadDraftGAFinalEpsilon() {
  // Preserve historical experiments unless an explicit exploration quota is set.
  const char* env = std::getenv("TVM_DRAFT_GA_FINAL_EPS");
  if (env == nullptr) return 0.0;
  char* end = nullptr;
  double epsilon = std::strtod(env, &end);
  TVM_FFI_CHECK(end != env && *end == '\0' && std::isfinite(epsilon) && epsilon >= 0.0 &&
                    epsilon <= 1.0,
                ValueError)
      << "TVM_DRAFT_GA_FINAL_EPS must be a finite probability";
  return epsilon;
}

double ReadDraftBeta() {
  const char* env = std::getenv("TVM_DRAFT_BETA");
  if (env == nullptr) return kDefaultDraftBeta;
  char* end = nullptr;
  double beta = std::strtod(env, &end);
  return end != env && *end == '\0' && std::isfinite(beta) && beta >= 0.0 ? beta
                                                                          : kDefaultDraftBeta;
}

double ReadDraftEpsilon() {
  const char* env = std::getenv("TVM_DRAFT_EPS");
  if (env == nullptr) return kDefaultDraftEpsilon;
  char* end = nullptr;
  double epsilon = std::strtod(env, &end);
  return end != env && *end == '\0' && std::isfinite(epsilon) && epsilon >= 0.0 && epsilon <= 1.0
             ? epsilon
             : kDefaultDraftEpsilon;
}

std::optional<double> ReadPositiveDouble(const char* name) {
  const char* env = std::getenv(name);
  if (env == nullptr) return std::nullopt;
  char* end = nullptr;
  double value = std::strtod(env, &end);
  if (end == env || *end != '\0' || !std::isfinite(value) || value <= 0.0) {
    return std::nullopt;
  }
  return value;
}

std::optional<int64_t> ReadPositiveInt(const char* name) {
  std::optional<double> value = ReadPositiveDouble(name);
  if (!value.has_value() || std::floor(value.value()) != value.value()) return std::nullopt;
  return static_cast<int64_t>(value.value());
}

int ReadDraftGAAnnealMode() {
  const char* env = std::getenv("TVM_DRAFT_GA_ANNEAL_MODE");
  if (env == nullptr) return 0;
  int mode = std::atoi(env);
  return 0 <= mode && mode <= 2 ? mode : 0;
}

int ReadDraftGAG3Mode() {
  const char* env = std::getenv("TVM_DRAFT_GA_G3_MODE");
  if (env == nullptr) return 0;
  std::string value(env);
  if (value == "off") return 0;
  if (value == "shadow") return 1;
  if (value == "enforce") return 2;
  int mode = std::atoi(env);
  return 0 <= mode && mode <= 2 ? mode : 0;
}

const char* DraftG3ModeName(int mode) {
  return mode == 1 ? "shadow" : (mode == 2 ? "enforce" : "off");
}

double ReadDraftGAAnnealT0() {
  const char* env = std::getenv("TVM_DRAFT_GA_ANNEAL_T0");
  if (env == nullptr) return 0.10;
  char* end = nullptr;
  double value = std::strtod(env, &end);
  return end != env && *end == '\0' && std::isfinite(value) && value >= 0.0 ? value : 0.10;
}

double ReadDraftGAAnnealDecay() {
  const char* env = std::getenv("TVM_DRAFT_GA_ANNEAL_DECAY");
  if (env == nullptr) return 0.95;
  char* end = nullptr;
  double value = std::strtod(env, &end);
  return end != env && *end == '\0' && std::isfinite(value) && value > 0.0 && value <= 1.0
             ? value
             : 0.95;
}

bool ReadExactInitPopulation() {
  const char* env = std::getenv("TVM_EVO_EXACT_INIT_POPULATION");
  return env != nullptr && std::atoi(env) != 0;
}

int ReadExactInitBatchSize(int population_size) {
  std::optional<int64_t> value = ReadPositiveInt("TVM_EVO_INIT_SAMPLE_BATCH_SIZE");
  if (!value.has_value()) return population_size;
  return static_cast<int>(std::min<int64_t>(value.value(), population_size));
}

std::optional<double> MedianValidRunSecs(const RunnerResult& result) {
  if (result->error_msg.has_value() || !result->run_secs.has_value() ||
      result->run_secs.value().empty()) {
    return std::nullopt;
  }
  std::vector<double> values;
  values.reserve(result->run_secs.value().size());
  for (const FloatImm& value : result->run_secs.value()) {
    if (!std::isfinite(value->value) || value->value <= 0.0 ||
        value->value == SortTuningRecordByMeanRunSecs::kMaxMeanTime) {
      return std::nullopt;
    }
    values.push_back(value->value);
  }
  std::sort(values.begin(), values.end());
  int n = static_cast<int>(values.size());
  return n % 2 == 0 ? (values[n / 2 - 1] + values[n / 2]) * 0.5 : values[n / 2];
}

std::optional<PrunerDraftHardwareProfile> ReadDraftHardwareProfile(const TuneContext& context) {
  if (!context->target.has_value()) return std::nullopt;
  std::optional<double> peak_gflops = ReadPositiveDouble("TVM_DRAFT_PEAK_GFLOPS");
  std::optional<double> memory_gbps = ReadPositiveDouble("TVM_DRAFT_MEMORY_GBPS");
  std::optional<int64_t> sm_count = ReadPositiveInt("TVM_DRAFT_SM_COUNT");
  if (!peak_gflops.has_value() || !memory_gbps.has_value() || !sm_count.has_value()) {
    return std::nullopt;
  }

  const Target& target = context->target.value();
  PrunerDraftHardwareProfile profile;
  profile.peak_gflops = peak_gflops.value();
  profile.global_memory_gbps = memory_gbps.value();
  profile.sm_count = sm_count.value();
  profile.warp_size = target->GetAttr<int64_t>("thread_warp_size").value_or(-1);
  profile.active_blocks_per_sm = ReadPositiveInt("TVM_DRAFT_ACTIVE_BLOCKS_PER_SM").value_or(1);
  profile.sm_partitions = ReadPositiveInt("TVM_DRAFT_SM_PARTITIONS").value_or(4);
  profile.register_capacity = ReadPositiveDouble("TVM_DRAFT_REG_CAP").value_or(128.0);
  return profile.IsValid() ? std::optional<PrunerDraftHardwareProfile>(profile) : std::nullopt;
}

bool IsSm86Target(const Target& target) {
  if (ffi::Optional<ffi::String> arch = target->GetAttr<ffi::String>("arch")) {
    return arch.value() == "sm_86";
  }
  if (ffi::Optional<ffi::String> mcpu = target->GetAttr<ffi::String>("mcpu")) {
    return mcpu.value() == "sm_86";
  }
  return false;
}

int64_t CeilDiv(int64_t value, int64_t divisor) {
  return value / divisor + static_cast<int64_t>(value % divisor != 0);
}

/*! \brief Return a neutral 1.0 for unknown inputs and a performance penalty otherwise. */
double ThreadSoftPenalty(int64_t threads, int64_t warp_size) {
  if (threads <= 0 || warp_size <= 0) return 1.0;

  int64_t scheduled_warps = CeilDiv(threads, warp_size);
  double lane_utilization =
      static_cast<double>(threads) / static_cast<double>(scheduled_warps * warp_size);
  double warp_scheduler_utilization =
      static_cast<double>(scheduled_warps) /
      static_cast<double>(CeilDiv(scheduled_warps, kSm86WarpSchedulersPerSm) *
                          kSm86WarpSchedulersPerSm);

  int64_t resident_blocks = std::min<int64_t>(kSm86MaxResidentBlocksPerSm,
                                              kSm86ThreadsPerSm / std::max<int64_t>(threads, 1));
  double thread_occupancy =
      std::min(1.0, static_cast<double>(resident_blocks * threads) / kSm86ThreadsPerSm);

  double product = lane_utilization * warp_scheduler_utilization * thread_occupancy;
  return std::cbrt(std::max(0.0, std::min(1.0, product)));
}

std::vector<double> ApplySoftPenalties(const std::vector<double>& raw_scores,
                                       const std::vector<double>& penalties, int mode,
                                       double weight) {
  TVM_FFI_ICHECK_EQ(raw_scores.size(), penalties.size());
  std::vector<double> adjusted = raw_scores;
  if (mode != 2 || weight == 0.0) return adjusted;
  for (int i = 0, n = adjusted.size(); i < n; ++i) {
    adjusted[i] *= std::pow(std::max(0.0, std::min(1.0, penalties[i])), weight);
  }
  return adjusted;
}

std::vector<int> TopKIndices(const std::vector<double>& scores, int k) {
  std::vector<int> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  auto greater_score = [&scores](int lhs, int rhs) {
    if (scores[lhs] != scores[rhs]) return scores[lhs] > scores[rhs];
    return lhs < rhs;
  };
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), greater_score);
  indices.resize(k);
  return indices;
}

int TopKOverlap(const std::vector<double>& lhs, const std::vector<double>& rhs, int k) {
  if (k == 0) return 0;
  std::vector<int> lhs_top = TopKIndices(lhs, k);
  std::vector<int> rhs_top = TopKIndices(rhs, k);
  std::sort(lhs_top.begin(), lhs_top.end());
  std::sort(rhs_top.begin(), rhs_top.end());
  int overlap = 0;
  for (int i = 0, j = 0; i < k && j < k;) {
    if (lhs_top[i] == rhs_top[j]) {
      ++overlap;
      ++i;
      ++j;
    } else if (lhs_top[i] < rhs_top[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return overlap;
}

}  // namespace

/*!
 * \brief Assemble measure candidates from the given candidate traces.
 * \param traces The picked candidate traces.
 * \return The assembled measure candidates.
 */
ffi::Array<MeasureCandidate> AssembleCandidates(const std::vector<Schedule>& picks) {
  ffi::Array<MeasureCandidate> measure_inputs;
  measure_inputs.reserve(picks.size());
  for (const Schedule& sch : picks) {
    measure_inputs.push_back(
        MeasureCandidate(sch, ArgInfo::FromEntryFunc(sch->mod(), /*remove_preproc=*/true)));
  }
  return measure_inputs;
}

/*!
 * \brief Predict the normalized score of each candidate.
 * \param candidates The candidates for prediction
 * \param task The search task
 * \param space The search space
 * \return The normalized score in the prediction
 */
std::vector<double> PredictCostModelNormalizedScore(const std::vector<Schedule>& candidates,
                                                    const TuneContext& context,
                                                    const CostModel& cost_model) {
  auto _ = Profiler::TimedScope("EvoSearch/Evolve/PredictNormalizedScore");
  TVM_FFI_ICHECK(!candidates.empty())
      << "Candidates given for score prediction can not be empty list!";
  std::vector<double> scores = cost_model->Predict(context, AssembleCandidates(candidates));
  for (double& score : scores) {
    score = std::max(0.0, score);
  }
  return scores;
}

std::vector<double> PredictNormalizedScore(const std::vector<Schedule>& candidates,
                                           const TuneContext& context,
                                           const CostModel& cost_model) {
  std::vector<double> scores =
      PredictCostModelNormalizedScore(candidates, context, cost_model);

  TVM_FFI_ICHECK_EQ(scores.size(), candidates.size());
  int n = candidates.size();

  int soft_filter_mode = ReadSoftFilterMode();
  if (soft_filter_mode != 0) {
    const auto soft_filter_start = std::chrono::steady_clock::now();
    std::vector<double> penalties(n, 1.0);
    std::vector<int> known(n, 0);

    int64_t max_threads_per_block = -1;
    int64_t warp_size = -1;
    bool has_sm86_profile = false;
    if (context->target.has_value()) {
      const Target& target = context->target.value();
      max_threads_per_block =
          target->GetAttr<int64_t>("max_threads_per_block")
              .value_or(target->GetAttr<int64_t>("max_num_threads").value_or(-1));
      warp_size = target->GetAttr<int64_t>("thread_warp_size").value_or(-1);
      has_sm86_profile = IsSm86Target(target);
    }

    if (has_sm86_profile && max_threads_per_block > 0 && warp_size > 0) {
      auto analyze = [&candidates, &penalties, &known, max_threads_per_block, warp_size](
                         int, int task_id) {
        ThreadFeasibilityResult result =
            AnalyzeThreadFeasibility(candidates[task_id]->mod(), max_threads_per_block, warp_size,
                                     /*max_shared_memory_per_block=*/-1);
        if (result.all_extents_known && result.max_threads > 0) {
          penalties[task_id] = ThreadSoftPenalty(result.max_threads, warp_size);
          known[task_id] = 1;
        }
      };
      support::parallel_for_dynamic(0, n, context->num_threads, analyze);
    }

    double weight = ReadSoftFilterWeight();
    std::vector<double> hypothetical = ApplySoftPenalties(scores, penalties, /*mode=*/2, weight);

    std::vector<double> known_penalties;
    known_penalties.reserve(n);
    for (int i = 0; i < n; ++i) {
      if (known[i]) known_penalties.push_back(penalties[i]);
    }
    std::sort(known_penalties.begin(), known_penalties.end());
    double penalty_min = 1.0;
    double penalty_median = 1.0;
    double penalty_mean = 1.0;
    double penalty_max = 1.0;
    if (!known_penalties.empty()) {
      penalty_min = known_penalties.front();
      penalty_median = known_penalties[known_penalties.size() / 2];
      penalty_mean = std::accumulate(known_penalties.begin(), known_penalties.end(), 0.0) /
                     known_penalties.size();
      penalty_max = known_penalties.back();
    }

    int top_k = std::min(16, n);
    int top_k_overlap = TopKOverlap(scores, hypothetical, top_k);
    int64_t elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - soft_filter_start)
                             .count();
    std::ostringstream os;
    os << "Thread soft filter: mode=" << (soft_filter_mode == 1 ? "shadow" : "rerank")
       << " profile=" << (has_sm86_profile ? "sm_86-thread-v0" : "neutral") << " candidates=" << n
       << "->" << hypothetical.size() << " known=" << known_penalties.size()
       << " unknown=" << n - known_penalties.size() << " weight=" << std::fixed
       << std::setprecision(3) << weight << " penalty[min/p50/mean/max]=" << penalty_min << '/'
       << penalty_median << '/' << penalty_mean << '/' << penalty_max << " top" << top_k
       << "_overlap=" << top_k_overlap << '/' << top_k << " time_us=" << elapsed_us;
    TVM_PY_LOG(INFO, context->logger) << os.str();
    if (soft_filter_mode == 2) scores = std::move(hypothetical);
  }

  int draft_mode = ReadDraftMode();
  if (draft_mode == 0) return scores;

  const auto draft_start = std::chrono::steady_clock::now();
  std::optional<PrunerDraftHardwareProfile> profile = ReadDraftHardwareProfile(context);
  std::vector<PrunerDraftScore> draft_scores(n);
  if (profile.has_value()) {
    auto analyze = [&candidates, &draft_scores, &profile](int, int task_id) {
      draft_scores[task_id] = AnalyzePrunerDraftScore(candidates[task_id]->mod(), profile.value());
    };
    support::parallel_for_dynamic(0, n, context->num_threads, analyze);
  }
  std::vector<double> normalized = NormalizePrunerDraftScores(draft_scores);
  double beta = ReadDraftBeta();
  std::vector<double> hypothetical = ApplyPrunerDraftScores(scores, normalized, /*mode=*/2, beta);

  std::vector<double> known_raw;
  std::vector<double> known_normalized;
  int64_t statement_count = 0;
  for (int i = 0; i < n; ++i) {
    statement_count += draft_scores[i].statement_count;
    if (draft_scores[i].known) {
      known_raw.push_back(draft_scores[i].score);
      known_normalized.push_back(normalized[i]);
    }
  }
  auto distribution = [](std::vector<double> values) {
    std::array<double, 4> result{{1.0, 1.0, 1.0, 1.0}};
    if (values.empty()) return result;
    std::sort(values.begin(), values.end());
    result[0] = values.front();
    result[1] = values[values.size() / 2];
    result[2] = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    result[3] = values.back();
    return result;
  };
  std::array<double, 4> raw_distribution = distribution(known_raw);
  std::array<double, 4> normalized_distribution = distribution(known_normalized);
  int top_k = std::min(16, n);
  int top_k_overlap = TopKOverlap(scores, hypothetical, top_k);
  int64_t elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - draft_start)
                           .count();
  const char* profile_name = std::getenv("TVM_DRAFT_PROFILE");
  std::ostringstream os;
  os << "Pruner draft model: mode=" << (draft_mode == 1 ? "shadow" : "rank") << " profile="
     << (profile.has_value() ? (profile_name ? profile_name : "explicit") : "neutral")
     << " candidates=" << n << "->" << hypothetical.size() << " known=" << known_raw.size()
     << " unknown=" << n - known_raw.size() << " statements=" << statement_count
     << " beta=" << std::fixed << std::setprecision(3) << beta << " epsilon=" << ReadDraftEpsilon()
     << std::scientific << std::setprecision(6)
     << " score[min/p50/mean/max]=" << raw_distribution[0] << '/' << raw_distribution[1] << '/'
     << raw_distribution[2] << '/' << raw_distribution[3]
     << " normalized[min/p50/mean/max]=" << normalized_distribution[0] << '/'
     << normalized_distribution[1] << '/' << normalized_distribution[2] << '/'
     << normalized_distribution[3] << " top" << top_k << "_overlap=" << top_k_overlap << '/'
     << top_k << " time_us=" << elapsed_us;
  TVM_PY_LOG(INFO, context->logger) << os.str();

  return draft_mode == 2 ? hypothetical : scores;
}

/**************** Evolutionary Search ****************/

/*!\brief A search strategy that generates measure candidates using evolutionary search. */
class EvolutionarySearchNode : public SearchStrategyNode {
 public:
  /*! \brief The state of the search strategy. */
  struct State {
    /*! \brief The search strategy itself */
    EvolutionarySearchNode* self;
    /*! \brief The number of total trials. */
    int max_trials;
    /*! \brief The number of trials per iteration. */
    int num_trials_per_iter;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int st;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int ed;
    /*! \brief The counter of returning empty results. */
    int num_empty_iters;
    /*! \brief The design spaces. Decisions are not used so traces only. */
    ffi::Array<s_tir::Trace> design_spaces;
    /*! \brief Pre thread data including module to be tuned and random state. */
    std::vector<PerThreadData> per_thread_data_;
    /*!
     * \brief The workloads that are already measured.
     * TODO(junrushao1994): add records from the database to avoid re-measuring.
     * */
    IRModuleSet measured_workloads_;
    /*! \brief A Database for selecting useful candidates. */
    Database database_{ffi::UnsafeInit()};
    /*! \brief A cost model helping to explore the search space */
    CostModel cost_model_{ffi::UnsafeInit()};
    /*! \brief The token registered for the given workload in database. */
    Workload token_{ffi::UnsafeInit()};
    /*! \brief G3 mode: 0=off, 1=shadow, 2=enforce. */
    int g3_mode_{0};
    /*! \brief Mutator arms with positive mass, fixed for this State lifetime. */
    std::vector<Mutator> g3_arms_;
    /*! \brief Original (unnormalized) probability of each G3 mutator arm. */
    std::vector<double> g3_base_probs_;
    /*! \brief Real-measurement-only UCB1 reward book. */
    std::unique_ptr<MeasuredMutatorCreditBook> g3_credit_book_;
    /*! \brief Successful RunnerResult latency cache from earlier batches. */
    MeasuredLatencyTable g3_measured_latency_;
    /*! \brief Lineage aligned exactly with the outstanding MeasureCandidate array. */
    std::vector<std::optional<MutationLineage>> g3_pending_lineages_;
    /*! \brief Lifetime actual and shadow mutator selection counters. */
    std::vector<int64_t> g3_selected_counts_;
    std::vector<int64_t> g3_shadow_selected_counts_;
    std::vector<int64_t> g3_accepted_counts_;
    std::vector<int64_t> g3_batch_selected_counts_;
    std::vector<int64_t> g3_batch_accepted_counts_;
    int64_t g3_batch_id_{0};
    explicit State(EvolutionarySearchNode* self, int max_trials, int num_trials_per_iter,
                   ffi::Array<Schedule> design_space_schedules, Database database,
                   CostModel cost_model)
        : self(self),
          max_trials(max_trials),
          num_trials_per_iter(num_trials_per_iter),
          st(0),
          ed(num_trials_per_iter),
          num_empty_iters(0),
          measured_workloads_(database->GetModuleEquality()) {
      design_spaces.reserve(design_space_schedules.size());
      for (const Schedule& space : design_space_schedules) {
        design_spaces.push_back(space->trace().value()->Simplified(true));
      }
      const TuneContextNode* ctx = self->ctx_;
      IRModule mod = ctx->mod.value();
      this->per_thread_data_.resize(ctx->num_threads);
      for (PerThreadData& data : this->per_thread_data_) {
        data.mod = DeepCopyIRModule(mod);
        data.rand_state = ForkSeed(&self->rand_state_);
      }
      this->database_ = database;
      this->cost_model_ = cost_model;
      this->token_ = database->CommitWorkload(mod);
      this->g3_mode_ = ReadDraftGAG3Mode();
      if (this->g3_mode_ != 0) {
        for (const auto& kv : self->mutator_probs_) {
          if (kv.second->value > 0.0) {
            this->g3_arms_.push_back(kv.first);
            this->g3_base_probs_.push_back(kv.second->value);
          }
        }
        if (!this->g3_arms_.empty()) {
          this->g3_credit_book_ =
              std::make_unique<MeasuredMutatorCreditBook>(
                  static_cast<int>(this->g3_arms_.size()));
          this->g3_selected_counts_.assign(this->g3_arms_.size(), 0);
          this->g3_shadow_selected_counts_.assign(this->g3_arms_.size(), 0);
          this->g3_accepted_counts_.assign(this->g3_arms_.size(), 0);
          this->g3_batch_selected_counts_.assign(this->g3_arms_.size(), 0);
          this->g3_batch_accepted_counts_.assign(this->g3_arms_.size(), 0);
          std::ostringstream os;
          for (int i = 0, n = static_cast<int>(this->g3_arms_.size()); i < n; ++i) {
            if (i != 0) os << ',';
            os << i << ':' << this->g3_arms_[i]->GetTypeKey() << '@' << std::fixed
               << std::setprecision(4) << this->g3_base_probs_[i];
          }
          TVM_PY_LOG(INFO, self->ctx_->logger)
              << "Draft GA G3 arms: mode=" << DraftG3ModeName(this->g3_mode_)
              << " arms=[" << os.str() << ']';
        }
      }
    }

    int FindG3MutatorIndex(const Mutator& mutator) const {
      for (int i = 0, n = static_cast<int>(g3_arms_.size()); i < n; ++i) {
        if (g3_arms_[i].same_as(mutator)) return i;
      }
      return -1;
    }

    /*!
     * \brief Pick up best candidates from database.
     * \param num The number of traces to produce.
     * \return The picked best candidates.
     */
    inline std::vector<Schedule> PickBestFromDatabase(int num);
    /*!
     * \brief Sample the initial population from previous measured results and randomly generated
     *  traces via trace replaying.
     * \param num The number of traces to produce.
     * \return The initial population of traces sampled.
     */
    inline std::vector<Schedule> SampleInitPopulation(int num);
    /*!
     * \brief Evolve the initial population using mutators and samplers.
     * \param population The initial population of traces sampled.
     * \param num The number of traces to produce.
     * \return The evolved traces from initial population.
     */
    inline std::vector<Schedule> EvolveWithCostModel(
        std::vector<Schedule> population, int num,
        std::vector<std::optional<MutationLineage>>* result_lineages = nullptr);
    /*!
     * \brief Pick final candidates from the given initial population and bests of evolved ones.
     * \param inits The initial population of traces sampled.
     * \param bests The best candidates predicted from evolved traces.
     * \param num The number of traces to produce.
     * \return The final picked candidates with a ratio of both.
     */
    inline std::vector<Schedule> PickWithEpsGreedy(
        const std::vector<Schedule>& inits, const std::vector<Schedule>& bests, int num,
        const std::vector<std::optional<MutationLineage>>* best_lineages = nullptr,
        std::vector<std::optional<MutationLineage>>* picked_lineages = nullptr);
    /*! \brief An interface method to be called by it's counterpart in EvolutionarySearchNode */
    inline ffi::Optional<ffi::Array<MeasureCandidate>> GenerateMeasureCandidates();
    /*! \brief An interface method to be called by it's counterpart in EvolutionarySearchNode */
    inline void NotifyRunnerResults(const ffi::Array<MeasureCandidate>& measure_candidates,
                                    const ffi::Array<RunnerResult>& results);
    /*!
     * \brief Compute the hash for the given module.
     * \param mod The input TIR module.
     * \return The calculated hash.
     */
    inline size_t ModuleHash(const IRModule& mod) const;
  };

  /*! \brief The tuning context of the evolutionary search strategy. */
  const TuneContextNode* ctx_{nullptr};
  /*! \brief The postprocessors */
  ffi::Array<Postproc> postprocs_;
  /*! \brief The mutators and their probability. */
  ffi::Map<Mutator, FloatImm> mutator_probs_;
  /*! \brief The random state. To be initialized with TuneContext. */
  TRandState rand_state_;
  /*! \brief The state of the search strategy. */
  std::unique_ptr<State> state_ = nullptr;

  /*** Configuration: global ***/
  /*! \brief The population size in the evolutionary search. */
  int population_size;
  /*!
   * \brief The maximum number of iterations before early stopping to confirm the search space is
   * exhausted
   */
  int num_empty_iters_before_early_stop;
  /*** Configuration: the initial population ***/
  /*! \brief The ratio of measured states used in the initial population */
  double init_measured_ratio;
  /*! \brief The minimal size of unmeasured population in the initial sampling.*/
  int init_min_unmeasured;
  /*! \brief The maximum number of failure during initial sampling. */
  int max_fail_count;
  /*** Configuration: evolution ***/
  /*! \brief The number of iterations performed by generic algorithm. */
  int genetic_num_iters;
  /*! \brief The probability to perform mutation */
  double genetic_mutate_prob;
  /*! \brief The maximum number to try evolving the given trace. */
  int genetic_max_fail_count;
  /*** Configuration: pick states for measurement ***/
  /*! \brief The ratio of measurements to use randomly sampled states. */
  double eps_greedy;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<EvolutionarySearchNode>()
        .def_ro("population_size", &EvolutionarySearchNode::population_size)
        .def_ro("num_empty_iters_before_early_stop",
                &EvolutionarySearchNode::num_empty_iters_before_early_stop)
        .def_ro("init_measured_ratio", &EvolutionarySearchNode::init_measured_ratio)
        .def_ro("init_min_unmeasured", &EvolutionarySearchNode::init_min_unmeasured)
        .def_ro("max_fail_count", &EvolutionarySearchNode::max_fail_count)
        .def_ro("genetic_num_iters", &EvolutionarySearchNode::genetic_num_iters)
        .def_ro("genetic_mutate_prob", &EvolutionarySearchNode::genetic_mutate_prob)
        .def_ro("genetic_max_fail_count", &EvolutionarySearchNode::genetic_max_fail_count)
        .def_ro("eps_greedy", &EvolutionarySearchNode::eps_greedy);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.EvolutionarySearch",
                                    EvolutionarySearchNode, SearchStrategyNode);

  void InitializeWithTuneContext(const TuneContext& ctx) final {
    TVM_FFI_CHECK(ctx->num_threads > 0, ValueError) << "`TuneContext.num_threads` must be > 0";
    TVM_FFI_CHECK(ctx->space_generator.defined(), ValueError)
        << "`TuneContext.space_generator` must be defined";
    TVM_FFI_CHECK(ctx->space_generator.value()->postprocs.defined(), ValueError)
        << "`TuneContext.space_generator.postprocs` must be defined";
    TVM_FFI_CHECK(ctx->space_generator.value()->mutator_probs.defined(), ValueError)
        << "`TuneContext.space_generator.mutator_probs` must be defined";
    this->ctx_ = ctx.get();
    this->postprocs_ = ctx->space_generator.value()->postprocs.value();
    this->mutator_probs_ = ctx->space_generator.value()->mutator_probs.value();
    this->rand_state_ = ForkSeed(&ctx->rand_state);
    this->state_.reset();
  }

  void PreTuning(int max_trials, int num_trials_per_iter, const ffi::Array<Schedule>& design_spaces,
                 const ffi::Optional<Database>& database,
                 const ffi::Optional<CostModel>& cost_model) final {
    TVM_FFI_ICHECK(!design_spaces.empty());
    TVM_FFI_CHECK(this->ctx_ != nullptr, ValueError)
        << "Did you forget to initialize the TuneContext?";
    TVM_FFI_CHECK(database.defined(), ValueError)
        << "Database is not supplied in PreTuning. Evolutionary"
           "search algorithm requires a database to be present, so that it "
           "could sample from previously-explored population. If you do not "
           "intent to store data on disk, please use "
           "`tvm.s_tir.meta_schedule.database.MemoryDatabase`";
    TVM_FFI_CHECK(cost_model.defined(), ValueError)
        << "CostModel is not supplied in PreTuning. Evolutionary search "
           "algorithm expects a cost model to filter out potentially less efficient kernels. If "
           "you do not expect a cost model to help, please use "
           "`tvm.s_tir.meta_schedule.cost_model.RandomModel`";
    TVM_FFI_CHECK(this->state_ == nullptr, ValueError)
        << "`PreTuning` is already invoked without corresponding `PostTuning`.";
    this->state_ = std::make_unique<State>(this, max_trials, num_trials_per_iter, design_spaces,
                                           database.value(), cost_model.value());
  }

  void PostTuning() final {
    TVM_FFI_CHECK(this->state_ != nullptr, ValueError)
        << "`PostTuning` is invoked without corresponding "
           "`PreTuning`, or `PostTuning` is already invoked.";
    this->state_.reset();
  }

  ffi::Optional<ffi::Array<MeasureCandidate>> GenerateMeasureCandidates() final {
    TVM_FFI_ICHECK(this->state_ != nullptr);
    return this->state_->GenerateMeasureCandidates();
  }

  void NotifyRunnerResults(const ffi::Array<MeasureCandidate>& measure_candidates,
                           const ffi::Array<RunnerResult>& results) final {
    TVM_FFI_ICHECK(this->state_ != nullptr);
    this->state_->NotifyRunnerResults(measure_candidates, results);
  }

  SearchStrategy Clone() const final {
    ffi::ObjectPtr<EvolutionarySearchNode> n = ffi::make_object<EvolutionarySearchNode>();
    n->population_size = this->population_size;
    n->num_empty_iters_before_early_stop = this->num_empty_iters_before_early_stop;
    n->init_measured_ratio = this->init_measured_ratio;
    n->init_min_unmeasured = this->init_min_unmeasured;
    n->max_fail_count = this->max_fail_count;
    n->genetic_num_iters = this->genetic_num_iters;
    n->genetic_mutate_prob = this->genetic_mutate_prob;
    n->genetic_max_fail_count = this->genetic_max_fail_count;
    n->eps_greedy = this->eps_greedy;
    n->ctx_ = this->ctx_;
    n->rand_state_ = this->rand_state_;
    n->state_ = nullptr;  // cleared the state
    return SearchStrategy(n);
  }
};

std::vector<Schedule> EvolutionarySearchNode::State::PickBestFromDatabase(int num) {
  auto _ = Profiler::TimedScope("EvoSearch/PickBestFromDatabase");
  std::vector<s_tir::Trace> measured_traces;
  measured_traces.reserve(num);
  ffi::Array<TuningRecord> top_records = this->database_->GetTopK(this->token_, num);
  for (TuningRecord record : top_records) {
    measured_traces.push_back(record->trace);
  }
  int actual_num = measured_traces.size();
  ThreadedTraceApply pp(self->postprocs_, self->ctx_->target);
  std::vector<Schedule> results(actual_num, Schedule{nullptr});
  auto f_proc_measured = [this, &measured_traces, &results, &pp](int thread_id,
                                                                 int trace_id) -> void {
    PerThreadData& data = this->per_thread_data_.at(thread_id);
    TRandState* rand_state = &data.rand_state;
    const IRModule& mod = data.mod;
    s_tir::Trace trace = measured_traces.at(trace_id);
    Schedule& result = results.at(trace_id);
    TVM_FFI_ICHECK(!result.defined());
    if (ffi::Optional<Schedule> sch = pp.Apply(mod, trace, rand_state)) {
      result = sch.value();
    }
  };
  support::parallel_for_dynamic(0, actual_num, self->ctx_->num_threads, f_proc_measured);
  TVM_PY_LOG(INFO, self->ctx_->logger) << "Pick-Best-From-Database summary:\n"
                                       << pp.SummarizeFailures();
  if (pp.TraceFailCount() > 0) {
    TVM_PY_LOG(WARNING, self->ctx_->logger)
        << "PickBestFromDatabase skipped " << pp.TraceFailCount()
        << " candidate(s) due to trace replay failures";
  }
  std::vector<Schedule> filtered;
  filtered.reserve(actual_num);
  for (const Schedule& sch : results) {
    if (sch.defined()) {
      filtered.push_back(sch);
    }
  }
  return filtered;
}

std::vector<Schedule> EvolutionarySearchNode::State::SampleInitPopulation(int num) {
  auto _ = Profiler::TimedScope("EvoSearch/SampleInitPopulation");
  auto start = std::chrono::steady_clock::now();
  ThreadedTraceApply pp(self->postprocs_, self->ctx_->target);
  std::vector<Schedule> out_schs;
  bool exact_population = ReadExactInitPopulation();
  int target_num = exact_population ? num : self->init_min_unmeasured;
  int sample_batch_size = exact_population ? ReadExactInitBatchSize(num) : num;
  int batches = 0;
  int generated = 0;
  int postproc_passed = 0;
  int discarded = 0;
  out_schs.reserve(target_num);
  int fail_count = 0;
  while (static_cast<int>(out_schs.size()) < target_num && fail_count < self->max_fail_count) {
    std::vector<Schedule> results(sample_batch_size, Schedule{nullptr});
    auto f_proc_unmeasured = [this, &results, &pp](int thread_id, int trace_id) -> void {
      PerThreadData& data = this->per_thread_data_.at(thread_id);
      TRandState* rand_state = &data.rand_state;
      const IRModule& mod = data.mod;
      Schedule& result = results.at(trace_id);
      TVM_FFI_ICHECK(!result.defined());
      int design_space_index = s_tir::SampleInt(rand_state, 0, design_spaces.size());
      s_tir::Trace trace(design_spaces[design_space_index]->insts, {});
      if (ffi::Optional<Schedule> sch = pp.Apply(mod, trace, rand_state)) {
        result = sch.value();
      }
    };
    support::parallel_for_dynamic(0, sample_batch_size, self->ctx_->num_threads,
                                  f_proc_unmeasured);
    ++batches;
    generated += sample_batch_size;
    bool found_new = false;
    for (int i = 0; i < sample_batch_size; i++) {
      if (results[i].defined()) {
        found_new = true;
        ++postproc_passed;
        if (!exact_population || static_cast<int>(out_schs.size()) < target_num) {
          out_schs.push_back(results[i]);
        } else {
          ++discarded;
        }
      }
    }
    fail_count += !found_new;
    TVM_PY_LOG(INFO, self->ctx_->logger) << "Sample-Init-Population summary:\n"
                                         << pp.SummarizeFailures();
    if (pp.TraceFailCount() > 0) {
      TVM_PY_LOG(WARNING, self->ctx_->logger)
          << "SampleInitPopulation encountered " << pp.TraceFailCount()
          << " trace replay failure(s); invalid candidates were skipped";
    }
  }
  if (exact_population) {
    int64_t elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    TVM_PY_LOG(INFO, self->ctx_->logger)
        << "Exact init population: target=" << target_num << " returned=" << out_schs.size()
        << " generated=" << generated << " postproc_passed=" << postproc_passed
        << " discarded=" << discarded << " batches=" << batches << " time_us=" << elapsed_us;
  }
  return out_schs;
}

std::vector<Schedule> EvolutionarySearchNode::State::EvolveWithCostModel(
    std::vector<Schedule> population, int num,
    std::vector<std::optional<MutationLineage>>* result_lineages) {
  IRModuleSet learned_exists(database_->GetModuleEquality());
  IRModuleSet draft_exists(database_->GetModuleEquality());
  {
    auto _ = Profiler::TimedScope("EvoSearch/Evolve/Misc/CopyMeasuredWorkloads");
    TVM_FFI_ICHECK_GT(num, 0);
    learned_exists = this->measured_workloads_;
    draft_exists = this->measured_workloads_;
  }

  const int draft_ga_mode = ReadDraftGAMode();
  if (g3_mode_ != 0) {
    TVM_FFI_CHECK_EQ(draft_ga_mode, 2, ValueError)
        << "TVM_DRAFT_GA_G3_MODE requires TVM_DRAFT_GA_MODE=2 so that GA fitness remains "
           "Draft-only";
  }
  const bool g3_active = g3_mode_ != 0 && g3_credit_book_ != nullptr;
  const bool g3_enforce = g3_active && g3_mode_ == 2;
  std::vector<std::optional<MutationLineage>> population_lineages(
      g3_active ? population.size() : 0);
  const int anneal_mode = draft_ga_mode == 2 ? ReadDraftGAAnnealMode() : 0;
  const double anneal_t0 = ReadDraftGAAnnealT0();
  const double anneal_decay = ReadDraftGAAnnealDecay();
  const int draft_verify_count =
      PrunerDraftGAVerifyCount(self->population_size, num, ReadDraftGAVerifyRatio());
  const TuneContext context = ffi::GetRef<TuneContext>(self->ctx_);
  std::optional<PrunerDraftHardwareProfile> draft_profile;
  if (draft_ga_mode != 0) {
    draft_profile = ReadDraftHardwareProfile(context);
    if (draft_ga_mode == 2) {
      TVM_FFI_CHECK(draft_profile.has_value(), ValueError)
          << "Draft-guided GA requires a complete TVM_DRAFT_* hardware profile";
    }
  }

  SizedHeap learned_heap(num);
  SizedHeap draft_archive(draft_verify_count);
  int ga_xgb_calls = 0;
  int64_t ga_xgb_input = 0;
  int64_t ga_xgb_time_us = 0;
  int64_t draft_time_us_total = 0;
  uint64_t draft_archive_order = 0;
  DraftAnnealStats anneal_stats;

  for (int iter = 0;; ++iter) {
    if (draft_ga_mode == 2 && population.empty()) break;
    const double anneal_temperature = anneal_t0 * std::pow(anneal_decay, iter);
    std::vector<double> scores;
    std::vector<PrunerDraftScore> generation_draft_scores;
    if (draft_ga_mode != 0) {
      const auto draft_start = std::chrono::steady_clock::now();
      generation_draft_scores.resize(population.size());
      if (draft_profile.has_value()) {
        auto analyze = [&population, &generation_draft_scores, &draft_profile](int, int task_id) {
          generation_draft_scores[task_id] =
              AnalyzePrunerDraftScore(population[task_id]->mod(), draft_profile.value());
        };
        support::parallel_for_dynamic(0, population.size(), self->ctx_->num_threads, analyze);
      }

      scores = NormalizePrunerDraftGAFitness(generation_draft_scores);
      int known_count = 0;
      int64_t statement_count = 0;
      for (const PrunerDraftScore& score : generation_draft_scores) {
        known_count += score.known;
        statement_count += score.statement_count;
      }

      for (int i = 0, n = static_cast<int>(population.size()); i < n; ++i) {
        Schedule sch = population[i];
        IRModule mod = sch->mod();
        size_t shash = ModuleHash(mod);
        if (!draft_exists.Has(mod, shash)) {
          draft_exists.Add(mod, shash);
          bool archive_known =
              generation_draft_scores[i].known && std::isfinite(generation_draft_scores[i].score);
          double archive_score = archive_known ? generation_draft_scores[i].score : 0.0;
          draft_archive.Push(sch, archive_score, g3_active ? population_lineages[i] : std::nullopt,
                             archive_known ? 1 : 0, draft_archive_order++);
        }
      }
      int64_t draft_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - draft_start)
                                  .count();
      draft_time_us_total += draft_time_us;
      const char* profile_name = std::getenv("TVM_DRAFT_PROFILE");
      TVM_PY_LOG(INFO, self->ctx_->logger)
          << "Draft GA generation: iter=" << iter
          << " mode=" << (draft_ga_mode == 1 ? "shadow" : "enforce") << " profile="
          << (draft_profile.has_value() ? (profile_name ? profile_name : "explicit") : "neutral")
          << " candidates=" << population.size() << " known=" << known_count
          << " unknown=" << population.size() - known_count << " statements=" << statement_count
          << " archive=" << draft_archive.heap.size() << '/' << draft_verify_count
          << " ga_xgb_calls=" << (draft_ga_mode == 2 ? 0 : iter + 1)
          << " draft_time_us=" << draft_time_us;
    }

    if (draft_ga_mode != 2) {
      const auto predict_start = std::chrono::steady_clock::now();
      scores = PredictNormalizedScore(population, context, this->cost_model_);
      ga_xgb_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - predict_start)
                            .count();
      ++ga_xgb_calls;
      ga_xgb_input += population.size();
    }

    {
      auto _ = Profiler::TimedScope("EvoSearch/Evolve/Misc");
      TVM_FFI_ICHECK_EQ(scores.size(), population.size());
      if (draft_ga_mode != 2) {
        for (int i = 0, n = population.size(); i < n; ++i) {
          Schedule sch = population.at(i);
          IRModule mod = sch->mod();
          size_t shash = ModuleHash(mod);
          double score = scores.at(i);
          if (!learned_exists.Has(mod, shash)) {
            learned_exists.Add(mod, shash);
            learned_heap.Push(sch, score,
                              g3_active ? population_lineages[i] : std::nullopt);
          }
        }
      }
      if (iter == self->genetic_num_iters) {
        break;
      }
      // In enforce mode, these probabilities come exclusively from Draft.
      for (PerThreadData& data : this->per_thread_data_) {
        data.Set(scores, self->genetic_mutate_prob, self->mutator_probs_);
      }
    }
    {
      auto _ = Profiler::TimedScope("EvoSearch/Evolve/Mutation");
      ThreadedTraceApply pp(self->postprocs_, self->ctx_->target);
      ConcurrentBitmask cbmask(self->population_size);
      std::vector<Schedule> next_population(self->population_size, Schedule{nullptr});
      std::vector<std::optional<MutationLineage>> next_population_lineages(
          g3_active ? self->population_size : 0);
      std::vector<std::vector<int64_t>> g3_generation_selected(
          g3_active ? self->ctx_->num_threads : 0,
          std::vector<int64_t>(g3_active ? g3_arms_.size() : 0, 0));
      std::vector<std::vector<int64_t>> g3_generation_shadow_selected(
          g3_active ? self->ctx_->num_threads : 0,
          std::vector<int64_t>(g3_active ? g3_arms_.size() : 0, 0));
      std::vector<std::vector<int64_t>> g3_generation_accepted(
          g3_active ? self->ctx_->num_threads : 0,
          std::vector<int64_t>(g3_active ? g3_arms_.size() : 0, 0));
      // The worker function
      auto f_find_candidate = [&cbmask, &population, &next_population, &pp,
                               &generation_draft_scores, &draft_profile,
                               &anneal_stats, &population_lineages, &next_population_lineages,
                               &g3_generation_selected, &g3_generation_shadow_selected,
                               &g3_generation_accepted, anneal_mode, anneal_temperature, g3_active,
                               g3_enforce, iter, this](int thread_id, int trace_id) {
        // Prepare samplers
        PerThreadData& data = this->per_thread_data_.at(thread_id);
        TRandState* rand_state = &data.rand_state;
        const IRModule& mod = data.mod;
        std::function<int()>& trace_sampler = data.trace_sampler;
        std::function<ffi::Optional<Mutator>()>& mutator_sampler = data.mutator_sampler;
        Schedule& result = next_population.at(trace_id);
        int sampled_trace_id = -1;
        // Enforce-only, per-child retry state.  Failed arms are tried without
        // replacement, but are not assigned a synthetic reward and are
        // available again for the next child.
        std::vector<uint8_t> g3_unavailable(g3_enforce ? g3_arms_.size() : 0, 0);
        auto accept_with_annealing = [&](const PrunerDraftScore& parent,
                                         const PrunerDraftScore& child) {
          if (anneal_mode == 0) return true;
          if (!parent.known || !child.known || !std::isfinite(parent.score) ||
              !std::isfinite(child.score) || parent.score <= 0.0 || child.score <= 0.0) {
            anneal_stats.unknown.fetch_add(1, std::memory_order_relaxed);
            return true;
          }
          anneal_stats.compared.fetch_add(1, std::memory_order_relaxed);
          if (child.score >= parent.score) {
            anneal_stats.improved_or_equal.fetch_add(1, std::memory_order_relaxed);
            return true;
          }
          anneal_stats.worse.fetch_add(1, std::memory_order_relaxed);
          double probability = PrunerDraftAnnealAcceptanceProbability(
              parent.score, child.score, anneal_temperature);
          constexpr int kAnnealProbabilityScale = 1000000;
          TRandState shadow_rand_state = *rand_state;
          TRandState* decision_rand_state =
              anneal_mode == 1 ? &shadow_rand_state : rand_state;
          bool would_accept =
              probability >= 1.0 ||
              s_tir::SampleInt(decision_rand_state, 0, kAnnealProbabilityScale) <
                  static_cast<int>(probability * kAnnealProbabilityScale);
          (would_accept ? anneal_stats.accepted_worse : anneal_stats.rejected_worse)
              .fetch_add(1, std::memory_order_relaxed);
          // Shadow mode records the counterfactual decision without changing the
          // candidate or consuming the worker's random stream.
          return anneal_mode == 1 || would_accept;
        };
        // Loop until success
        for (int fail_count = 0; fail_count <= self->genetic_max_fail_count; ++fail_count) {
          sampled_trace_id = trace_sampler();
          sampled_trace_id = sampled_trace_id % self->population_size;
          s_tir::Trace trace = population.at(sampled_trace_id)->trace().value();
          if (ffi::Optional<Mutator> opt_mutator = mutator_sampler()) {
            // Decision: mutate
            Mutator mutator = opt_mutator.value();
            int g3_mutator_index = -1;
            if (g3_active) {
              int baseline_mutator_index = FindG3MutatorIndex(mutator);
              TVM_FFI_ICHECK_GE(baseline_mutator_index, 0);
              int g3_choice = baseline_mutator_index;
              if (g3_enforce) {
                g3_choice = g3_credit_book_->SelectMutator(baseline_mutator_index,
                                                           g3_unavailable);
                if (g3_choice == -1) {
                  std::fill(g3_unavailable.begin(), g3_unavailable.end(), 0);
                  g3_choice = g3_credit_book_->SelectMutator(baseline_mutator_index,
                                                             g3_unavailable);
                }
              } else {
                g3_choice = g3_credit_book_->SelectMutator(baseline_mutator_index);
              }
              TVM_FFI_ICHECK_GE(g3_choice, 0);
              if (g3_mode_ == 1) {
                ++g3_generation_shadow_selected[thread_id][g3_choice];
              } else if (g3_enforce) {
                mutator = g3_arms_[g3_choice];
              }
              g3_mutator_index = g3_enforce ? g3_choice : baseline_mutator_index;
              ++g3_generation_selected[thread_id][g3_mutator_index];
            }
            auto set_mutated_result = [&](const Schedule& child) {
              result = child;
              if (g3_active) {
                const IRModule& parent_mod = population.at(sampled_trace_id)->mod();
                next_population_lineages[trace_id] = MutationLineage{
                    parent_mod, g3_mutator_index, g3_measured_latency_.Get(parent_mod), iter};
                ++g3_generation_accepted[thread_id][g3_mutator_index];
              }
            };
            auto mark_g3_attempt_failed = [&]() {
              if (g3_enforce) g3_unavailable[g3_mutator_index] = 1;
            };
            if (ffi::Optional<s_tir::Trace> new_trace = mutator->Apply(trace, rand_state)) {
              if (ffi::Optional<Schedule> sch = pp.Apply(mod, new_trace.value(), rand_state)) {
                // note that sch's trace is different from new_trace
                // because it contains post-processing information
                if (anneal_mode == 0) {
                  set_mutated_result(sch.value());
                  break;
                }
                const auto anneal_draft_start = std::chrono::steady_clock::now();
                PrunerDraftScore child_score =
                    AnalyzePrunerDraftScore(sch.value()->mod(), draft_profile.value());
                anneal_stats.draft_time_us.fetch_add(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - anneal_draft_start)
                        .count(),
                    std::memory_order_relaxed);
                if (accept_with_annealing(generation_draft_scores[sampled_trace_id],
                                          child_score)) {
                  set_mutated_result(sch.value());
                  break;
                }
              }
            }
            mark_g3_attempt_failed();
          } else if (cbmask.QueryAndMark(sampled_trace_id)) {
            // Decision: do not mutate
            break;
          }
        }
        // if retry count exceeds the limit, reuse an old sample
        if (!result.defined()) {
          result = population.at(sampled_trace_id);
          if (g3_active) {
            next_population_lineages[trace_id] = population_lineages[sampled_trace_id];
          }
        }
      };
      support::parallel_for_dynamic(0, self->population_size, self->ctx_->num_threads,
                                    f_find_candidate);

      if (g3_active) {
        std::ostringstream selected_os;
        std::ostringstream shadow_os;
        std::ostringstream accepted_os;
        for (int arm = 0, n = static_cast<int>(g3_arms_.size()); arm < n; ++arm) {
          int64_t selected = 0;
          int64_t shadow_selected = 0;
          int64_t accepted = 0;
          for (int thread_id = 0; thread_id < self->ctx_->num_threads; ++thread_id) {
            selected += g3_generation_selected[thread_id][arm];
            shadow_selected += g3_generation_shadow_selected[thread_id][arm];
            accepted += g3_generation_accepted[thread_id][arm];
          }
          g3_selected_counts_[arm] += selected;
          g3_shadow_selected_counts_[arm] += shadow_selected;
          g3_accepted_counts_[arm] += accepted;
          g3_batch_selected_counts_[arm] += selected;
          g3_batch_accepted_counts_[arm] += accepted;
          if (arm != 0) {
            selected_os << ',';
            shadow_os << ',';
            accepted_os << ',';
          }
          selected_os << arm << ':' << selected;
          shadow_os << arm << ':' << shadow_selected;
          accepted_os << arm << ':' << accepted;
        }
        TVM_PY_LOG(INFO, self->ctx_->logger)
            << "Draft GA G3 generation: batch=" << g3_batch_id_ << " iter=" << iter
            << " mode=" << DraftG3ModeName(g3_mode_)
            << " cold_start=" << g3_credit_book_->InColdStart()
            << " selected=[" << selected_os.str() << "] shadow_selected=[" << shadow_os.str()
            << "] accepted=[" << accepted_os.str() << ']';
        population_lineages.swap(next_population_lineages);
      }
      population.swap(next_population);
      TVM_PY_LOG(INFO, self->ctx_->logger) << "Evolve iter #" << iter << " done. Summary:\n"
                                           << pp.SummarizeFailures();
    }
  }

  int final_xgb_calls = 0;
  int final_xgb_input = 0;
  int64_t final_xgb_time_us = 0;
  if (anneal_mode != 0) {
    TVM_PY_LOG(INFO, self->ctx_->logger)
        << "Draft GA annealing summary: mode=" << (anneal_mode == 1 ? "shadow" : "enforce")
        << " t0=" << std::fixed << std::setprecision(4) << anneal_t0
        << " decay=" << anneal_decay
        << " compared=" << anneal_stats.compared.load(std::memory_order_relaxed)
        << " improved_or_equal="
        << anneal_stats.improved_or_equal.load(std::memory_order_relaxed)
        << " worse=" << anneal_stats.worse.load(std::memory_order_relaxed)
        << " accepted_worse=" << anneal_stats.accepted_worse.load(std::memory_order_relaxed)
        << " rejected_worse=" << anneal_stats.rejected_worse.load(std::memory_order_relaxed)
        << " unknown=" << anneal_stats.unknown.load(std::memory_order_relaxed)
        << " draft_time_us=" << anneal_stats.draft_time_us.load(std::memory_order_relaxed);
  }
  if (draft_ga_mode == 2 && !draft_archive.heap.empty()) {
    std::sort(draft_archive.heap.begin(), draft_archive.heap.end());
    std::vector<Schedule> finalists;
    std::vector<std::optional<MutationLineage>> finalist_lineages;
    finalists.reserve(draft_archive.heap.size());
    if (g3_active) finalist_lineages.reserve(draft_archive.heap.size());
    for (const SizedHeap::Item& item : draft_archive.heap) {
      finalists.push_back(item.sch);
      if (g3_active) finalist_lineages.push_back(item.lineage);
    }

    const auto predict_start = std::chrono::steady_clock::now();
    std::vector<double> final_scores =
        PredictCostModelNormalizedScore(finalists, context, this->cost_model_);
    TVM_FFI_ICHECK_EQ(final_scores.size(), finalists.size());
    final_xgb_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - predict_start)
                            .count();
    final_xgb_calls = 1;
    final_xgb_input = finalists.size();
    for (int i = 0, n = static_cast<int>(finalists.size()); i < n; ++i) {
      learned_heap.Push(finalists[i], final_scores[i],
                        g3_active ? finalist_lineages[i] : std::nullopt);
    }
  }

  // Return the best states from the heap, sorting from higher score to lower ones
  {
    auto _ = Profiler::TimedScope("EvoSearch/Evolve/Misc");
    std::sort(learned_heap.heap.begin(), learned_heap.heap.end());
    std::vector<Schedule> results;
    results.reserve(num);
    if (result_lineages != nullptr) {
      result_lineages->clear();
      result_lineages->reserve(num);
    }
    for (const SizedHeap::Item& item : learned_heap.heap) {
      results.push_back(item.sch);
      if (result_lineages != nullptr) result_lineages->push_back(item.lineage);
    }

    constexpr int kNumScoresPerLine = 16;
    std::ostringstream os;
    int n = learned_heap.heap.size();
    for (int st = 0; st < n; st += kNumScoresPerLine) {
      os << std::endl;
      int ed = std::min(st + kNumScoresPerLine, n);
      os << "[" << (st + 1) << " : " << ed << "]:\t";
      for (int i = st; i < ed; ++i) {
        if (i != st) {
          os << "  ";
        }
        os << std::fixed << std::setprecision(4) << learned_heap.heap.at(i).score;
      }
    }
    TVM_PY_LOG(INFO, self->ctx_->logger)
        << "Scores of the best " << n << " candidates:" << os.str();
    TVM_PY_LOG(INFO, self->ctx_->logger)
        << "Evolutionary scoring summary: mode="
        << (draft_ga_mode == 0 ? "baseline" : (draft_ga_mode == 1 ? "draft-shadow"
                                                                    : "draft-enforce"))
        << " ga_generations=" << self->genetic_num_iters + 1
        << " ga_xgb_calls=" << ga_xgb_calls << " ga_xgb_input=" << ga_xgb_input
        << " final_xgb_calls=" << final_xgb_calls << " final_xgb_input=" << final_xgb_input
        << " draft_archive=" << (draft_ga_mode == 0 ? 0 : draft_archive.heap.size())
        << " verify_top_k=" << (draft_ga_mode == 0 ? 0 : draft_verify_count)
        << " returned=" << results.size() << " draft_time_us=" << draft_time_us_total
        << " ga_xgb_time_us=" << ga_xgb_time_us
        << " final_xgb_time_us=" << final_xgb_time_us;
    return results;
  }
}

std::vector<Schedule> EvolutionarySearchNode::State::PickWithEpsGreedy(
    const std::vector<Schedule>& unmeasured, const std::vector<Schedule>& bests, int num,
    const std::vector<std::optional<MutationLineage>>* best_lineages,
    std::vector<std::optional<MutationLineage>>* picked_lineages) {
  auto _ = Profiler::TimedScope("EvoSearch/PickWithEpsGreedy");
  if (best_lineages != nullptr) {
    TVM_FFI_ICHECK_EQ(best_lineages->size(), bests.size());
  }
  if (picked_lineages != nullptr) {
    picked_lineages->clear();
    picked_lineages->reserve(num);
  }
  double epsilon = ReadDraftGAMode() == 2
                       ? ReadDraftGAFinalEpsilon()
                       : (ReadDraftMode() == 2 ? ReadDraftEpsilon() : self->eps_greedy);
  int num_rands = num * epsilon;
  int num_bests = num - num_rands;
  TVM_PY_LOG(INFO, self->ctx_->logger)
      << "Final selection quota: epsilon=" << epsilon << " model=" << num_bests
      << " random=" << num_rands;
  std::vector<int> rands =
      s_tir::SampleWithoutReplacement(&self->rand_state_, unmeasured.size(), unmeasured.size());
  std::vector<Schedule> results;
  results.reserve(num);
  IRModuleSet& measured_workloads = this->measured_workloads_;
  for (int i = 0, i_bests = 0, i_rands = 0; i < num; ++i) {
    bool has_best = i_bests < static_cast<int>(bests.size());
    bool has_rand = i_rands < static_cast<int>(rands.size());
    // Pick a schedule
    Schedule sch{nullptr};
    std::optional<MutationLineage> lineage;
    // If needs `bests`, then prefer `bests`
    if (i < num_bests) {
      if (has_best) {
        int best_index = i_bests++;
        sch = bests[best_index];
        if (best_lineages != nullptr) lineage = best_lineages->at(best_index);
      } else if (has_rand) {
        sch = unmeasured[rands[i_rands++]];
      } else {
        break;
      }
    } else {
      // Else prefer `rands`
      if (has_rand) {
        sch = unmeasured[rands[i_rands++]];
      } else if (has_best) {
        int best_index = i_bests++;
        sch = bests[best_index];
        if (best_lineages != nullptr) lineage = best_lineages->at(best_index);
      } else {
        break;
      }
    }
    IRModule mod = sch->mod();
    size_t shash = ModuleHash(mod);
    if (!measured_workloads.Has(mod, shash)) {
      measured_workloads.Add(mod, shash);
      results.push_back(sch);
      if (picked_lineages != nullptr) picked_lineages->push_back(std::move(lineage));
    }
  }
  return results;
}

ffi::Optional<ffi::Array<MeasureCandidate>>
EvolutionarySearchNode::State::GenerateMeasureCandidates() {
  if (st >= max_trials) {
    return std::nullopt;
  }
  int sample_num = num_trials_per_iter;
  if (ed > max_trials) {
    sample_num = max_trials - st;
    ed = max_trials;
  }
  TVM_FFI_ICHECK_LT(st, ed);
  int pop = self->population_size;
  std::vector<Schedule> inits;
  inits.reserve(pop);

  TVM_PY_LOG(INFO, self->ctx_->logger) << "Generating candidates......";
  std::vector<Schedule> measured = PickBestFromDatabase(pop * self->init_measured_ratio);
  TVM_PY_LOG(INFO, self->ctx_->logger)
      << "Picked top " << measured.size() << " candidate(s) from database";
  std::vector<Schedule> unmeasured = SampleInitPopulation(pop - measured.size());
  if (static_cast<int>(unmeasured.size()) < self->init_min_unmeasured) {
    TVM_PY_LOG(WARNING, self->ctx_->logger)
        << "Cannot sample enough initial population, evolutionary search failed.";
    return std::nullopt;
  }
  TVM_PY_LOG(INFO, self->ctx_->logger) << "Sampled " << unmeasured.size() << " candidate(s)";
  inits.insert(inits.end(), measured.begin(), measured.end());
  inits.insert(inits.end(), unmeasured.begin(), unmeasured.end());
  const bool g3_active = g3_mode_ != 0 && g3_credit_book_ != nullptr;
  std::vector<std::optional<MutationLineage>> best_lineages;
  if (g3_active) {
    ++g3_batch_id_;
    std::fill(g3_batch_selected_counts_.begin(), g3_batch_selected_counts_.end(), 0);
    std::fill(g3_batch_accepted_counts_.begin(), g3_batch_accepted_counts_.end(), 0);
  }
  std::vector<Schedule> bests =
      g3_active ? EvolveWithCostModel(inits, sample_num, &best_lineages)
                : EvolveWithCostModel(inits, sample_num);
  TVM_PY_LOG(INFO, self->ctx_->logger)
      << "Got " << bests.size() << " candidate(s) with evolutionary search";
  std::vector<std::optional<MutationLineage>> picked_lineages;
  std::vector<Schedule> picks =
      g3_active ? PickWithEpsGreedy(unmeasured, bests, sample_num, &best_lineages,
                                    &picked_lineages)
                : PickWithEpsGreedy(unmeasured, bests, sample_num);
  if (g3_active) {
    TVM_FFI_ICHECK_EQ(picks.size(), picked_lineages.size());
    g3_pending_lineages_ = std::move(picked_lineages);
  }
  TVM_PY_LOG(INFO, self->ctx_->logger)
      << "Sending " << picks.size() << " candidates(s) for measurement";
  if (picks.empty()) {
    ++this->num_empty_iters;
    if (this->num_empty_iters >= self->num_empty_iters_before_early_stop) {
      return std::nullopt;
    }
  }
  return AssembleCandidates(picks);
}

void EvolutionarySearchNode::State::NotifyRunnerResults(
    const ffi::Array<MeasureCandidate>& measure_candidates,
    const ffi::Array<RunnerResult>& results) {
  TVM_FFI_ICHECK_EQ(measure_candidates.size(), results.size());
  const bool g3_active = g3_mode_ != 0 && g3_credit_book_ != nullptr;
  if (g3_active) {
    TVM_FFI_ICHECK_EQ(g3_pending_lineages_.size(), results.size());
    DraftG3FeedbackStats feedback;
    std::vector<int64_t> arm_credited(g3_arms_.size(), 0);
    std::vector<int64_t> arm_improved(g3_arms_.size(), 0);
    std::vector<std::optional<double>> child_latencies(results.size());
    for (int i = 0, n = static_cast<int>(results.size()); i < n; ++i) {
      child_latencies[i] = MedianValidRunSecs(results[i]);
      if (child_latencies[i].has_value()) ++feedback.child_measured;
    }

    // Rewards are resolved before inserting this batch into the latency cache.  Therefore a
    // parent latency can only have come from a successful RunnerResult in an earlier batch.
    for (int i = 0, n = static_cast<int>(results.size()); i < n; ++i) {
      if (!child_latencies[i].has_value()) {
        ++feedback.skipped_child_error;
        continue;
      }
      const std::optional<MutationLineage>& lineage = g3_pending_lineages_[i];
      if (!lineage.has_value()) {
        ++feedback.skipped_no_lineage;
        continue;
      }
      ++feedback.with_lineage;
      if (!lineage->measured_parent_latency_sec.has_value()) {
        ++feedback.skipped_parent_unmeasured;
        continue;
      }
      ++feedback.with_measured_parent;
      double parent_latency_sec = lineage->measured_parent_latency_sec.value();
      double child_latency_sec = child_latencies[i].value();
      g3_credit_book_->RecordMeasuredOutcome(lineage->mutator_index, parent_latency_sec,
                                             child_latency_sec);
      ++feedback.credited;
      ++arm_credited[lineage->mutator_index];
      bool improved = child_latency_sec < parent_latency_sec;
      feedback.improved += improved;
      arm_improved[lineage->mutator_index] += improved;
    }

    for (int i = 0, n = static_cast<int>(results.size()); i < n; ++i) {
      if (child_latencies[i].has_value()) {
        g3_measured_latency_.Set(measure_candidates[i]->sch->mod(), child_latencies[i].value());
      }
    }

    std::vector<double> ucb_scores = g3_credit_book_->UCB1Scores();
    std::ostringstream arms_os;
    for (int i = 0, n = static_cast<int>(g3_arms_.size()); i < n; ++i) {
      MeasuredMutatorCreditStats arm = g3_credit_book_->Stats(i);
      if (i != 0) arms_os << ',';
      arms_os << i << ":{type=" << g3_arms_[i]->GetTypeKey()
              << ",selected=" << g3_batch_selected_counts_[i]
              << ",selected_total=" << g3_selected_counts_[i]
              << ",shadow_selected=" << g3_shadow_selected_counts_[i]
              << ",accepted=" << g3_batch_accepted_counts_[i]
              << ",accepted_total=" << g3_accepted_counts_[i]
              << ",credited=" << arm_credited[i] << ",improved_batch=" << arm_improved[i]
              << ",window_outcomes=" << arm.window_count
              << ",window_improved=" << arm.window_improvements << ",mean=" << std::fixed
              << std::setprecision(4) << arm.mean_reward << ",ucb=" << ucb_scores[i] << '}';
    }
    TVM_PY_LOG(INFO, self->ctx_->logger)
        << "Draft GA G3 feedback: batch=" << g3_batch_id_
        << " mode=" << DraftG3ModeName(g3_mode_) << " source=runner sent=" << results.size()
        << " child_measured=" << feedback.child_measured
        << " with_lineage=" << feedback.with_lineage
        << " with_measured_parent=" << feedback.with_measured_parent
        << " credited=" << feedback.credited << " improved=" << feedback.improved
        << " skipped_child_error=" << feedback.skipped_child_error
        << " skipped_no_lineage=" << feedback.skipped_no_lineage
        << " skipped_parent_unmeasured=" << feedback.skipped_parent_unmeasured
        << " cold_start=" << g3_credit_book_->InColdStart()
        << " total_outcomes=" << g3_credit_book_->total_measured_outcomes() << " arms=["
        << arms_os.str() << ']';
    g3_pending_lineages_.clear();
  }
  st += results.size();
  ed += results.size();
}

size_t EvolutionarySearchNode::State::ModuleHash(const IRModule& mod) const {
  return database_->GetModuleEquality().Hash(mod);
}

SearchStrategy SearchStrategy::EvolutionarySearch(int population_size,         //
                                                  double init_measured_ratio,  //
                                                  int init_min_unmeasured,     //
                                                  int max_fail_count,          //
                                                  int genetic_num_iters,       //
                                                  double genetic_mutate_prob,  //
                                                  int genetic_max_fail_count,  //
                                                  double eps_greedy) {
  TVM_META_SCHEDULE_CHECK_PROB_RANGE(init_measured_ratio, "Initial measured ratio");
  TVM_META_SCHEDULE_CHECK_PROB_RANGE(genetic_mutate_prob, "Mutation probability");
  TVM_META_SCHEDULE_CHECK_PROB_RANGE(eps_greedy, "Greedy pick probability");
  ffi::ObjectPtr<EvolutionarySearchNode> n = ffi::make_object<EvolutionarySearchNode>();
  n->population_size = population_size;
  n->num_empty_iters_before_early_stop = 5;
  n->init_measured_ratio = init_measured_ratio;
  n->init_min_unmeasured = init_min_unmeasured;
  n->max_fail_count = max_fail_count;
  n->genetic_num_iters = genetic_num_iters;
  n->genetic_max_fail_count = genetic_max_fail_count;
  n->genetic_mutate_prob = genetic_mutate_prob;
  n->eps_greedy = eps_greedy;
  return SearchStrategy(n);
}

class EvolutionarySearch : public SearchStrategy {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(EvolutionarySearch, SearchStrategy,
                                                EvolutionarySearchNode);
};

ffi::Array<int64_t> EvolutionarySearchG3Stats(EvolutionarySearch self) {
  TVM_FFI_CHECK(self->state_ != nullptr, ValueError)
      << "EvolutionarySearch must be in a PreTuning/PostTuning lifecycle";
  const EvolutionarySearchNode::State& state = *self->state_;
  int64_t pending_with_lineage = 0;
  int64_t pending_with_measured_parent = 0;
  for (const std::optional<MutationLineage>& lineage : state.g3_pending_lineages_) {
    if (lineage.has_value()) {
      ++pending_with_lineage;
      pending_with_measured_parent += lineage->measured_parent_latency_sec.has_value();
    }
  }
  int64_t outcomes = state.g3_credit_book_ == nullptr
                         ? 0
                         : state.g3_credit_book_->total_measured_outcomes();
  int64_t cold_start =
      state.g3_credit_book_ == nullptr ? 1 : state.g3_credit_book_->InColdStart();
  return ffi::Array<int64_t>{state.g3_mode_,
                             outcomes,
                             static_cast<int64_t>(state.g3_measured_latency_.Size()),
                             static_cast<int64_t>(state.g3_pending_lineages_.size()),
                             pending_with_lineage,
                             pending_with_measured_parent,
                             cold_start,
                             static_cast<int64_t>(state.g3_arms_.size())};
}

ffi::Array<int64_t> EvolutionarySearchG3PendingMutators(EvolutionarySearch self) {
  TVM_FFI_CHECK(self->state_ != nullptr, ValueError)
      << "EvolutionarySearch must be in a PreTuning/PostTuning lifecycle";
  ffi::Array<int64_t> result;
  result.reserve(self->state_->g3_pending_lineages_.size());
  for (const std::optional<MutationLineage>& lineage : self->state_->g3_pending_lineages_) {
    result.push_back(lineage.has_value() ? lineage->mutator_index : -1);
  }
  return result;
}

ffi::Array<Schedule> EvolutionarySearchSampleInitPopulation(EvolutionarySearch self, int num) {
  std::vector<Schedule> results = self->state_->SampleInitPopulation(num);
  return ffi::Array<Schedule>(results.begin(), results.end());
}

ffi::Array<Schedule> EvolutionarySearchEvolveWithCostModel(EvolutionarySearch self,
                                                           ffi::Array<Schedule> population,
                                                           int num) {
  ffi::Array<Schedule> result;
  std::vector<Schedule> population_vec =
      std::vector<Schedule>(population.begin(), population.end());
  std::vector<Schedule> schs = self->state_->EvolveWithCostModel(population_vec, num);
  for (Schedule sch : schs) {
    IRModule mod = sch->mod();
    size_t shash = self->state_->ModuleHash(mod);
    if (!self->state_->measured_workloads_.Has(mod, shash)) {
      self->state_->measured_workloads_.Add(mod, shash);
      result.push_back(sch);
    }
  }
  return result;
}

TVM_FFI_STATIC_INIT_BLOCK() { EvolutionarySearchNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.meta_schedule.SearchStrategyEvolutionarySearch",
           SearchStrategy::EvolutionarySearch)
      .def("s_tir.meta_schedule.SearchStrategyEvolutionarySearchSampleInitPopulation",
           EvolutionarySearchSampleInitPopulation)
      .def("s_tir.meta_schedule.SearchStrategyEvolutionarySearchEvolveWithCostModel",
           EvolutionarySearchEvolveWithCostModel)
      .def("s_tir.meta_schedule.testing.EvolutionarySearchG3Stats",
           EvolutionarySearchG3Stats)
      .def("s_tir.meta_schedule.testing.EvolutionarySearchG3PendingMutators",
           EvolutionarySearchG3PendingMutators)
      .def(
          "s_tir.meta_schedule.testing.ApplyThreadSoftFilter",
          [](ffi::Array<FloatImm> raw_scores, ffi::Array<int64_t> thread_counts, int64_t warp_size,
             int mode, double weight) {
            TVM_FFI_CHECK_EQ(raw_scores.size(), thread_counts.size(), ValueError)
                << "Scores and thread counts must have the same length";
            std::vector<double> scores;
            std::vector<double> penalties;
            scores.reserve(raw_scores.size());
            penalties.reserve(raw_scores.size());
            for (int i = 0, n = raw_scores.size(); i < n; ++i) {
              scores.push_back(raw_scores[i]->value);
              penalties.push_back(mode == 0 ? 1.0 : ThreadSoftPenalty(thread_counts[i], warp_size));
            }
            std::vector<double> adjusted =
                ApplySoftPenalties(scores, penalties, mode, std::min(1.0, std::max(0.0, weight)));
            ffi::Array<FloatImm> result;
            result.reserve(penalties.size() + adjusted.size());
            for (double penalty : penalties) {
              result.push_back(FloatImm(DataType::Float(64), penalty));
            }
            for (double score : adjusted) {
              result.push_back(FloatImm(DataType::Float(64), score));
            }
            return result;
          });
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

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
#ifndef TVM_S_TIR_META_SCHEDULE_UTILS_H_
#define TVM_S_TIR_META_SCHEDULE_UTILS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/optional.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/meta_schedule/arg_info.h>
#include <tvm/s_tir/meta_schedule/builder.h>
#include <tvm/s_tir/meta_schedule/cost_model.h>
#include <tvm/s_tir/meta_schedule/database.h>
#include <tvm/s_tir/meta_schedule/extracted_task.h>
#include <tvm/s_tir/meta_schedule/feature_extractor.h>
#include <tvm/s_tir/meta_schedule/measure_callback.h>
#include <tvm/s_tir/meta_schedule/profiler.h>
#include <tvm/s_tir/meta_schedule/runner.h>
#include <tvm/s_tir/meta_schedule/schedule_rule.h>
#include <tvm/s_tir/meta_schedule/search_strategy.h>
#include <tvm/s_tir/meta_schedule/space_generator.h>
#include <tvm/s_tir/meta_schedule/task_scheduler.h>
#include <tvm/s_tir/meta_schedule/tune_context.h>
#include <tvm/s_tir/schedule/schedule.h>
#include <tvm/support/io.h>
#include <tvm/support/serializer.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../support/base64.h"
#include "../../support/bytes_io.h"
#include "../../support/utils.h"
#include "constraints.h"
#include "../schedule/primitive.h"
#include "../schedule/utils.h"
#include "../support/array_utils.h"
#include "../support/nd_int_set.h"
#include "../support/parallel_for.h"
#include "../support/table_printer.h"

#define TVM_PY_LOG(logging_level, logger)                                       \
  ::tvm::s_tir::meta_schedule::PyLogMessage(__FILE__, __LINE__, logger,         \
                                            PyLogMessage::Level::logging_level) \
      .stream()
#define TVM_PY_LOG_CLEAR_SCREEN(logging_func) clear_logging(__FILE__, __LINE__, logging_func)

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*!
 * \brief Class to accumulate an log message on the python side. Do not use directly, instead use
 * TVM_PY_LOG(DEBUG), TVM_PY_LOG(INFO), TVM_PY_LOG(WARNING), TVM_PY_ERROR(ERROR).
 * \sa TVM_PY_LOG
 * \sa TVM_PY_LOG_CLEAR_SCREEN
 */
class PyLogMessage {
 public:
  enum class Level : int32_t {
    CLEAR = -10,
    DEBUG = 10,
    INFO = 20,
    WARNING = 30,
    ERROR = 40,
    // FATAL not included
  };

  explicit PyLogMessage(const char* filename, int lineno, ffi::Function logger, Level logging_level)
      : filename_(filename), lineno_(lineno), logger_(logger), logging_level_(logging_level) {}

  TVM_FFI_NO_INLINE ~PyLogMessage() {
    TVM_FFI_ICHECK(logging_level_ != Level::CLEAR)
        << "Cannot use CLEAR as logging level in TVM_PY_LOG, please use TVM_PY_LOG_CLEAR_SCREEN.";
    if (this->logger_ != nullptr) {
      logger_(static_cast<int>(logging_level_), std::string(filename_), lineno_, stream_.str());
    } else {
      if (logging_level_ == Level::INFO) {
        runtime::detail::LogMessage(filename_, lineno_, TVM_LOG_LEVEL_INFO).stream()
            << stream_.str();
      } else if (logging_level_ == Level::WARNING) {
        runtime::detail::LogMessage(filename_, lineno_, TVM_LOG_LEVEL_WARNING).stream()
            << stream_.str();
      } else if (logging_level_ == Level::ERROR) {
        runtime::detail::LogMessage(filename_, lineno_, TVM_LOG_LEVEL_ERROR).stream()
            << stream_.str();
      } else if (logging_level_ == Level::DEBUG) {
        runtime::detail::LogMessage(filename_, lineno_, TVM_LOG_LEVEL_DEBUG).stream()
            << stream_.str();
      } else {
        runtime::detail::LogFatal(filename_, lineno_).stream() << stream_.str();
      }
    }
  }
  std::ostringstream& stream() { return stream_; }

 private:
  const char* filename_;
  int lineno_;
  std::ostringstream stream_;
  ffi::Function logger_;
  Level logging_level_;
};

/*!
 * \brief Whether the tuning is running on ipython kernel.
 * \return A boolean indicating whether ipython kernel is used.
 */
inline bool using_ipython() {
  bool flag = false;
  const auto f_using_ipython = tvm::ffi::Function::GetGlobal("s_tir.meta_schedule.using_ipython");
  if (f_using_ipython) {
    flag = (*f_using_ipython)().cast<bool>();
  }
  return flag;
}

/*!
 * \brief Print out the performance table interactively in jupyter notebook.
 * \param str The serialized performance table.
 */
inline void print_interactive_table(const ffi::String& data) {
  const auto f_print_interactive_table =
      tvm::ffi::Function::GetGlobal("s_tir.meta_schedule.print_interactive_table");
  TVM_FFI_ICHECK(f_print_interactive_table.has_value())
      << "Cannot find print_interactive_table function in registry.";
  (*f_print_interactive_table)(data);
}

/*!
 * \brief A helper function to clear logging output for ipython kernel and console.
 * \param file The file name.
 * \param lineno The line number.
 * \param logging_func The logging function.
 */
inline void clear_logging(const char* file, int lineno, ffi::Function logging_func) {
  if (const char* env_p = std::getenv("TVM_S_TIR_META_SCHEDULE_CLEAR_SCREEN")) {
    if (std::string(env_p) == "1") {
      if (logging_func.defined() && using_ipython()) {
        logging_func(static_cast<int>(PyLogMessage::Level::CLEAR), file, lineno, "");
      } else {
        // this would clear all logging output in the console
        runtime::detail::LogMessage(file, lineno, TVM_LOG_LEVEL_INFO).stream()
            << "\033c\033[3J\033[2J\033[0m\033[H";
      }
    }
  }
}

/*! \brief The type of the random state */
using TRandState = LinearCongruentialEngine::TRandState;

/*!
 * \brief Get the base64 encoded result of a string.
 * \param str The string to encode.
 * \return The base64 encoded string.
 */
inline std::string Base64Encode(std::string str) {
  std::string m_buffer;
  support::BytesOutStream m_stream(&m_buffer);
  support::Base64OutStream b64stream(&m_stream);
  static_cast<support::Stream*>(&b64stream)->Write(str);
  b64stream.Finish();
  return m_buffer;
}

/*!
 * \brief Get the base64 decoded result of a string.
 * \param str The string to decode.
 * \return The base64 decoded string.
 */
inline std::string Base64Decode(std::string str) {
  std::string result;
  support::BytesInStream m_stream(str);
  support::Base64InStream b64stream(&m_stream);
  b64stream.InitPosition();
  static_cast<support::Stream*>(&b64stream)->Read(&result);
  return result;
}

/*!
 * \brief Parses a json string into a json object.
 * \param json_str The json string.
 * \return The json object
 */
Any JSONLoads(std::string json_str);

/*!
 * \brief Dumps a json object into a json string.
 * \param json_obj The json object.
 * \return The json string
 */
std::string JSONDumps(Any json_obj);

/*!
 * \brief Converts a structural hash code to string
 * \param hash_code The hash code
 * \return The string representation of the hash code
 */
inline ffi::String SHash2Str(Workload::THashCode hash_code) { return std::to_string(hash_code); }

/*!
 * \brief Converts an TVM object to the hex string representation of its structural hash.
 * \param obj The TVM object.
 * \return The hex string representation of the hash code.
 */
inline ffi::String SHash2Hex(const ffi::ObjectRef& obj) {
  std::ostringstream os;
  size_t hash_code = 0;
  if (obj.defined()) {
    hash_code = ffi::StructuralHash()(obj);
  }
  os << "0x" << std::setw(16) << std::setfill('0') << std::hex << hash_code;
  return os.str();
}

/*!
 * \brief Fork a random state into another, i.e. PRNG splitting.
 * The given random state is also mutated.
 * \param rand_state The random state to be forked
 * \return The forked random state
 */
inline LinearCongruentialEngine::TRandState ForkSeed(
    LinearCongruentialEngine::TRandState* rand_state) {
  return LinearCongruentialEngine(rand_state).ForkSeed();
}

/*!
 * \brief Fork a random state into another ones, i.e. PRNG splitting.
 *  The given random state is also mutated.
 * \param rand_state The random state to be forked
 * \param n The number of forks
 * \return The forked random states
 */
inline std::vector<LinearCongruentialEngine::TRandState> ForkSeed(
    LinearCongruentialEngine::TRandState* rand_state, int n) {
  std::vector<LinearCongruentialEngine::TRandState> results;
  results.reserve(n);
  for (int i = 0; i < n; ++i) {
    results.push_back(LinearCongruentialEngine(rand_state).ForkSeed());
  }
  return results;
}

/*!
 * \brief Get deep copy of an IRModule.
 * \param mod The IRModule to make a deep copy.
 * \return The deep copy of the IRModule.
 */
inline IRModule DeepCopyIRModule(IRModule mod) {
  return ffi::FromJSONGraph(ffi::ToJSONGraph(mod)).cast<IRModule>();
}

/*!
 * \brief Concatenate strings
 * \param strs The strings to concatenate
 * \param delim The delimiter
 * \return The concatenated string
 */
inline std::string Concat(const ffi::Array<ffi::String>& strs, const std::string& delim) {
  if (strs.empty()) {
    return "";
  }
  std::ostringstream os;
  os << strs[0];
  for (int i = 1, n = strs.size(); i < n; ++i) {
    os << delim << strs[i];
  }
  return os.str();
}

/*! \brief Candidate-level primary reason reported by the VerifyGPUCode postprocessor. */
enum class VerifyGPUCodeFailureReason : int {
  kNone = 0,
  kThreadExtent,
  kThreadsPerBlock,
  kSharedMemory,
  kLocalMemory,
  kVThread,
  kVectorBytes,
  kKernelCount,
  kLoweringException,
  kUnknown,
  kCount,
};

inline const char* VerifyGPUCodeFailureReasonName(VerifyGPUCodeFailureReason reason) {
  switch (reason) {
    case VerifyGPUCodeFailureReason::kNone:
      return "none";
    case VerifyGPUCodeFailureReason::kThreadExtent:
      return "thread_extent";
    case VerifyGPUCodeFailureReason::kThreadsPerBlock:
      return "threads_per_block";
    case VerifyGPUCodeFailureReason::kSharedMemory:
      return "shared_memory";
    case VerifyGPUCodeFailureReason::kLocalMemory:
      return "local_memory";
    case VerifyGPUCodeFailureReason::kVThread:
      return "vthread";
    case VerifyGPUCodeFailureReason::kVectorBytes:
      return "vector_bytes";
    case VerifyGPUCodeFailureReason::kKernelCount:
      return "kernel_count";
    case VerifyGPUCodeFailureReason::kLoweringException:
      return "lowering_exception";
    case VerifyGPUCodeFailureReason::kUnknown:
      return "unknown_verify_failure";
    case VerifyGPUCodeFailureReason::kCount:
      break;
  }
  return "unknown_verify_failure";
}

/*!
 * \brief Get the current thread's primary VerifyGPUCode failure reason.
 *
 * The value is reset at the beginning of each VerifyGPUCode postprocessor call.
 */
VerifyGPUCodeFailureReason GetLastVerifyGPUCodeFailureReason();

/*!
 * \brief Get the SBlockRV from a block StmtSRef
 * \param sch The schedule
 * \param block_sref The block StmtSRef
 * \param global_var_name The global variable name
 * \return The SBlockRV
 */
inline s_tir::SBlockRV GetRVFromSRef(const s_tir::Schedule& sch, const tirx::StmtSRef& block_sref,
                                     const ffi::String& global_var_name) {
  const tirx::SBlockNode* block = TVM_SREF_TO_SBLOCK(block_sref);
  return sch->GetSBlock(block->name_hint, global_var_name);
}

/*!
 * \brief A helper data structure that replays a trace and collects failure counts
 * for each postprocessor
 */
struct ThreadedTraceApply {
  /*! \brief Constructor */
  explicit ThreadedTraceApply(const ffi::Array<Postproc>& postprocs,
                               ffi::Optional<Target> target = std::nullopt)
      : n_(postprocs.size()),
        items_(new Item[n_]),
        sampling_mode_(ReadMode("TVM_THREAD_CONSTRAINT_SAMPLE_MODE", 0)),
        hard_filter_mode_(ReadHardFilterMode()),
        hard_filter_position_(ReadHardFilterPosition()) {
    for (int i = 0; i < n_; ++i) {
      items_[i].postproc = postprocs[i];
      items_[i].fail_counter = 0;
      if (verify_gpu_index_ == -1 &&
          std::string(postprocs[i]->GetTypeKey()) == "s_tir.meta_schedule.VerifyGPUCode") {
        verify_gpu_index_ = i;
      }
    }
    if (target.has_value()) {
      const Target& value = target.value();
      max_threads_per_block_ =
          value->GetAttr<int64_t>("max_threads_per_block")
              .value_or(value->GetAttr<int64_t>("max_num_threads").value_or(-1));
      thread_warp_size_ = value->GetAttr<int64_t>("thread_warp_size").value_or(-1);
      max_shared_memory_per_block_ =
          value->GetAttr<int64_t>("max_shared_memory_per_block").value_or(-1);
    }
  }

  /*! \brief Destructor */
  ~ThreadedTraceApply() { delete[] items_; }

  /*!
   * \brief Apply the trace and postprocessors to an IRModule
   * \param mod The IRModule to be applied
   * \param trace The trace to apply to the IRModule
   * \param rand_state The random seed
   * \return The schedule created, or std::nullopt if any postprocessor fails
   */
  ffi::Optional<s_tir::Schedule> Apply(const IRModule& mod, const s_tir::Trace& trace,
                                       TRandState* rand_state) {
    generated_counter_++;
    bool plan_ready = sampling_mode_ != 0 ||
                      (hard_filter_mode_ != 0 && hard_filter_position_ == kTrace);
    TraceConstraintPlan plan;
    if (plan_ready) {
      const auto plan_start = std::chrono::steady_clock::now();
      plan = BuildTraceConstraintPlan(trace, max_threads_per_block_, thread_warp_size_);
      plan_time_ns_.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - plan_start)
              .count(),
          std::memory_order_relaxed);
      plan_built_counter_++;
      plan_contract_blocks_counter_.fetch_add(plan.contract_blocks, std::memory_order_relaxed);
      plan_enforceable_counter_.fetch_add(plan.enforceable_constraints,
                                          std::memory_order_relaxed);
    }

    bool hard_prediction = false;
    bool hard_known = false;
    if (hard_filter_mode_ != 0 && hard_filter_position_ == kTrace) {
      thread_checked_counter_++;
      const auto analysis_start = std::chrono::steady_clock::now();
      TraceConstraintEvaluation evaluation = EvaluateTraceConstraints(plan, trace);
      const int64_t elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - analysis_start)
                                  .count();
      trace_filter_time_ns_.fetch_add(elapsed, std::memory_order_relaxed);
      filter_time_ns_.fetch_add(elapsed, std::memory_order_relaxed);
      trace_checked_counter_++;
      if (evaluation.truth == TraceConstraintTruth::kUnknown) {
        trace_unknown_counter_++;
      } else {
        hard_known = true;
      }
      if (evaluation.truth == TraceConstraintTruth::kProvenInfeasible) {
        trace_predicted_reject_counter_++;
        hard_prediction = true;
        thread_predicted_reject_counter_++;
        size_t reason = HardRejectReasonIndex(evaluation);
        trace_predicted_reason_counters_[reason]++;
        thread_predicted_reason_counters_[reason]++;
        if (hard_filter_mode_ == kEnforce) {
          trace_enforced_reject_counter_++;
          thread_enforced_reject_counter_++;
          thread_enforced_reason_counters_[reason]++;
          return std::nullopt;
        }
      }
    }

    s_tir::Schedule sch{nullptr};
    bool sampling_shadow_known = false;
    bool sampling_shadow_prediction = false;
    int max_replay_attempts = sampling_mode_ == kEnforce ? kMaxConstraintReplayAttempts : 1;
    for (int attempt = 0; attempt < max_replay_attempts; ++attempt) {
      bool sampling_empty_domain = false;
      bool sampling_replaced_decision = false;
      int replay_instruction_index = 0;
      try {
        sch = s_tir::Schedule::Traced(mod,
                                      /*rand_state=*/ForkSeed(rand_state),
                                      /*debug_mode=*/0,
                                      /*error_render_level=*/
                                      s_tir::ScheduleErrorRenderLevel::kNone);
        s_tir::FTraceDecisionProvider decision_provider = nullptr;
        if (sampling_mode_ == kEnforce && plan_ready) {
          decision_provider =
              [&](const s_tir::Instruction& inst, const ffi::Array<Any>& inputs,
                  const ffi::Array<Any>& attrs, const Any& decision) -> Any {
            int instruction_index = replay_instruction_index++;
            if (decision != nullptr || sampling_empty_domain) return decision;
            int64_t loop_extent = -1;
            if (inst->kind->name == "SamplePerfectTile" && inputs.size() == 1) {
              if (auto loop_rv = inputs[0].try_cast<s_tir::LoopRV>()) {
                const int64_t* value = tirx::as_const_int(sch->Get(loop_rv.value())->extent);
                if (value != nullptr) loop_extent = *value;
              }
            }
            const auto sampling_start = std::chrono::steady_clock::now();
            TraceConstraintDecisionResult sampled = SampleConstrainedTraceDecision(
                plan, instruction_index, sch->trace().value(), loop_extent, rand_state);
            sampling_time_ns_.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - sampling_start)
                    .count(),
                std::memory_order_relaxed);
            if (!sampled.handled) return decision;
            sampling_handled_decision_counter_++;
            if (sampled.unknown) {
              sampling_decision_unknown_counter_++;
              return decision;
            }
            sampling_feasible_domain_total_.fetch_add(sampled.feasible_domain_size,
                                                       std::memory_order_relaxed);
            if (sampled.empty_domain) {
              sampling_empty_domain_counter_++;
              sampling_empty_domain = true;
              return decision;
            }
            sampling_constrained_decision_counter_++;
            sampling_replaced_decision = true;
            return sampled.decision;
          };
        }
        const auto replay_start = std::chrono::steady_clock::now();
        trace->ApplyToSchedule(sch, /*remove_postproc=*/true, decision_provider);
        trace_replay_time_ns_.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - replay_start)
                .count(),
            std::memory_order_relaxed);
      } catch (const s_tir::ScheduleError& e) {
        TVM_PY_LOG(WARNING, nullptr) << "Trace replay failed with ScheduleError: " << e.what();
        this->trace_fail_counter_++;
        return std::nullopt;
      } catch (const std::exception& e) {
        TVM_PY_LOG(WARNING, nullptr) << "Trace replay failed with exception: " << e.what();
        this->trace_fail_counter_++;
        return std::nullopt;
      }

      bool retry_sampling = sampling_empty_domain;
      if (!sampling_empty_domain && sampling_mode_ != kOff && plan_ready &&
          (sampling_mode_ == kShadow || sampling_replaced_decision)) {
        const auto check_start = std::chrono::steady_clock::now();
        TraceConstraintEvaluation evaluation =
            EvaluateTraceConstraints(plan, sch->trace().value());
        sampling_check_time_ns_.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - check_start)
                .count(),
            std::memory_order_relaxed);
        sampling_candidate_checked_counter_++;
        if (evaluation.truth == TraceConstraintTruth::kUnknown) {
          sampling_candidate_unknown_counter_++;
        } else {
          sampling_shadow_known = true;
        }
        if (evaluation.truth == TraceConstraintTruth::kProvenInfeasible) {
          sampling_predicted_reject_counter_++;
          sampling_shadow_prediction = true;
          if (sampling_mode_ == kEnforce && sampling_replaced_decision) retry_sampling = true;
        }
      }
      if (!retry_sampling) break;
      if (attempt + 1 < max_replay_attempts) {
        sampling_retry_counter_++;
        continue;
      }
      sampling_abandoned_counter_++;
      return std::nullopt;
    }

    bool shared_proxy_checked = false;
    bool shared_proxy_prediction = false;
    if (hard_filter_mode_ != kOff &&
        (hard_filter_position_ == kPostReplay || hard_filter_position_ == kTrace)) {
      if (hard_filter_position_ == kPostReplay) thread_checked_counter_++;
      const auto filter_start = std::chrono::steady_clock::now();
      ThreadFeasibilityResult feasibility = AnalyzeThreadFeasibility(
          sch->mod(), max_threads_per_block_, thread_warp_size_, max_shared_memory_per_block_);
      const int64_t elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - filter_start)
                                  .count();
      postreplay_filter_time_ns_.fetch_add(elapsed, std::memory_order_relaxed);
      filter_time_ns_.fetch_add(elapsed, std::memory_order_relaxed);
      postreplay_checked_counter_++;
      bool postreplay_known = feasibility.reject ||
                              (feasibility.all_extents_known && feasibility.all_contracts_known);
      if (!postreplay_known) postreplay_unknown_counter_++;
      hard_known = hard_known || postreplay_known;
      if (feasibility.reject) {
        postreplay_predicted_reject_counter_++;
        size_t reason = HardRejectReasonIndex(feasibility);
        postreplay_predicted_reason_counters_[reason]++;
        if (!hard_prediction) {
          hard_prediction = true;
          thread_predicted_reject_counter_++;
          thread_predicted_reason_counters_[reason]++;
        }
        if (hard_filter_mode_ == kEnforce) {
          postreplay_enforced_reject_counter_++;
          thread_enforced_reject_counter_++;
          thread_enforced_reason_counters_[reason]++;
          return std::nullopt;
        }
      }
      shared_proxy_checked = true;
      shared_proxy_prediction = feasibility.shared_memory_proxy_reject;
      RecordSharedProxyObservation(feasibility);
      if (!hard_known) thread_unknown_counter_++;
    }

    try {
      sch->EnterPostproc();
    } catch (const s_tir::ScheduleError& e) {
      TVM_PY_LOG(WARNING, nullptr) << "Entering postproc failed with ScheduleError: " << e.what();
      this->trace_fail_counter_++;
      return std::nullopt;
    } catch (const std::exception& e) {
      TVM_PY_LOG(WARNING, nullptr) << "Entering postproc failed with exception: " << e.what();
      this->trace_fail_counter_++;
      return std::nullopt;
    }

    for (int i = 0; i < n_; ++i) {
      Item& item = items_[i];
      if (i == verify_gpu_index_ && hard_filter_mode_ != kOff &&
          hard_filter_position_ == kLate) {
        thread_checked_counter_++;
        const auto filter_start = std::chrono::steady_clock::now();
        ThreadFeasibilityResult feasibility = AnalyzeThreadFeasibility(
            sch->mod(), max_threads_per_block_, thread_warp_size_, max_shared_memory_per_block_);
        const int64_t elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - filter_start)
                                    .count();
        late_filter_time_ns_.fetch_add(elapsed, std::memory_order_relaxed);
        filter_time_ns_.fetch_add(elapsed, std::memory_order_relaxed);
        late_checked_counter_++;
        hard_known = feasibility.reject ||
                     (feasibility.all_extents_known && feasibility.all_contracts_known);
        if (!hard_known) {
          late_unknown_counter_++;
          thread_unknown_counter_++;
        }
        if (feasibility.reject) {
          hard_prediction = true;
          late_predicted_reject_counter_++;
          thread_predicted_reject_counter_++;
          size_t reason = HardRejectReasonIndex(feasibility);
          late_predicted_reason_counters_[reason]++;
          thread_predicted_reason_counters_[reason]++;
        }
        shared_proxy_checked = true;
        shared_proxy_prediction = feasibility.shared_memory_proxy_reject;
        RecordSharedProxyObservation(feasibility);
        if (hard_filter_mode_ == kEnforce && feasibility.reject) {
          late_enforced_reject_counter_++;
          thread_enforced_reject_counter_++;
          thread_enforced_reason_counters_[HardRejectReasonIndex(feasibility)]++;
          return std::nullopt;
        }
      }

      bool success = true;
      const auto postproc_start = std::chrono::steady_clock::now();
      try {
        if (!item.postproc->Apply(sch)) {
          success = false;
        }
      } catch (const s_tir::ScheduleError& e) {
        DLOG(WARNING) << "Postproc #" << i << " failed with ScheduleError: " << e.what();
        success = false;
      } catch (const std::exception& e) {
        DLOG(WARNING) << "Postproc #" << i << " failed with exception: " << e.what();
        success = false;
      }
      const int64_t postproc_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - postproc_start)
                                           .count();
      item.time_ns.fetch_add(postproc_elapsed, std::memory_order_relaxed);
      if (i == verify_gpu_index_)
        verify_gpu_time_ns_.fetch_add(postproc_elapsed, std::memory_order_relaxed);
      VerifyGPUCodeFailureReason verify_reason = VerifyGPUCodeFailureReason::kNone;
      if (i == verify_gpu_index_ && !success) {
        verify_reason = GetLastVerifyGPUCodeFailureReason();
        if (verify_reason == VerifyGPUCodeFailureReason::kNone) {
          verify_reason = VerifyGPUCodeFailureReason::kUnknown;
        }
      }
      if (i == verify_gpu_index_) {
        bool thread_failed = !success &&
                             (verify_reason == VerifyGPUCodeFailureReason::kThreadExtent ||
                              verify_reason == VerifyGPUCodeFailureReason::kThreadsPerBlock);
        if (hard_filter_mode_ == kShadow && hard_known) {
          UpdateConfusion(hard_prediction, thread_failed, &shadow_tp_counter_,
                          &shadow_fp_counter_, &shadow_fn_counter_, &shadow_tn_counter_);
        }
        if (sampling_mode_ == kShadow && sampling_shadow_known) {
          UpdateConfusion(sampling_shadow_prediction, thread_failed, &sampling_shadow_tp_counter_,
                          &sampling_shadow_fp_counter_, &sampling_shadow_fn_counter_,
                          &sampling_shadow_tn_counter_);
        }
      }
      if (i == verify_gpu_index_ && shared_proxy_checked) {
        bool shared_memory_failed = verify_reason == VerifyGPUCodeFailureReason::kSharedMemory;
        if (shared_proxy_prediction) {
          if (shared_memory_failed) {
            shared_memory_proxy_tp_counter_++;
          } else {
            shared_memory_proxy_fp_counter_++;
          }
        } else if (shared_memory_failed) {
          shared_memory_proxy_fn_counter_++;
        } else {
          shared_memory_proxy_tn_counter_++;
        }
      }
      if (!success) {
        if (i == verify_gpu_index_) {
          verify_failure_counters_[static_cast<size_t>(verify_reason)]++;
        }
        item.fail_counter++;
        return std::nullopt;
      }
    }
    passed_counter_++;
    return sch;
  }

  /*! \brief Returns a string summarizing the failures on each postprocessor */
  std::string SummarizeFailures() const {
    std::ostringstream os;
    int64_t postproc_failures = 0;
    for (int i = 0; i < n_; ++i) {
      postproc_failures += items_[i].fail_counter.load();
    }
    int64_t accounted = passed_counter_.load() + trace_fail_counter_.load() + postproc_failures +
                        thread_enforced_reject_counter_.load() +
                        sampling_abandoned_counter_.load();
    int64_t conservation_delta = generated_counter_.load() - accounted;
    os << "stage.generated: " << generated_counter_.load()
       << "\nstage.trace_replay_failed: " << trace_fail_counter_.load()
       << "\nstage.constraint_sampling_abandoned: " << sampling_abandoned_counter_.load();
    for (int i = 0; i < n_; ++i) {
      os << "\nstage.postproc_failed[" << i << "]: " << items_[i].fail_counter.load();
    }
    for (int i = 1; i < static_cast<int>(VerifyGPUCodeFailureReason::kCount); ++i) {
      auto reason = static_cast<VerifyGPUCodeFailureReason>(i);
      os << "\nstage.verify_failed[" << VerifyGPUCodeFailureReasonName(reason)
         << "]: " << verify_failure_counters_[i].load();
    }
    os << "\nstage.hard_rejected[thread_extent_low]: "
       << thread_enforced_reason_counters_[0].load()
       << "\nstage.hard_rejected[thread_extent_high]: "
       << thread_enforced_reason_counters_[1].load()
       << "\nstage.hard_rejected[threads_per_block]: "
       << thread_enforced_reason_counters_[2].load() << "\nstage.passed: "
       << passed_counter_.load()
       << "\nstage.accounted: " << accounted
       << "\nstage.conservation_delta: " << conservation_delta
       << "\nTrace replay failures: " << this->trace_fail_counter_.load() << " failure(s)"
       << "\nConstraint sampling mode: " << ModeName(sampling_mode_)
       << "\nConstraint plans built: " << plan_built_counter_.load()
       << "\nConstraint plan contract blocks: " << plan_contract_blocks_counter_.load()
       << "\nConstraint plan enforceable constraints: " << plan_enforceable_counter_.load()
       << "\nConstraint sampling decisions handled: "
       << sampling_handled_decision_counter_.load()
       << "\nConstraint sampling decisions replaced: "
       << sampling_constrained_decision_counter_.load()
       << "\nConstraint sampling decision unknown: "
       << sampling_decision_unknown_counter_.load()
       << "\nConstraint sampling feasible domain total: "
       << sampling_feasible_domain_total_.load()
       << "\nConstraint sampling empty domains: " << sampling_empty_domain_counter_.load()
       << "\nConstraint sampling retries: " << sampling_retry_counter_.load()
       << "\nConstraint sampling abandoned: " << sampling_abandoned_counter_.load()
       << "\nConstraint sampling candidates checked: "
       << sampling_candidate_checked_counter_.load()
       << "\nConstraint sampling candidates unknown: "
       << sampling_candidate_unknown_counter_.load()
       << "\nConstraint sampling predicted rejections: "
       << sampling_predicted_reject_counter_.load()
       << "\nConstraint sampling shadow TP/FP/FN/TN (thread-only oracle): "
       << sampling_shadow_tp_counter_.load() << "/" << sampling_shadow_fp_counter_.load() << "/"
       << sampling_shadow_fn_counter_.load() << "/" << sampling_shadow_tn_counter_.load()
       << "\nConstraint plan time (us): " << plan_time_ns_.load() / 1000
       << "\nConstraint decision sampling time (us): " << sampling_time_ns_.load() / 1000
       << "\nConstraint sampling check time (us): " << sampling_check_time_ns_.load() / 1000
       << "\nTrace replay time (us): " << trace_replay_time_ns_.load() / 1000
       << "\nThread filter mode: " << ModeName(hard_filter_mode_)
       << "\nThread filter position: " << PositionName(hard_filter_position_)
       << "\nThread filter checked: " << this->thread_checked_counter_.load()
       << "\nThread filter unknown: " << this->thread_unknown_counter_.load()
       << "\nThread filter predicted rejections: "
       << this->thread_predicted_reject_counter_.load()
       << "\nThread filter enforced rejections: " << this->thread_enforced_reject_counter_.load()
       << "\nThread filter shadow TP/FP/FN/TN: " << this->shadow_tp_counter_.load() << "/"
       << this->shadow_fp_counter_.load() << "/" << this->shadow_fn_counter_.load() << "/"
       << this->shadow_tn_counter_.load()
       << "\nHard filter predicted by reason low/high/hardware: "
       << this->thread_predicted_reason_counters_[0].load() << "/"
       << this->thread_predicted_reason_counters_[1].load() << "/"
       << this->thread_predicted_reason_counters_[2].load()
       << "\nShared-memory pre-compaction proxy checked: "
       << this->shared_memory_proxy_checked_counter_.load()
       << "\nShared-memory pre-compaction proxy unknown: "
       << this->shared_memory_proxy_unknown_counter_.load()
       << "\nShared-memory pre-compaction proxy predicted rejections: "
       << this->shared_memory_proxy_predicted_reject_counter_.load()
       << "\nShared-memory pre-compaction proxy TP/FP/FN/TN: "
       << this->shared_memory_proxy_tp_counter_.load() << "/"
       << this->shared_memory_proxy_fp_counter_.load() << "/"
       << this->shared_memory_proxy_fn_counter_.load() << "/"
       << this->shared_memory_proxy_tn_counter_.load()
       << "\nH-trace checked/unknown/predicted/enforced: " << trace_checked_counter_.load() << "/"
       << trace_unknown_counter_.load() << "/" << trace_predicted_reject_counter_.load() << "/"
       << trace_enforced_reject_counter_.load()
       << "\nH-postreplay checked/unknown/predicted/enforced: "
       << postreplay_checked_counter_.load() << "/" << postreplay_unknown_counter_.load() << "/"
       << postreplay_predicted_reject_counter_.load() << "/"
       << postreplay_enforced_reject_counter_.load()
       << "\nH-late checked/unknown/predicted/enforced: " << late_checked_counter_.load() << "/"
       << late_unknown_counter_.load() << "/" << late_predicted_reject_counter_.load() << "/"
       << late_enforced_reject_counter_.load()
       << "\nH-trace time (us): " << trace_filter_time_ns_.load() / 1000
       << "\nH-postreplay time (us): " << postreplay_filter_time_ns_.load() / 1000
       << "\nH-late time (us): " << late_filter_time_ns_.load() / 1000
       << "\nThread filter analysis time (us): " << this->filter_time_ns_.load() / 1000
       << "\nVerifyGPUCode time (us): " << this->verify_gpu_time_ns_.load() / 1000;
    if (n_ > 0) {
      os << "\n";
    }
    for (int i = 0; i < n_; ++i) {
      const Item& item = items_[i];
      os << "Postproc #" << i << " [" << item.postproc  //
         << "]: " << item.fail_counter.load() << " failure(s), time (us): "
         << item.time_ns.load() / 1000;
      if (i != n_ - 1) {
        os << "\n";
      }
    }
    return os.str();
  }

  /*! \brief Returns the number of trace replay failures. */
  int TraceFailCount() const { return this->trace_fail_counter_.load(); }

  /*! \brief Returns the number of candidates rejected by the enforced thread filter. */
  int PruneFailCount() const { return this->thread_enforced_reject_counter_.load(); }

 private:
  /*! \brief A helper data structure that stores the fail count for each postprocessor. */
  struct Item {
    /*! \brief The postprocessor. */
    Postproc postproc{ffi::UnsafeInit()};
    /*! \brief The thread-safe postprocessor failure counter. */
    std::atomic<int> fail_counter{0};
    /*! \brief Total wall-clock time spent in this postprocessor. */
    std::atomic<int64_t> time_ns{0};
  };

  static constexpr int kOff = 0;
  static constexpr int kShadow = 1;
  static constexpr int kEnforce = 2;
  static constexpr int kLate = 0;
  static constexpr int kPostReplay = 1;
  static constexpr int kTrace = 2;
  static constexpr int kMaxConstraintReplayAttempts = 8;

  /*! \brief Read off/shadow/enforce, while accepting the legacy numeric spelling. */
  static int ReadMode(const char* name, int default_value) {
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0') return default_value;
    std::string value(env);
    if (value == "off") return kOff;
    if (value == "shadow") return kShadow;
    if (value == "enforce") return kEnforce;
    return std::clamp(std::atoi(env), kOff, kEnforce);
  }

  /*! \brief New hard-filter variable takes priority over TVM_PRUNE_MODE. */
  static int ReadHardFilterMode() {
    if (std::getenv("TVM_THREAD_HARD_FILTER_MODE") != nullptr) {
      return ReadMode("TVM_THREAD_HARD_FILTER_MODE", kOff);
    }
    return ReadMode("TVM_PRUNE_MODE", kOff);
  }

  static int ReadHardFilterPosition() {
    const char* env = std::getenv("TVM_THREAD_HARD_FILTER_POSITION");
    if (env == nullptr || *env == '\0') return kLate;
    std::string value(env);
    if (value == "postreplay") return kPostReplay;
    if (value == "trace") return kTrace;
    return kLate;
  }

  static const char* ModeName(int mode) {
    return mode == kShadow ? "shadow" : (mode == kEnforce ? "enforce" : "off");
  }

  static const char* PositionName(int position) {
    return position == kTrace ? "trace" : (position == kPostReplay ? "postreplay" : "late");
  }

  /*! \brief Return a stable primary-reason index: low, high, then hardware. */
  static size_t HardRejectReasonIndex(const ThreadFeasibilityResult& result) {
    if (result.reject_thread_extent_low) return 0;
    if (result.reject_thread_extent_high) return 1;
    return 2;
  }

  static size_t HardRejectReasonIndex(const TraceConstraintEvaluation& result) {
    if (result.reject_thread_extent_low) return 0;
    if (result.reject_thread_extent_high) return 1;
    return 2;
  }

  static void UpdateConfusion(bool predicted, bool actual, std::atomic<int64_t>* tp,
                              std::atomic<int64_t>* fp, std::atomic<int64_t>* fn,
                              std::atomic<int64_t>* tn) {
    if (predicted) {
      (actual ? *tp : *fp)++;
    } else {
      (actual ? *fn : *tn)++;
    }
  }

  void RecordSharedProxyObservation(const ThreadFeasibilityResult& feasibility) {
    shared_memory_proxy_checked_counter_++;
    if (!feasibility.all_shared_allocations_known) shared_memory_proxy_unknown_counter_++;
    if (feasibility.shared_memory_proxy_reject) shared_memory_proxy_predicted_reject_counter_++;
  }

  /*! \brief The number of total postprocessors. */
  int n_;
  /*! \brief The thread-safe trace replay failure counter. */
  std::atomic<int> trace_fail_counter_{0};
  /*! \brief Candidate accounting for the replay/postprocessing pipeline. */
  std::atomic<int64_t> generated_counter_{0};
  std::atomic<int64_t> passed_counter_{0};
  /*! \brief The pointer to the list of postprocessor items. */
  Item* items_;
  /*! \brief Cached modes for constraint-aware sampling and hard filtering. */
  int sampling_mode_{kOff};
  int hard_filter_mode_{kOff};
  int hard_filter_position_{kLate};
  int verify_gpu_index_{-1};
  int64_t max_threads_per_block_{-1};
  int64_t thread_warp_size_{-1};
  int64_t max_shared_memory_per_block_{-1};
  /*! \brief Constraint-plan and initialization-sampling counters. */
  std::atomic<int64_t> plan_built_counter_{0};
  std::atomic<int64_t> plan_contract_blocks_counter_{0};
  std::atomic<int64_t> plan_enforceable_counter_{0};
  std::atomic<int64_t> sampling_handled_decision_counter_{0};
  std::atomic<int64_t> sampling_constrained_decision_counter_{0};
  std::atomic<int64_t> sampling_decision_unknown_counter_{0};
  std::atomic<int64_t> sampling_feasible_domain_total_{0};
  std::atomic<int64_t> sampling_empty_domain_counter_{0};
  std::atomic<int64_t> sampling_retry_counter_{0};
  std::atomic<int64_t> sampling_abandoned_counter_{0};
  std::atomic<int64_t> sampling_candidate_checked_counter_{0};
  std::atomic<int64_t> sampling_candidate_unknown_counter_{0};
  std::atomic<int64_t> sampling_predicted_reject_counter_{0};
  std::atomic<int64_t> sampling_shadow_tp_counter_{0};
  std::atomic<int64_t> sampling_shadow_fp_counter_{0};
  std::atomic<int64_t> sampling_shadow_fn_counter_{0};
  std::atomic<int64_t> sampling_shadow_tn_counter_{0};
  std::atomic<int64_t> plan_time_ns_{0};
  std::atomic<int64_t> sampling_time_ns_{0};
  std::atomic<int64_t> sampling_check_time_ns_{0};
  std::atomic<int64_t> trace_replay_time_ns_{0};
  /*! \brief Thread-safe filter counters. */
  std::atomic<int64_t> thread_checked_counter_{0};
  std::atomic<int64_t> thread_unknown_counter_{0};
  std::atomic<int64_t> thread_predicted_reject_counter_{0};
  std::atomic<int64_t> thread_enforced_reject_counter_{0};
  std::array<std::atomic<int64_t>, 3> thread_predicted_reason_counters_{};
  std::array<std::atomic<int64_t>, 3> thread_enforced_reason_counters_{};
  std::atomic<int64_t> trace_checked_counter_{0};
  std::atomic<int64_t> trace_unknown_counter_{0};
  std::atomic<int64_t> trace_predicted_reject_counter_{0};
  std::atomic<int64_t> trace_enforced_reject_counter_{0};
  std::array<std::atomic<int64_t>, 3> trace_predicted_reason_counters_{};
  std::atomic<int64_t> postreplay_checked_counter_{0};
  std::atomic<int64_t> postreplay_unknown_counter_{0};
  std::atomic<int64_t> postreplay_predicted_reject_counter_{0};
  std::atomic<int64_t> postreplay_enforced_reject_counter_{0};
  std::array<std::atomic<int64_t>, 3> postreplay_predicted_reason_counters_{};
  std::atomic<int64_t> late_checked_counter_{0};
  std::atomic<int64_t> late_unknown_counter_{0};
  std::atomic<int64_t> late_predicted_reject_counter_{0};
  std::atomic<int64_t> late_enforced_reject_counter_{0};
  std::array<std::atomic<int64_t>, 3> late_predicted_reason_counters_{};
  std::atomic<int64_t> shadow_tp_counter_{0};
  std::atomic<int64_t> shadow_fp_counter_{0};
  std::atomic<int64_t> shadow_fn_counter_{0};
  std::atomic<int64_t> shadow_tn_counter_{0};
  std::atomic<int64_t> shared_memory_proxy_checked_counter_{0};
  std::atomic<int64_t> shared_memory_proxy_unknown_counter_{0};
  std::atomic<int64_t> shared_memory_proxy_predicted_reject_counter_{0};
  std::atomic<int64_t> shared_memory_proxy_tp_counter_{0};
  std::atomic<int64_t> shared_memory_proxy_fp_counter_{0};
  std::atomic<int64_t> shared_memory_proxy_fn_counter_{0};
  std::atomic<int64_t> shared_memory_proxy_tn_counter_{0};
  std::atomic<int64_t> trace_filter_time_ns_{0};
  std::atomic<int64_t> postreplay_filter_time_ns_{0};
  std::atomic<int64_t> late_filter_time_ns_{0};
  std::atomic<int64_t> filter_time_ns_{0};
  std::atomic<int64_t> verify_gpu_time_ns_{0};
  /*! \brief Candidate-level primary VerifyGPUCode failure counters. */
  std::array<std::atomic<int64_t>, static_cast<size_t>(VerifyGPUCodeFailureReason::kCount)>
      verify_failure_counters_{};
};

/*!
 * \brief Get the number of cores in CPU
 * \param target The target
 * \return The number of cores.
 */
inline int GetTargetNumCores(const Target& target) {
  int num_cores = target->GetAttr<int64_t>("num-cores").value_or(-1);
  if (num_cores == -1) {
    static const auto f_cpu_count = tvm::ffi::Function::GetGlobal("s_tir.meta_schedule.cpu_count");
    TVM_FFI_CHECK(f_cpu_count.has_value(), ValueError)
        << "Cannot find the packed function \"s_tir.meta_schedule._cpu_count\"";
    num_cores = (*f_cpu_count)(false).cast<int>();
    TVM_FFI_THROW(InternalError)
        << "Target does not have attribute \"num-cores\", physical core number must be "
           "defined! For example, on the local machine, the target must be \"llvm -num-cores "
        << num_cores << "\"";
  }
  return num_cores;
}

/*!
 * \brief Get the median of the running time from RunnerResult in millisecond
 * \param results The results from RunnerResult
 * \return The median of the running time in millisecond
 */
inline double GetRunMsMedian(const RunnerResult& runner_result) {
  ffi::Array<FloatImm> run_secs = runner_result->run_secs.value();
  TVM_FFI_ICHECK(!run_secs.empty());
  std::vector<double> v;
  v.reserve(run_secs.size());
  std::transform(run_secs.begin(), run_secs.end(), std::back_inserter(v),
                 [](const FloatImm& f) -> double { return f->value; });
  std::sort(v.begin(), v.end());
  int n = v.size();
  if (n % 2 == 0) {
    return (v[n / 2 - 1] + v[n / 2]) * 0.5 * 1000.0;
  } else {
    return v[n / 2] * 1000.0;
  }
}

/*!
 * \brief Convert the given object to an array of floating point numbers
 * \param obj The object to be converted
 * \return The array of floating point numbers
 */
inline ffi::Array<FloatImm> AsFloatArray(const ffi::ObjectRef& obj) {
  const ffi::ArrayObj* arr = obj.as<ffi::ArrayObj>();
  TVM_FFI_CHECK(arr, TypeError) << "Expect an array, but gets: " << obj->GetTypeKey();
  ffi::Array<FloatImm> results;
  results.reserve(arr->size());
  for (Any val : *arr) {
    auto float_value = [&]() -> FloatImm {
      if (auto opt_int_imm = val.try_cast<IntImm>()) {
        return FloatImm(DataType::Float(32), (*opt_int_imm)->value);
      } else if (auto opt_float_imm = val.try_cast<FloatImm>()) {
        return *std::move(opt_float_imm);
      } else {
        TVM_FFI_THROW(TypeError) << "Expect an array of float or int, but gets: "
                                 << val.GetTypeKey();
        TVM_FFI_UNREACHABLE();
      }
    }();

    results.push_back(float_value);
  }
  return results;
}

/*!
 * \brief Convert the given object to an array of integers
 * \param obj The object to be converted
 * \return The array of integers
 */
inline ffi::Array<int64_t> AsIntArray(const ffi::ObjectRef& obj) {
  const ffi::ArrayObj* arr = obj.as<ffi::ArrayObj>();
  TVM_FFI_CHECK(arr, TypeError) << "Expect an array, but gets: " << obj->GetTypeKey();
  ffi::Array<int64_t> results;
  results.reserve(arr->size());
  for (Any val : *arr) {
    auto int_value = [&]() -> int64_t {
      if (auto opt_int_imm = val.try_cast<IntImm>()) {
        return (*opt_int_imm)->value;
      } else {
        TVM_FFI_THROW(TypeError) << "Expect an array of integers, but gets: " << val.GetTypeKey();
        TVM_FFI_UNREACHABLE();
      }
    }();
    results.push_back(int_value);
  }
  return results;
}

/*! \brief The struct defining comparison function of sorting by mean run seconds. */
struct SortTuningRecordByMeanRunSecs {
  static const constexpr double kMaxMeanTime = 1e10;

  static double Mean(const ffi::Array<FloatImm>& a) {
    if (a.empty()) {
      return kMaxMeanTime;
    }
    double sum = 0.0;
    for (const FloatImm& i : a) {
      sum += i->value;
    }
    return sum / a.size();
  }

  bool operator()(const TuningRecord& a, const TuningRecord& b) const {
    double a_time = Mean(a->run_secs.value_or({}));
    double b_time = Mean(b->run_secs.value_or({}));
    return a_time < b_time;
  }
};

/*!
 * \brief The helper function to clone schedule rules, postprocessors, and mutators.
 * \param src The source space generator.
 * \param dst The destination space generator.
 */
inline void CloneRules(const SpaceGeneratorNode* src, SpaceGeneratorNode* dst) {
  if (src->sch_rules.defined()) {
    ffi::Array<ScheduleRule> original = src->sch_rules.value();
    ffi::Array<ScheduleRule> sch_rules;
    sch_rules.reserve(original.size());
    for (const ScheduleRule& sch_rule : original) {
      sch_rules.push_back(sch_rule->Clone());
    }
    dst->sch_rules = std::move(sch_rules);
  }
  if (src->postprocs.defined()) {
    ffi::Array<Postproc> original = src->postprocs.value();
    ffi::Array<Postproc> postprocs;
    postprocs.reserve(original.size());
    for (const Postproc& postproc : original) {
      postprocs.push_back(postproc->Clone());
    }
    dst->postprocs = std::move(postprocs);
  }
  if (src->mutator_probs.defined()) {
    ffi::Map<Mutator, FloatImm> original = src->mutator_probs.value();
    ffi::Map<Mutator, FloatImm> mutator_probs;
    for (const auto& kv : original) {
      mutator_probs.Set(kv.first->Clone(), kv.second);
    }
    dst->mutator_probs = std::move(mutator_probs);
  }
}

/*! \brief Returns true if the given target is one of the supported gpu targets. */
inline bool IsGPUTarget(const std::string& target_name) {
  static const std::unordered_set<std::string> gpu_targets{"cuda", "rocm", "vulkan", "metal",
                                                           "opencl"};
  return gpu_targets.count(target_name);
}

/*!
 * \brief Create an AutoInline schedule rule for the given target.
 * \param target_name The name of the target ("llvm", "cuda", etc.)
 * \return The AutoInline schedule rule for the given target.
 */
inline ScheduleRule GetDefaultAutoInline(const std::string& target_name) {
  ffi::Array<ScheduleRule> rules{nullptr};
  if (target_name == "llvm") {
    rules = ScheduleRule::DefaultLLVM();
  } else if (target_name == "hexagon") {
    rules = ScheduleRule::DefaultHexagon();
  } else if (IsGPUTarget(target_name)) {
    rules = ScheduleRule::DefaultCUDA();
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported target: " << target_name;
  }
  for (const ScheduleRule& rule : rules) {
    if (rule->GetTypeKey() == "s_tir.meta_schedule.AutoInline") {
      return rule;
    }
  }
  TVM_FFI_THROW(ValueError) << "AutoInline rule is not found in the default rules for target: "
                            << target_name;
  throw;
}

/*!
 * \brief Summarize the run time of the given FloatImm array.
 * \param arr The array of FloatImm.
 * \return The summary of the values in the given array.
 */
inline double Sum(const ffi::Array<FloatImm>& arr) {
  double sum = 0;
  for (const FloatImm& f : arr) {
    sum += f->value;
  }
  return sum;
}

/*! \brief Collecting all the blocks */
class SBlockCollector : public tirx::StmtVisitor {
 public:
  static ffi::Array<s_tir::SBlockRV> Collect(const s_tir::Schedule& sch,
                                             const ffi::Function f_block_filter = nullptr) {  //
    return SBlockCollector(sch, f_block_filter).Run();
  }

 private:
  /*! \brief Entry point */
  ffi::Array<s_tir::SBlockRV> Run() {
    std::vector<s_tir::SBlockRV> results;
    auto f_collect = [this, &results](tirx::PrimFunc func, ffi::String func_name) {
      func_name_ = func_name;
      block_names_.clear();
      blocks_to_collect_.clear();
      VisitStmt(func->body);
      for (const ffi::String& name : blocks_to_collect_) {
        results.push_back(sch_->GetSBlock(name, func_name_));
      }
    };

    if (sch_->func_working_on().defined()) {
      GlobalVar gv = sch_->func_working_on().value();
      tirx::PrimFunc func = Downcast<tirx::PrimFunc>(sch_->mod()->functions[gv]);
      f_collect(func, gv->name_hint);
    } else {
      for (const auto& [gv, base_func] : sch_->mod()->functions) {
        // `gv->name_hint` is the name of the function
        // `base_func` can be PrimFunc or relax::Function
        if (const auto* func = base_func.as<tirx::PrimFuncNode>()) {
          f_collect(ffi::GetRef<tirx::PrimFunc>(func), gv->name_hint);
        }
      }
    }
    return results;
  }
  /*! \brief Constructor */
  explicit SBlockCollector(const s_tir::Schedule& sch, const ffi::Function f_block_filter = nullptr)
      : sch_(sch), f_block_filter_(f_block_filter) {}
  /*! \brief Override the Stmt visiting behaviour */
  void VisitStmt_(const tirx::SBlockNode* block) override {
    tirx::StmtVisitor::VisitStmt_(block);
    TVM_FFI_ICHECK(block_names_.count(block->name_hint) == 0)
        << "Duplicated block name " << block->name_hint << " in function " << func_name_
        << " not supported!";
    block_names_.insert(block->name_hint);

    // If filter function is provided, use it to selectively collect blocks.
    // Otherwise collect all blocks.
    bool collect_block = true;
    if (f_block_filter_ != nullptr) {
      collect_block = f_block_filter_(ffi::GetRef<tirx::SBlock>(block)).cast<IntImm>()->value != 0;
    }
    if (collect_block) {
      blocks_to_collect_.push_back(block->name_hint);
    }
  }

  /*! \brief The schedule to be collected */
  const s_tir::Schedule& sch_;
  /*! \brief An optional packed func that allows only certain blocks to be collected. */
  const ffi::Function f_block_filter_;
  /*! \brief The set of func name and block name pair */
  std::unordered_set<ffi::String> block_names_;
  /* \brief The list of blocks to collect in order */
  ffi::Array<ffi::String> blocks_to_collect_;
  /*! \brief Name of the current PrimFunc */
  ffi::String func_name_;
};

void JSONFileAppendLine(const ffi::String& path, const std::string& line);
std::vector<Any> JSONFileReadLines(const ffi::String& path, int num_threads, bool allow_missing);
}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_UTILS_H_

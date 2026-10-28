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

#include "draft_mutator_credit.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

MeasuredMutatorCreditBook::MeasuredMutatorCreditBook(int num_mutators, int window_size,
                                                     int cold_start_measurements,
                                                     double exploration_scale)
    : window_size_(window_size),
      cold_start_measurements_(cold_start_measurements),
      exploration_scale_(exploration_scale) {
  TVM_FFI_CHECK_GT(num_mutators, 0, ValueError);
  TVM_FFI_CHECK_GT(window_size, 0, ValueError);
  TVM_FFI_CHECK_GE(cold_start_measurements, 0, ValueError);
  TVM_FFI_CHECK(std::isfinite(exploration_scale) && exploration_scale >= 0.0, ValueError)
      << "exploration_scale must be finite and non-negative, but got " << exploration_scale;
  arms_.resize(num_mutators);
}

void MeasuredMutatorCreditBook::CheckMutatorIndex(int mutator_index) const {
  TVM_FFI_CHECK(0 <= mutator_index && mutator_index < num_mutators(), ValueError)
      << "mutator_index must be within [0, " << num_mutators() << "), but got "
      << mutator_index;
}

void MeasuredMutatorCreditBook::RecordMeasuredOutcome(int mutator_index,
                                                      double measured_parent_latency_sec,
                                                      double measured_child_latency_sec) {
  CheckMutatorIndex(mutator_index);
  TVM_FFI_CHECK(std::isfinite(measured_parent_latency_sec) &&
                    measured_parent_latency_sec > 0.0,
                ValueError)
      << "measured_parent_latency_sec must be finite and positive";
  TVM_FFI_CHECK(std::isfinite(measured_child_latency_sec) && measured_child_latency_sec > 0.0,
                ValueError)
      << "measured_child_latency_sec must be finite and positive";

  Arm& arm = arms_[mutator_index];
  uint8_t improved = measured_child_latency_sec < measured_parent_latency_sec ? 1 : 0;
  arm.outcomes.push_back(improved);
  arm.improvements += improved;
  if (static_cast<int>(arm.outcomes.size()) > window_size_) {
    arm.improvements -= arm.outcomes.front();
    arm.outcomes.pop_front();
  }
  ++total_measured_outcomes_;
}

bool MeasuredMutatorCreditBook::InColdStart() const {
  return total_measured_outcomes_ < cold_start_measurements_;
}

std::vector<double> MeasuredMutatorCreditBook::UCB1Scores() const {
  int total_window_outcomes = std::accumulate(
      arms_.begin(), arms_.end(), 0,
      [](int total, const Arm& arm) { return total + static_cast<int>(arm.outcomes.size()); });
  double log_total = std::log(static_cast<double>(std::max(1, total_window_outcomes)));

  std::vector<double> scores;
  scores.reserve(arms_.size());
  for (const Arm& arm : arms_) {
    int count = static_cast<int>(arm.outcomes.size());
    if (count == 0) {
      scores.push_back(std::numeric_limits<double>::infinity());
      continue;
    }
    double mean_reward = static_cast<double>(arm.improvements) / count;
    double exploration =
        exploration_scale_ * std::sqrt(2.0 * log_total / static_cast<double>(count));
    scores.push_back(mean_reward + exploration);
  }
  return scores;
}

int MeasuredMutatorCreditBook::SelectMutator(int cold_start_choice) const {
  return SelectMutator(cold_start_choice, {});
}

int MeasuredMutatorCreditBook::SelectMutator(
    int cold_start_choice, const std::vector<uint8_t>& unavailable) const {
  TVM_FFI_CHECK(unavailable.empty() || unavailable.size() == arms_.size(), ValueError)
      << "unavailable must be empty or contain one entry per mutator";
  auto is_available = [&unavailable](int index) {
    return unavailable.empty() || unavailable[index] == 0;
  };

  if (InColdStart()) {
    CheckMutatorIndex(cold_start_choice);
    if (is_available(cold_start_choice)) return cold_start_choice;
    // A failed baseline arm is retried without replacement.  Cycling from the
    // caller's choice is deterministic and consumes no additional RNG.
    for (int offset = 1; offset < num_mutators(); ++offset) {
      int candidate = (cold_start_choice + offset) % num_mutators();
      if (is_available(candidate)) return candidate;
    }
    return -1;
  }

  std::vector<double> scores = UCB1Scores();
  int selected = -1;
  for (int index = 0; index < num_mutators(); ++index) {
    if (!is_available(index)) continue;
    if (selected == -1 || scores[index] > scores[selected]) selected = index;
  }
  return selected;
}

MeasuredMutatorCreditStats MeasuredMutatorCreditBook::Stats(int mutator_index) const {
  CheckMutatorIndex(mutator_index);
  const Arm& arm = arms_[mutator_index];
  int count = static_cast<int>(arm.outcomes.size());
  return MeasuredMutatorCreditStats{
      count, arm.improvements,
      count == 0 ? 0.0 : static_cast<double>(arm.improvements) / static_cast<double>(count)};
}

namespace {

class TestingMeasuredMutatorCreditBookNode : public ffi::Object {
 public:
  TestingMeasuredMutatorCreditBookNode(int num_mutators, int window_size,
                                       int cold_start_measurements, double exploration_scale)
      : book(num_mutators, window_size, cold_start_measurements, exploration_scale) {}

  MeasuredMutatorCreditBook book;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TestingMeasuredMutatorCreditBookNode>();
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.testing.MeasuredMutatorCreditBook",
                                    TestingMeasuredMutatorCreditBookNode, ffi::Object);
};

class TestingMeasuredMutatorCreditBook : public ffi::ObjectRef {
 public:
  TestingMeasuredMutatorCreditBook(int num_mutators, int window_size,
                                   int cold_start_measurements, double exploration_scale) {
    data_ = ffi::make_object<TestingMeasuredMutatorCreditBookNode>(
        num_mutators, window_size, cold_start_measurements, exploration_scale);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TestingMeasuredMutatorCreditBook, ffi::ObjectRef,
                                                TestingMeasuredMutatorCreditBookNode);
};

TestingMeasuredMutatorCreditBook MakeTestingMeasuredMutatorCreditBook(
    int num_mutators, int window_size, int cold_start_measurements, double exploration_scale) {
  return TestingMeasuredMutatorCreditBook(num_mutators, window_size, cold_start_measurements,
                                          exploration_scale);
}

ffi::Array<double> TestingCreditStats(const TestingMeasuredMutatorCreditBook& credit_book,
                                      int mutator_index) {
  MeasuredMutatorCreditStats stats = credit_book->book.Stats(mutator_index);
  return {static_cast<double>(stats.window_count),
          static_cast<double>(stats.window_improvements), stats.mean_reward};
}

ffi::Array<double> TestingCreditUCB1Scores(
    const TestingMeasuredMutatorCreditBook& credit_book) {
  std::vector<double> scores = credit_book->book.UCB1Scores();
  return ffi::Array<double>(scores.begin(), scores.end());
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  TestingMeasuredMutatorCreditBookNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.meta_schedule.testing.MakeMeasuredMutatorCreditBook",
           MakeTestingMeasuredMutatorCreditBook)
      .def("s_tir.meta_schedule.testing.RecordMeasuredMutatorOutcome",
           [](const TestingMeasuredMutatorCreditBook& credit_book, int mutator_index,
              double measured_parent_latency_sec, double measured_child_latency_sec) {
             credit_book->book.RecordMeasuredOutcome(mutator_index, measured_parent_latency_sec,
                                                     measured_child_latency_sec);
           })
      .def("s_tir.meta_schedule.testing.MeasuredMutatorCreditStats", TestingCreditStats)
      .def("s_tir.meta_schedule.testing.MeasuredMutatorCreditTotal",
           [](const TestingMeasuredMutatorCreditBook& credit_book) {
             return credit_book->book.total_measured_outcomes();
           })
      .def("s_tir.meta_schedule.testing.MeasuredMutatorCreditInColdStart",
           [](const TestingMeasuredMutatorCreditBook& credit_book) {
             return credit_book->book.InColdStart();
           })
      .def("s_tir.meta_schedule.testing.MeasuredMutatorCreditUCB1Scores",
           TestingCreditUCB1Scores)
      .def("s_tir.meta_schedule.testing.SelectMeasuredMutatorCredit",
           [](const TestingMeasuredMutatorCreditBook& credit_book, int cold_start_choice) {
             return credit_book->book.SelectMutator(cold_start_choice);
           })
      .def("s_tir.meta_schedule.testing.SelectMeasuredMutatorCreditExcluding",
           [](const TestingMeasuredMutatorCreditBook& credit_book, int cold_start_choice,
              const ffi::Array<int64_t>& unavailable) {
             std::vector<uint8_t> mask;
             mask.reserve(unavailable.size());
             for (int64_t value : unavailable) {
               TVM_FFI_CHECK(value == 0 || value == 1, ValueError)
                   << "unavailable entries must be zero or one";
               mask.push_back(static_cast<uint8_t>(value));
             }
             return credit_book->book.SelectMutator(cold_start_choice, mask);
           });
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

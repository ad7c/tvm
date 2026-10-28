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
#ifndef TVM_S_TIR_META_SCHEDULE_DRAFT_MUTATOR_CREDIT_H_
#define TVM_S_TIR_META_SCHEDULE_DRAFT_MUTATOR_CREDIT_H_

#include <cstdint>
#include <deque>
#include <vector>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*! \brief Statistics retained for one mutator in the sliding reward window. */
struct MeasuredMutatorCreditStats {
  /*! \brief Number of binary outcomes currently retained in the window. */
  int window_count;
  /*! \brief Number of retained outcomes where the measured child was faster. */
  int window_improvements;
  /*! \brief Mean binary reward in the current window. */
  double mean_reward;
};

/*!
 * \brief Sliding-window UCB1 bookkeeping for mutators in Draft-guided search.
 *
 * This class deliberately has no API that accepts a Draft or learned cost-model score.
 * A reward can only be recorded from a pair of measured parent/child latencies.  The
 * caller should skip an outcome unless both candidates were actually run and are a
 * valid ancestry pair.  A child receives reward one exactly when its measured latency
 * is lower than the parent's, and zero otherwise.
 *
 * Selection is deterministic.  During cold start, the caller supplies the mutator selected
 * by the baseline policy, preserving its original prior weights.  Afterwards, the mutator
 * with maximum sliding-window UCB1 score is selected, with the lowest index breaking ties.
 */
class MeasuredMutatorCreditBook {
 public:
  static constexpr int kDefaultWindowSize = 50;
  static constexpr int kDefaultColdStartMeasurements = 32;
  static constexpr double kDefaultExplorationScale = 1.0;

  explicit MeasuredMutatorCreditBook(
      int num_mutators, int window_size = kDefaultWindowSize,
      int cold_start_measurements = kDefaultColdStartMeasurements,
      double exploration_scale = kDefaultExplorationScale);

  /*!
   * \brief Record one reward derived exclusively from real measured latency.
   * \param mutator_index Index of the most recent mutator credited for this child.
   * \param measured_parent_latency_sec Runner-measured latency of the paired parent.
   * \param measured_child_latency_sec Runner-measured latency of the child.
   */
  void RecordMeasuredOutcome(int mutator_index, double measured_parent_latency_sec,
                             double measured_child_latency_sec);

  /*! \brief Whether fewer than the configured number of measured outcomes exist. */
  bool InColdStart() const;

  /*! \brief Compute the current sliding-window UCB1 score for every mutator. */
  std::vector<double> UCB1Scores() const;

  /*!
   * \brief Select a mutator without owning randomness.
   *
   * During cold start, `cold_start_choice` is returned and must be a valid index sampled by
   * the caller's baseline policy.  After cold start it is ignored and UCB1 is used.
   */
  int SelectMutator(int cold_start_choice) const;

  /*!
   * \brief Select a mutator while temporarily excluding failed arms.
   *
   * `unavailable` is local retry state owned by the caller; it never changes
   * the measured reward window.  An empty mask is equivalent to
   * `SelectMutator(cold_start_choice)`.  Returns -1 when every arm is masked.
   */
  int SelectMutator(int cold_start_choice, const std::vector<uint8_t>& unavailable) const;

  /*! \brief Return the retained sliding-window statistics for one mutator. */
  MeasuredMutatorCreditStats Stats(int mutator_index) const;

  /*! \brief Lifetime number of accepted real-measurement outcomes. */
  int64_t total_measured_outcomes() const { return total_measured_outcomes_; }

  int num_mutators() const { return static_cast<int>(arms_.size()); }
  int window_size() const { return window_size_; }
  int cold_start_measurements() const { return cold_start_measurements_; }

 private:
  struct Arm {
    std::deque<uint8_t> outcomes;
    int improvements{0};
  };

  void CheckMutatorIndex(int mutator_index) const;

  std::vector<Arm> arms_;
  int window_size_;
  int cold_start_measurements_;
  double exploration_scale_;
  int64_t total_measured_outcomes_{0};
};

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_DRAFT_MUTATOR_CREDIT_H_

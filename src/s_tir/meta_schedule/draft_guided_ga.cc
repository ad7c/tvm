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

#include "draft_guided_ga.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

std::vector<double> NormalizePrunerDraftGAFitness(const std::vector<PrunerDraftScore>& scores) {
  double best = 0.0;
  for (const PrunerDraftScore& score : scores) {
    if (score.known && std::isfinite(score.score) && score.score > best) {
      best = score.score;
    }
  }

  if (best <= 0.0) return std::vector<double>(scores.size(), 1.0);

  constexpr double kUnknownFitness = 1.0e-6;
  std::vector<double> fitness(scores.size(), kUnknownFitness);
  for (int i = 0, n = static_cast<int>(scores.size()); i < n; ++i) {
    if (scores[i].known && std::isfinite(scores[i].score) && scores[i].score > 0.0) {
      fitness[i] = std::max(kUnknownFitness, std::min(1.0, scores[i].score / best));
    }
  }
  return fitness;
}

int PrunerDraftGAVerifyCount(int population_size, int output_size, double verify_ratio) {
  TVM_FFI_CHECK_GT(population_size, 0, ValueError);
  TVM_FFI_CHECK_GT(output_size, 0, ValueError);
  TVM_FFI_CHECK(std::isfinite(verify_ratio) && verify_ratio > 0.0 && verify_ratio <= 1.0,
                ValueError)
      << "verify_ratio must be within (0, 1], but got " << verify_ratio;
  return std::max(output_size, static_cast<int>(std::ceil(population_size * verify_ratio)));
}

int SelectPrunerDraftMutationDirection(const PrunerDraftScore& parent,
                                       const std::vector<PrunerDraftScore>& children, bool explore,
                                       int exploration_index) {
  if (children.empty()) return -1;
  if (explore) {
    TVM_FFI_CHECK(0 <= exploration_index && exploration_index < static_cast<int>(children.size()),
                  ValueError)
        << "exploration_index is outside the child range";
    return exploration_index;
  }

  int best_index = -1;
  double best_score = 0.0;
  for (int i = 0, n = static_cast<int>(children.size()); i < n; ++i) {
    const PrunerDraftScore& child = children[i];
    if (child.known && std::isfinite(child.score) && child.score > best_score) {
      best_index = i;
      best_score = child.score;
    }
  }
  if (best_index == -1) return -1;
  if (!parent.known || !std::isfinite(parent.score) || parent.score <= 0.0) {
    return best_index;
  }
  return best_score > parent.score ? best_index : -1;
}

double PrunerDraftAnnealAcceptanceProbability(double parent_score, double child_score,
                                              double temperature) {
  TVM_FFI_CHECK(std::isfinite(parent_score), ValueError)
      << "parent_score must be finite, but got " << parent_score;
  TVM_FFI_CHECK(std::isfinite(child_score), ValueError)
      << "child_score must be finite, but got " << child_score;
  TVM_FFI_CHECK(!std::isnan(temperature), ValueError) << "temperature must not be NaN";

  if (child_score >= parent_score) return 1.0;
  if (temperature <= 0.0) return 0.0;

  constexpr double kRelativeScoreEpsilon = 1.0e-12;
  double relative_drop =
      (parent_score - child_score) / std::max(std::abs(parent_score), kRelativeScoreEpsilon);
  return std::exp(-relative_drop / temperature);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.meta_schedule.testing.NormalizePrunerDraftGAFitness",
           [](ffi::Array<double> raw_scores, ffi::Array<int64_t> known) {
             TVM_FFI_CHECK_EQ(raw_scores.size(), known.size(), ValueError)
                 << "Scores and known flags must have the same length";
             std::vector<PrunerDraftScore> scores;
             scores.reserve(raw_scores.size());
             for (int i = 0, n = raw_scores.size(); i < n; ++i) {
               scores.push_back(PrunerDraftScore{known[i] != 0, raw_scores[i], 1});
             }
             std::vector<double> fitness = NormalizePrunerDraftGAFitness(scores);
             return ffi::Array<double>(fitness.begin(), fitness.end());
           })
      .def("s_tir.meta_schedule.testing.PrunerDraftGAVerifyCount", PrunerDraftGAVerifyCount)
      .def("s_tir.meta_schedule.testing.SelectPrunerDraftMutationDirection",
           [](bool parent_known, double parent_score, ffi::Array<int64_t> child_known,
              ffi::Array<double> child_scores, bool explore, int exploration_index) {
             TVM_FFI_CHECK_EQ(child_known.size(), child_scores.size(), ValueError);
             std::vector<PrunerDraftScore> children;
             children.reserve(child_scores.size());
             for (int i = 0, n = child_scores.size(); i < n; ++i) {
               children.push_back(PrunerDraftScore{child_known[i] != 0, child_scores[i], 1});
             }
             return SelectPrunerDraftMutationDirection(
                 PrunerDraftScore{parent_known, parent_score, 1}, children, explore,
                 exploration_index);
           })
      .def("s_tir.meta_schedule.testing.PrunerDraftAnnealAcceptanceProbability",
           PrunerDraftAnnealAcceptanceProbability);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

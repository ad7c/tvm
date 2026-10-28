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
#ifndef TVM_S_TIR_META_SCHEDULE_DRAFT_GUIDED_GA_H_
#define TVM_S_TIR_META_SCHEDULE_DRAFT_GUIDED_GA_H_

#include <vector>

#include "draft_model.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*!
 * \brief Convert analytic Draft scores into the sole fitness used by Draft-guided GA.
 *
 * Known scores are normalized to the best known candidate.  An unparseable candidate
 * receives a small, non-zero weight so it does not look better than a known candidate,
 * while mutation can still explore from it.  If the complete population is unknown,
 * every candidate receives the same weight and selection becomes uniform.
 */
std::vector<double> NormalizePrunerDraftGAFitness(const std::vector<PrunerDraftScore>& scores);

/*!
 * \brief Number of cross-generation Draft elites sent once to the learned cost model.
 *
 * The verification pool is at least the requested output size, and otherwise is
 * `ceil(population_size * verify_ratio)`.
 */
int PrunerDraftGAVerifyCount(int population_size, int output_size, double verify_ratio);

/*!
 * \brief Select a mutation direction using only Draft scores.
 *
 * \return A child index, or -1 to keep the parent.  Exploitation selects the
 * highest-scoring known child only when it improves on a known parent.  If the
 * parent is unknown, the best known child supplies a usable direction.  The
 * caller supplies the exploration decision/index so RNG ownership remains in
 * the evolutionary-search thread.
 */
int SelectPrunerDraftMutationDirection(const PrunerDraftScore& parent,
                                       const std::vector<PrunerDraftScore>& children, bool explore,
                                       int exploration_index);

/*!
 * \brief Compute the simulated-annealing acceptance probability using Draft scores.
 *
 * Draft scores are oriented so that larger is better.  An equal or better child is
 * therefore always accepted.  For a worse child, the loss is made scale independent
 * using `(parent_score - child_score) / max(abs(parent_score), eps)` before applying
 * `exp(-relative_drop / temperature)`.  A non-positive temperature disables acceptance
 * of worse children.
 */
double PrunerDraftAnnealAcceptanceProbability(double parent_score, double child_score,
                                              double temperature);

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_DRAFT_GUIDED_GA_H_

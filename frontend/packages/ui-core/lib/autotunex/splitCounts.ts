// Copyright Granite.Build Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * The train/validation record counts an auto-split upload will get, computed the
 * way the server does it (`_validation_indices` in services/datasets_io.py):
 * validation is `round(total * validationPct / 100)` -- Python's round, half to
 * even -- and train is the rest.
 *
 * Upload and Review used to compute this separately, flooring and rounding
 * respectively, and disagreed with each other for most totals.
 */
export function splitCounts(total: number, validationPct: number): { train: number; validation: number } {
  const exact = (total * validationPct) / 100
  let validation = Math.round(exact)
  if (Math.abs(exact % 1) === 0.5 && validation % 2 !== 0) validation -= 1
  return { train: total - validation, validation }
}

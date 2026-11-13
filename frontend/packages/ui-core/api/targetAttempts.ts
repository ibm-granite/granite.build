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

import type { BuildTargetRun } from '../types'

/**
 * Order attempts of one target by `started_at ?? finished_at`. Attempts run
 * one after another (n+1 is created only once n is terminal), so every
 * timestamp of n+1 is later than every timestamp of n and mixing the two
 * fields is sound. A run with neither is the queued current attempt —
 * `started_at` is only stamped on first entry into RUNNING — so it sorts
 * newest; only one attempt is in flight at a time, so two never tie there.
 */
export function attemptOrder(t: Pick<BuildTargetRun, 'started_at' | 'finished_at'>): number {
  const at = t.started_at ?? t.finished_at
  const ms = at ? Date.parse(at) : NaN
  return Number.isNaN(ms) ? Infinity : ms
}

/** Whether `candidate` is the same target's current attempt relative to `existing`. */
export function isLaterAttempt(
  candidate: Pick<BuildTargetRun, 'started_at' | 'finished_at'>,
  existing: Pick<BuildTargetRun, 'started_at' | 'finished_at'> | undefined,
): boolean {
  return !existing || attemptOrder(candidate) >= attemptOrder(existing)
}

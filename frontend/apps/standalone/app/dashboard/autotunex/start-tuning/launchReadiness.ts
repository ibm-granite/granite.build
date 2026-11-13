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
 * The first step before `lastStepIndex` whose own Next gate fails, or null.
 *
 * Launch has to re-check every earlier step: `completedSteps` only records that
 * a step was once passed, so re-entering one from Review and invalidating it --
 * clearing the model, "Choose Existing" leaving no configuration, clearing the
 * reward code -- left Review reachable, and the launch created and uploaded the
 * dataset before the server rejected the job.
 */
export function firstIncompleteStep(isStepValid: (step: number) => boolean, lastStepIndex: number): number | null {
  for (let step = 0; step < lastStepIndex; step++) {
    if (!isStepValid(step)) return step
  }
  return null
}

/**
 * Per step, whether the indicator should flag it: it was completed once but no
 * longer passes its own gate. The step being edited is left alone -- its Next is
 * already disabled, and a warning on the page you are on is noise.
 *
 * The same gate as `firstIncompleteStep`, so the step that disables Launch is the
 * step that is flagged.
 */
export function invalidCompletedSteps(
  completedSteps: boolean[],
  isStepValid: (step: number) => boolean,
  lastStepIndex: number,
  currentStep: number
): boolean[] {
  return completedSteps.map((done, step) => done && step < lastStepIndex && step !== currentStep && !isStepValid(step))
}

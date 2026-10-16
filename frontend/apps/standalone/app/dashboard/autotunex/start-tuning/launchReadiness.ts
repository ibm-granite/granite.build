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

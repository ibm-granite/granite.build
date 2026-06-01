# Build- and Step-level Retry
The framework expects to operate in environments that deliver intermittent failure
modalities to the build - such things as network outages, unavailable resources, etc.
To provide greater resilience in the face of such issues, two levels of resiliency 
are available, build- and step-level retries, discussed below:

## Build-level Retries
A build that has completed and is marked as `FAILED` (or `CANCELLED`?) can be _retried_.
A build that is retried will skip any targets (and output artifacts)  of the build.yaml 
that have already been computed (on the previous `FAILED` run), and the build 
will pick up where it left off with the remaining `FAILED` targets. 

1. A new build id is used to run the retried build. 
1. Apart from the upstream output artifacts of previous targets, 
the retried targets will start from scratch with no state from the previous failed run of the target.
1. Previously run (`SUCCESS`ful) targets are identified by comparing a hash computed
on the target section of the build and looking this hash up in the gb_targets table
1. `FAILED` builds will be retried automatically if configured in the build.yaml.
See [build-retry.md](build-retry.md) for details. 

## Step-Level Retries
When a step, either internal (e.g., lhpull, lhpush) or external 
(e.g, tuning, digit, etc), encounters an error, it can be retried through the RetryHandler and Environment implementations.  

1. Step-level retries are performed within the framework of a single build execution.
1. In general, but as defined by the environment implementation, a retried step is provided the same on-disk state as when the step failed. 
This allows resumable steps, that can start from the point they left off (i.e. failed),
to avoid unnecessary recomputations.  Steps such as `tuning` or `digit` allow this.
1. When retried, a step may be retried transparently or non-transparently.  
     * Tanparent retries result in the build target and step metadata 
     (gb_target/steps tables) being left as if the step had never been retried.  
     * Non-transparent retries leave the FAILED target and step
     records in the build metadata tables.
1. Step retry configuration is set the step definition (step.yaml) or the step section of the build (build.yaml). For more detail on step-level retry
configuration, see [step-retry-configuration](step-retry-configuration.md).
1. Currently only K8s (helm) and Lsf (bsub) environments support step-level retries.
1. Currently only the built-in steps lhpull/push and s3pull/push steps are enabled for retry. NOTE: there is no way for the end user to disable retry for these steps.

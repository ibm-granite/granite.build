/**
 * How long to keep polling a freshly uploaded dataset before giving up.
 *
 * The multipart upload endpoint returns 202 with status "uploading" and finishes
 * processing off-request, so every upload path has to poll the dataset row until
 * the server marks it ready (or error). Both callers — the Start Tuning wizard's
 * waitForDatasetReady and the Settings "create dataset" modal — share this
 * deadline so a stuck "uploading" dataset is abandoned after the same window
 * rather than polling forever.
 *
 * The poll *interval* is deliberately not shared: the wizard polls faster
 * (it blocks a launch) than the Settings modal (a background create).
 */
export const DATASET_READY_TIMEOUT_MS = 5 * 60 * 1000

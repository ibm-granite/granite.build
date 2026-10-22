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

/**
 * How long to keep polling a HuggingFace import before giving up.
 *
 * Deliberately NOT the same constant as DATASET_READY_TIMEOUT_MS above, which
 * both the wizard's upload path and the Settings create modal alias: an import
 * streams parquet shards from the Hub and takes minutes, while an upload's
 * server-side processing takes seconds. Raising the shared constant to fit an
 * import would silently extend the upload paths' deadline too.
 */
export const HF_IMPORT_READY_TIMEOUT_MS = 20 * 60 * 1000

/**
 * Poll interval for an import. Slower than the wizard's 1.5 s upload poll: this
 * runs for minutes, so a tighter interval buys nothing but requests.
 */
export const HF_IMPORT_POLL_MS = 3000

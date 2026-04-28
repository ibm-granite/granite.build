# llmbsub - LLM.build Job Submission with Input Lineage Tracking

## Overview

`llmbsub` is a bsub wrapper that submits jobs to LSF on BlueVela while tracking input/output artifacts for LLM.build lineage. When enabled, it also uploads input datasets to Lakehouse (DMF) for input lineage tracking.

## Architecture

```
llmbsub CLI (llmbsub_cli.py)
        │
        ├── Submits main LSF job via bsub
        │
        └── If LLMBSUB_INPUT_UPLOAD=true:
                │
                └── submit_upload_job() → bsub (CPU-only job)
                        │
                        └── upload_wrapper.sh
                                │
                                └── input_upload_service.py
                                        │
                                        ├── Calculate checksum (llmbsum.sh)
                                        ├── Check cache (checksum_cache.py)
                                        ├── Query gbserver /artifact/status API
                                        └── If not exists: dmf push
```

## Key Components

### 1. llmbsub_cli.py

Main CLI entry point. Wraps `bsub` to:
- Parse `--llmbin` inputs and `--llmbout` outputs
- Generate build.yaml for LLM.build tracking
- Submit the main LSF job
- Optionally submit a separate upload job for inputs (when `LLMBSUB_INPUT_UPLOAD=true`)

**Key functions:**
- `submit_bsub()` - Main submission logic
- `submit_upload_job()` - Submits CPU-only job for input uploads

### 2. services/input_upload_service.py

Core upload service that:
- Calculates checksums using `llmbsum.sh` (xxh64sum + parallel)
- Queries gbserver `/artifact/status` API
- Handles concurrent upload coordination (pending status polling)
- Uploads to DMF using `dmf fileset push`
- Monitors source integrity during upload (aborts if files change)
- Implements retry logic for transient failures

**Key classes:**
- `InputUploadService` - Main service class
- `UploadStatus` - Enum: COMPLETED, EXISTS, IN_PROGRESS, ABORTED, ERROR
- `UploadResult` - Dataclass with input_path, checksum, dmf_uri, status, message
- `TransientError` / `PermanentError` - For retry logic

**Configuration constants:**
- `RETRY_DELAYS = [30, 60, 120]` - Seconds between retries
- `INTEGRITY_CHECK_INTERVAL = 30` - Seconds between mtime checks
- `PENDING_POLL_INTERVAL = 30` - Seconds between status polls
- `PENDING_POLL_MAX_WAIT = 3600` - Max wait for pending upload (1 hour)

### 3. utils/checksum_cache.py

Local checksum cache with mtime-based invalidation:
- Cache stored at `/proj/granite-build/llmb/upload/checksum_cache/checksums.json`
- Keyed by path hash (sha256[:16])
- Invalidated when directory mtime changes
- Uses `portalocker` for concurrent access safety

**Key methods:**
- `get(path)` - Returns cached checksum if mtime unchanged
- `set(path, checksum)` - Stores checksum with current mtime
- `get_all_mtimes(path)` - For integrity monitoring during upload

### 4. utils/llmbsum.sh

Bash script for parallel checksum calculation:
```bash
# Uses xxh64sum with GNU parallel
find "$directory_path" -type f | sort | \
    parallel -P"$concurrency" --keep-order xxh64sum -H1 > output.all
xxh64sum -H1 output.all > output.cksum
```

**Dependencies:**
- `xxh64sum` - BlueVela: `/proj/granite-build/tools/xxHash/xxh64sum`
- `parallel` - GNU parallel for concurrent execution

### 5. utils/upload_wrapper.sh

Bsub job wrapper that:
- Traps SIGTERM/SIGINT to finish current upload before exit
- Calls Python upload service
- Logs progress to upload.log

## gbserver API Contract

```
POST /api/v1/artifact/status
Request:  { path, checksum, space_name }
Response:
  - If exists:     { status: "exists", uri: "lh://..." }
  - If pending:    { status: "pending" }  # Another job uploading
  - If not exists: { status: "not_found", target_uri: "lh://..." }
```

## Concurrent Upload Handling

When gbserver returns `status: "pending"`:
1. Poll with backoff every 30 seconds
2. Max wait: 1 hour (`PENDING_POLL_MAX_WAIT`)
3. Outcomes:
   - `exists` → Other job succeeded, skip upload
   - `not_found` → Other job failed, attempt our upload
   - Timeout → Error, report failure

## Data Integrity During Upload

To detect modifications while uploading:
1. Record mtime of all files before upload
2. Background thread checks mtimes every 30 seconds
3. If mtime changed → abort upload, return ABORTED status

## Feature Flag

Upload feature is disabled by default:
```bash
export LLMBSUB_INPUT_UPLOAD=true  # Enable input uploads
```

Controlled in `llmbsub_constants.py`:
```python
UPLOAD_FEATURE_FLAG = getenv_boolean("LLMBSUB_INPUT_UPLOAD", False)
```

## Testing

### Test Files

- `tests/test_input_upload.py` - Comprehensive test suite with mocks
- `tests/test_upload_abort_on_modify.py` - Standalone integrity monitoring test
- `tests/conftest.py` - Pytest markers (gpfs, slow)

### Test Architecture

Tests are **standalone** (runnable without gbcli installed):
- Inline mock classes: `MockGBServerAPI`, `MockDMFCLI`, `MockChecksumScript`, `MockChecksumCache`
- Real GPFS tests marked with `@pytest.mark.gpfs`
- Performance benchmarks marked with `@pytest.mark.slow`

### Running Tests

```bash
# Unit tests (anywhere)
pytest test_input_upload.py -v -m "not gpfs and not slow"

# GPFS tests (BlueVela only)
pytest test_input_upload.py -v -m "gpfs"

# Standalone abort test
python test_upload_abort_on_modify.py /path/to/data 5.0
```

### Key Test Categories

1. **ChecksumCache Tests** - Cache hits/misses, mtime invalidation, file locking
2. **InputUploadService Tests** - API responses, pending polling, retries
3. **URI Parsing Tests** - Lakehouse URI component extraction
4. **Integrity Monitoring Tests** - Abort on file modification
5. **GPFS Benchmarks** - Checksum performance at scale

## Common Issues

### llmbsum.sh Silent Failures

The script can return rc=0 even when `parallel` fails:
- `.all` file will be empty (0 bytes)
- `xxh64sum` still produces a checksum of the empty file

**Validation needed:**
- Check stderr for "command not found"
- Verify `.all` file has content

### Local Development (Mac)

Update paths in scripts:
```bash
# llmbsum.sh
sum_tool="/opt/homebrew/bin/xxh64sum -H1"  # Install: brew install xxhash

# test_upload_abort_on_modify.py
LLMBSUM_SCRIPT_PATH = "/path/to/llmbsub/utils/llmbsum.sh"
```

Install dependencies:
```bash
brew install xxhash parallel
```

## File Structure

```
llmbsub/
├── __main__.py
├── llmbsub_cli.py           # Main CLI entry point
├── services/
│   ├── input_upload_service.py  # Core upload logic
│   └── llmbsub_service_build.py # Build submission
├── utils/
│   ├── checksum_cache.py    # Local checksum caching
│   ├── llmbsum.sh           # Parallel checksum script
│   ├── upload_wrapper.sh    # Bsub job wrapper
│   ├── llmbsub_constants.py # Configuration constants
│   └── scriptutil.py        # Build YAML generation
└── tests/
    ├── conftest.py          # Pytest configuration
    ├── test_input_upload.py # Main test suite
    └── test_upload_abort_on_modify.py  # Standalone test
```

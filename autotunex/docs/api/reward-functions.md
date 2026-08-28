# Reward Functions API

A **reward function** is a user-supplied Python function that scores a model's output during
online-RL tuning. Because the code is written by a caller and runs on the server, it is
checked before it is ever used: this endpoint statically analyses the source and, on request,
runs it against sample test cases inside a hardened subprocess sandbox. This page documents
the endpoint under the `/api/v1/reward-functions` prefix.

See [overview.md](overview.md) for shared conventions, [authentication.md](authentication.md)
for credential modes, and [../concepts.md](../concepts.md) for when a reward function is
required at all.

## The one thing to get right: this endpoint always returns 200

A reward function that fails syntax, security, signature, or execution checks is **not** a
`4xx`. It is `success: false` inside a normal `200` body. Clients must branch on the
`success` field, never on the status code. A `4xx` here means the *request* was wrong (bad
credential, malformed body) — not that the reward function was rejected.

The endpoint validates code; it does not own or store anything. It is authenticated like
every data route, but takes no `scope` query parameter and reads no rows.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/reward-functions/validate` | Static-check, and optionally sandbox-run, a reward function |

---

## POST /api/v1/reward-functions/validate

Run three in-process static phases (syntax, security, signature) over the supplied source
and, only when `test_execution` is `true` **and** every static phase passed, execute the
function in the sandbox against the test cases. Returns `200` with a
`RewardValidationResponse`. Unknown top-level fields are rejected (`extra="forbid"`).

### Request body — `RewardValidationRequest`

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `code` | string | yes | — | The reward function's Python source. Rejected if blank, or above 50 000 bytes |
| `function_name` | string | no | `compute_score` | The entry point to look for in `code` |
| `test_execution` | bool | no | `false` | When `false`, only static analysis runs and `test_result` is `null` |
| `test_inputs` | `RewardTestCase` \| `RewardTestCase[]` \| null | no | `null` | A single case **or** a list of them. When `null`, one built-in sample case is used |

**`RewardTestCase`** — all four canonical fields are optional and `extra="allow"`: any
additional key you send is preserved and forwarded to the reward function as a keyword
argument alongside them.

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `data_source` | string \| null | no | `null` | The prompt the model answered |
| `solution_str` | string \| null | no | `null` | The model output to score |
| `ground_truth` | any \| null | no | `null` | The reference answer, in whatever shape your function expects |
| `extra_info` | object \| null | no | `null` | Free-form extra context |
| *(any other key)* | any | no | — | Allowed and forwarded as a kwarg |

Each case is passed to the function as `fn(**case)`, with the canonical keys always present
(a `null` field is sent as `null`, not omitted). Only the **first 10** cases of a list are
executed.

```bash
curl -X POST https://example.com/api/v1/reward-functions/validate \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def compute_score(data_source, solution_str, ground_truth=None, extra_info=None):\n    return 1.0 if str(ground_truth) in solution_str else 0.0\n",
    "function_name": "compute_score",
    "test_execution": true,
    "test_inputs": [
      {
        "data_source": "What is the capital of France?",
        "solution_str": "The capital of France is Paris.",
        "ground_truth": "Paris"
      }
    ]
  }'
```

### Response `200` — `RewardValidationResponse`

| Field | Type | Notes |
| --- | --- | --- |
| `success` | bool | `true` only if every static check passed **and** (when run) execution raised nothing, in no case |
| `validation` | `RewardValidationChecks` | The four status booleans |
| `security_issues` | string[] | One human-readable finding per blocked import, call, or attribute; `[]` if clean |
| `syntax_errors` | string[] | The syntax/size/empty-code errors on an early failure; otherwise the function-found and signature errors |
| `test_result` | `RewardTestResult` \| null | `null` unless `test_execution` was `true` |

**`RewardValidationChecks`** — what the UI renders as status pills:

| Field | Type | Notes |
| --- | --- | --- |
| `syntax_valid` | bool | The source parsed as Python |
| `security_valid` | bool | `security_issues` is empty |
| `function_found` | bool | A `def` named `function_name` exists |
| `function_signature_valid` | bool | That `def` declares at least two positional parameters |

**`RewardTestResult`**:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `executed` | bool | — | `false` when the sandbox never ran the code (static checks failed, timeout, or abnormal exit) |
| `results` | `RewardCaseResult[]` | `[]` | One entry per executed case |
| `stdout` | string | `""` | Captured `print` output, truncated to 2 000 characters |
| `error` | string \| null | `null` | A whole-run failure (import/compile error, timeout, sandbox crash) |
| `execution_time_ms` | float \| null | `null` | Wall-clock time of the whole run, as measured in the sandbox |

**`RewardCaseResult`**:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `case` | int | — | 1-based index of the case |
| `inputs` | object | — | The exact kwargs the function was called with |
| `return_value` | any \| null | `null` | The returned value; coerced with `str()` if not JSON-serializable |
| `return_type` | string \| null | `null` | The Python type name of the return value (e.g. `float`) |
| `error` | string \| null | `null` | `TypeName: message` if **this case** raised |

A per-case error is data, not a failed request — the other cases still run, and the response
is still `200`. It does set `success` to `false`.

```json
{
  "success": true,
  "validation": {
    "syntax_valid": true,
    "security_valid": true,
    "function_found": true,
    "function_signature_valid": true
  },
  "security_issues": [],
  "syntax_errors": [],
  "test_result": {
    "executed": true,
    "results": [
      {
        "case": 1,
        "inputs": {
          "data_source": "What is the capital of France?",
          "solution_str": "The capital of France is Paris.",
          "ground_truth": "Paris",
          "extra_info": null
        },
        "return_value": 1.0,
        "return_type": "float",
        "error": null
      }
    ],
    "stdout": "",
    "error": null,
    "execution_time_ms": 1.4
  }
}
```

### Notable statuses

| Status | When |
| --- | --- |
| `200` | Always, for any validation verdict — pass or fail. Branch on `success` |
| `400` | Declared: a malformed request, e.g. presenting two credentials at once |
| `401` | No credential, or a credential that failed to verify |
| `422` | An unknown top-level field, or a field of the wrong type (`extra="forbid"`) |
| `503` | Declared for a sandbox or upstream failure. The validation service is built unconditionally from settings and depends on no optional upstream, so no code path raises it today |

Note that a rejected reward function never appears here: empty code, oversized code, a
syntax error, a blocked import, a missing function, and a bad signature are all `200` with
`success: false`.

## What the static analysis checks

Three pure, in-process phases run over the AST before the sandbox is even considered
(`services/reward/static_analysis.py`). Anything flagged here never reaches the sandbox.

1. **Syntax** — `ast.parse`. On failure, `syntax_errors` carries
   `Syntax error at line <n>: <msg>` and every other check reports `false`. Two cheaper
   guards run first, reporting the same way: blank `code` gives `Code cannot be empty`, and
   source above 50 000 bytes gives `Code exceeds maximum allowed size (50KB)`.
2. **Security** — the tree is walked for three classes of finding, listed in
   `security_issues` with their line numbers:
   - **Forbidden imports** — any `import`/`from … import` whose top-level module is on the
     blocklist (`os`, `sys`, `subprocess`, `socket`, `urllib`, `requests`, `ctypes`,
     `pickle`, `importlib`, `pathlib`, `io`, `builtins`, `threading`, `multiprocessing`, and
     others), **or** whose top-level name starts with `_`.
   - **Forbidden builtin calls** — a direct call to `exec`, `eval`, `compile`,
     `__import__`, `open`, `input`, `breakpoint`, `exit`, `quit`, `globals`, `locals`,
     `vars`, `dir`, `getattr`, `setattr`, or `delattr`.
   - **Forbidden attribute access** — the dunders used for sandbox escapes:
     `__import__`, `__builtins__`, `__subclasses__`, `__class__`, `__bases__`,
     `__globals__`, `__code__`, `__closure__`, `__dict__`, `__module__`, `__qualname__`.
3. **Signature** — a `def` named `function_name` must exist and declare **at least two**
   positional parameters, matching the `(data_source, solution_str, …)` contract. Extra
   parameters and defaults are fine; fewer than two is
   `function_signature_valid: false`. Only a plain `def` is matched, so an `async def` of
   that name reports `function_found: false`.

The blocklists are the security boundary and are carried forward deliberately rather than
tuned; they live in `services/reward/constants.py`.

## The execution sandbox

When `test_execution` is `true` and all static checks passed, the code runs through the
`RewardExecutor` seam. The one shipped implementation
(`services/reward/subprocess_executor.py`) starts a **separate Python process** —
`python -m autotunex.services.reward._child` — in its own session/process group, and:

- **Restricts builtins.** The code is `exec`'d against a namespace exposing only a
  whitelist of safe builtins and exception types. `__import__` is replaced by a restricted
  version that permits a small module allowlist (`math`, `re`, `json`, `string`,
  `collections`, `functools`, `itertools`, `typing`, `dataclasses`, `enum`, `decimal`,
  `fractions`, `statistics`, `copy`, `numbers`, `abc`, `textwrap`, `difflib`,
  `unicodedata`) and raises `ImportError` for anything else. `operator` is deliberately
  excluded — its `attrgetter`/`methodcaller` take attribute names as strings and so bypass
  both the removed `getattr` builtin and the AST dunder blocklist.
- **Withholds the environment.** Only a small allowlist of interpreter/locale variables
  (`PATH`, `PYTHONPATH`, `PYTHONHOME`, `PYTHONIOENCODING`, `HOME`, `TMPDIR`, `LANG`,
  `LC_ALL`, `LC_CTYPE`, and Windows-startup essentials) is copied into the child. Database
  URLs, tokens, and OIDC/session secrets in the parent's environment are never passed, so
  even a sandbox escape reaching `os.environ` finds no credentials.
- **Caps memory.** `RLIMIT_AS` is set to `reward_memory_bytes` (default 512 MiB). Enforced
  on Linux; on macOS it may be a no-op.
- **Caps CPU and wall clock.** The child sets `RLIMIT_CPU` to the timeout plus a 30-second
  buffer, deliberately generous so it acts only as a backstop. The parent is the primary
  defense: one second past `reward_timeout_seconds` it `SIGKILL`s the child's whole process
  group and returns `executed: false` with `Execution timed out after <n>s` — where `<n>` is
  the configured timeout, not the extra second the parent actually waited. A child that exits
  non-zero or writes nothing yields
  `Sandbox process exited abnormally (possible resource limit or crash)`.
- **Truncates output.** Captured stdout is cut to 2 000 characters; at most 10 test cases
  are executed.

A compile/exec failure inside the sandbox — for example importing a module the allowlist
rejects — comes back as `executed: true` with a whole-run `error` string, still inside a
`200` response.

If `test_execution` is `true` but a static check failed, no process is started and
`test_result` is `{"executed": false, "error": "Cannot execute: validation failed"}`.

### Operator settings

Both knobs are read at request time from settings; see
[../operations/configuration.md](../operations/configuration.md).

| Environment variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_REWARD_TIMEOUT_SECONDS` | Hard wall-clock (and CPU-rlimit) budget for one sandboxed run. Must be ≥ 1 | `5` |
| `AUTOTUNEX_REWARD_MEMORY_BYTES` | Address-space rlimit for the sandbox child. Must be ≥ 1 | `536870912` (512 MiB) |

## The other half of the reward step

This endpoint is one of two halves of the same online-RL wizard step, and both share the
schemas in `models/reward.py`:

- **`POST /api/v1/reward-functions/validate`** (this page) checks the function.
- **`POST /api/v1/jobs/generate-test-solutions`** (see [jobs.md](jobs.md)) LLM-generates
  sample model answers to seed the test cases you then validate against.

Validating a function is independent of submitting a job — nothing here is persisted. A job
whose referenced configuration is for an online-RL tuner requires `reward_function_code` and
`reward_function_name` on the submission itself; see [jobs.md](jobs.md).

## See also

- [overview.md](overview.md) — base URL, error shape, status codes
- [authentication.md](authentication.md) — credential modes
- [jobs.md](jobs.md) — `generate-test-solutions`, and the reward fields on job submission
- [../concepts.md](../concepts.md) — which tuners require a reward function
- [../operations/configuration.md](../operations/configuration.md) — sandbox settings

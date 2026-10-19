# HuggingFace models API

Read-only lookups the tuning wizard uses to pick a base model. Both routes query the
Hub **server-side**, so private repos in an allowlisted namespace are visible — the
browser can never see those, and every model AutoTuneX pushes is private.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/hf/models/search` | Search HuggingFace for model repo ids |
| `GET` | `/api/v1/hf/models/card` | Read a model's `README.md` (plain text) |

## Token scope

The token in the env var named by `AUTOTUNEX_HF_TOKEN_ENV` (default `HF_TOKEN`) is sent
only where `AUTOTUNEX_HF_IMPORT_NAMESPACES` allows:

- **search** runs one anonymous query, plus one query per allowlisted namespace pinned
  with `author=<namespace>` and carrying the token. Each pinned query keeps only its
  private hits; hits owned by any other namespace are discarded too. Results: private
  allowlisted hits first, then public, de-duplicated, cut to `limit`.
- **card** sends the token only when `repo_id`'s owner is allowlisted.
- A tokened request never follows a redirect. Allowlist owners match exactly, case
  included: the Hub redirects a non-canonical case, so list namespaces as the Hub spells
  them.

With no token or an empty allowlist, both routes are public-only.

Both routes also return the shared `401` (no credential, or a credential that failed to
verify) and `400` (a malformed request, e.g. presenting two credentials at once).

## `GET /hf/models/search?query=<text>&limit=<n>`

`query` is required (1+ characters); `limit` defaults to `20` (1–100). Returns a plain
JSON list of repo id strings. `503` when the Hub cannot be reached (a failed
allowlisted query alone degrades to public results instead).

## `GET /hf/models/card?repo_id=<owner/name>`

`repo_id` also accepts a bare name with no owner (3–200 characters, matching
`^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$`); a bare name is never
allowlisted, so it is always read anonymously. Returns `text/plain`. `404` when the repo
is absent **or** private and not readable with the server's token — the two are
deliberately indistinguishable — and when a tokened read is redirected (a renamed repo,
or a non-canonical case). `422` for a malformed `repo_id`; `503` when the Hub cannot be
reached.

Unlike `/datasets/hf/*`, these routes are not gated on dataset import being available.

## Tuning on a private model

A `huggingface`-source model in an allowlisted namespace (`AUTOTUNEX_HF_IMPORT_NAMESPACES`)
is bound as an `hf://` `base_model` input artifact instead of being passed by name — on
`job_backend=llmb` custom_code and LSF builds only; the bash standalone spec always binds
the model as `inputs.model`, and the `local` backend passes it by name. A bound model's
binding path is passed as `--model_name_or_path`; gbserver pulls it with the space's
credentials, because the tuning workload itself has no HF token.

## Who can see what

Any authenticated AutoTuneX user can find, read the card of, and tune on any private
model in an allowlisted namespace — including other users' `autotunex_*` outputs,
since the base-model binding pulls with the space's credentials, not per-user access
control. Accepted as-is on 2026-09-26; restricting `autotunex_*` repos to the caller's
own jobs is a possible follow-up.

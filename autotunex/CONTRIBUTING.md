# Contributing to AutoTuneX

Thanks for your interest in contributing! This document explains how to
propose changes, report problems, and get your contributions merged.

By participating in this project, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** — open a [GitHub issue](../../issues) describing what you
  expected, what happened, and how to reproduce it.
- **Request a feature** — open an issue explaining the use case before writing
  code, so we can agree on the approach.
- **Submit a fix or feature** — open a pull request (see below).
- **Report a security vulnerability** — do **not** open a public issue.
  Follow [SECURITY.md](SECURITY.md) instead.

## Developer Certificate of Origin (DCO)

All contributions must be signed off under the
[Developer Certificate of Origin](https://developercertificate.org/). This
certifies that you wrote the patch or otherwise have the right to submit it
under the project's open-source license.

Sign off each commit by adding a line to the commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

Git can add this automatically with the `-s` flag:

```bash
git commit -s -m "Your commit message"
```

## Licensing of contributions

AutoTuneX is licensed under the [Apache License 2.0](LICENSE). By submitting a
contribution, you agree that it is licensed under the same terms. New source
files should carry the standard header:

```
# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
```

(Use the comment syntax appropriate to the file's language.)

## Pull request process

1. Fork the repository and create a topic branch from `main`.
2. Make your change in small, focused commits with clear messages.
3. Sign off every commit (`git commit -s`).
4. Run the checks for the service(s) you touched (see below).
5. Open a pull request against `main`, describing the change and linking any
   related issue.
6. Address review feedback; keep the branch up to date with `main`.

## Project layout & local checks

AutoTuneX is composed of three services plus a shared MySQL database. See
[`CLAUDE.md`](CLAUDE.md) and [`README.md`](README.md) for full architecture
and setup details.

| Service    | Directory     | Local checks                                  |
|------------|---------------|-----------------------------------------------|
| API Server | `api/`        | Manual verification via Swagger at `/fmtune/try` |
| API Bridge | `api-bridge/` | Manual verification of the logging endpoints  |
| Frontend   | `ux/`         | `npm run check`, `npm run lint`               |

For the frontend, please run formatting and type/lint checks before opening a
PR:

```bash
cd ux
npm run format   # Prettier auto-format
npm run lint     # Prettier check + ESLint
npm run check    # Svelte/TypeScript type checking
```

There is currently no automated test framework for the Python services; verify
API changes manually through the Swagger UI.

## Coding conventions

- **Python 3.10+** for `api/` and `api-bridge/`.
- The frontend uses **SvelteKit + Carbon Design System** — follow existing
  Carbon patterns for new UI work.
- Match the style, naming, and structure of the surrounding code.
- Keep changes scoped; unrelated refactors belong in separate PRs.

## Questions

Open a [GitHub issue](../../issues) or start a discussion. We're happy to help.

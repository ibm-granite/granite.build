# Security Policy

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, report them privately using GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability):

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** to open a private advisory.

This keeps the report confidential between you and the maintainers until a fix
is available.

Please include as much of the following as you can, to help us triage quickly:

- The type of issue (e.g. authentication bypass, injection, information
  disclosure, credential exposure).
- The affected component — API Server (`api/`), API Bridge (`api-bridge/`), or
  Frontend (`ux/`) — and the file(s) and line(s) involved.
- Step-by-step instructions to reproduce the issue.
- Proof-of-concept or exploit code, if available.
- The potential impact, including how an attacker might exploit it.

## Response process

- We will acknowledge your report as soon as we are able.
- We will investigate, keep you informed of our progress, and let you know when
  the issue is resolved.
- We ask that you give us a reasonable amount of time to address the issue
  before any public disclosure, and that you avoid privacy violations,
  data destruction, or service disruption while researching.

## Supported versions

Security fixes are applied to the latest released version on the `main` branch.
Older versions are not guaranteed to receive backported fixes.

## Deployment hardening notes

When self-hosting AutoTuneX, review the deployment configuration for your
environment. In particular:

- Do not commit real secrets. Use the `*.env.example` templates and supply
  credentials at runtime; rotate any credential that may have been exposed.
- Restrict CORS to the specific origins your deployment serves.
- Place the API Bridge's write endpoints behind network controls or an
  authentication layer appropriate to your environment.
- Treat user-supplied reward functions as an untrusted-code trust boundary and
  run them with suitable isolation.

Thank you for helping keep AutoTuneX and its users safe.

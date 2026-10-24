# Security Policy

## Supported versions

Only the latest `main` branch of
[ibm-granite/granite.build](https://github.com/ibm-granite/granite.build)
(fm-tune lives at `autotunex/src/fm-tune`) receives security fixes.

## Reporting a vulnerability

Please **do not open a public issue** for a suspected vulnerability.

Report it privately through GitHub's private vulnerability reporting: open the
repository's [Security tab](https://github.com/ibm-granite/granite.build/security)
and choose **Report a vulnerability**. Include the affected version or commit,
steps to reproduce, and the impact you expect.

We aim to acknowledge a report within 5 business days and will keep you updated
while we investigate and prepare a fix.

## Scope notes

- **`trust_remote_code` is on by default.** Every Hugging Face `from_pretrained`
  call passes `trust_remote_code=True`, so loading a model executes any Python
  modeling code shipped in its repository. Only load models you trust, or set
  `FMTUNE_TRUST_REMOTE_CODE=0` to opt out (architectures bundled with
  `transformers` keep working).
- **Ray clusters are not authenticated.** fm-tune starts or joins a Ray cluster;
  anyone who can reach its ports can submit work to it. On shared networks, run
  Ray on an isolated network or behind a firewall.

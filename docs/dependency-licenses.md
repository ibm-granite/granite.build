# Dependency License Audit

**Project:** granite-build-tools (gbserver + gbcli monorepo)
**License:** Apache License 2.0
**Audit date:** 2026-04-10
**Tool:** [pip-licenses](https://pypi.org/project/pip-licenses/) v5.5.5
**Auditor:** Constantin Adam (issue #45 - gbcli dependency audit)

This document records the licenses of all direct dependencies to verify compatibility
with the project Apache 2.0 license, as part of open-source release preparation.

## Core Dependencies

These packages are installed with a bare pip install (no extras).

| Name             | Version      | License                    | Compatible? | Notes |
|------------------|--------------|----------------------------|-------------|-------|
| GitPython        | 3.0.6        | BSD-3-Clause (*)           | Yes         | pip show UNKNOWN; classifiers confirm BSD-3-Clause |
| Jinja2           | 3.1.6        | BSD License (*)            | Yes         | pip show blank; classifiers confirm BSD |
| PyYAML           | 6.0.3        | MIT License                | Yes         |       |
| SQLAlchemy       | 2.0.49       | MIT                        | Yes         |       |
| aiohttp          | 3.13.5       | Apache-2.0 AND MIT         | Yes         | NOTICE entries added for transitive deps |
| click            | 8.1.8        | BSD-3-Clause (*)           | Yes         | pip show blank; classifiers confirm BSD |
| dateparser       | 1.4.0        | BSD                        | Yes         |       |
| fastapi          | 0.135.3      | MIT                        | Yes         |       |
| filelock         | 3.25.2       | MIT                        | Yes         |       |
| giturlparse      | 0.14.0       | Apache v2                  | Yes         |       |
| huggingface_hub  | 1.10.1       | Apache-2.0                 | Yes         |       |
| humanize         | 4.15.0       | MIT                        | Yes         |       |
| jsonpatch        | 1.33         | Modified BSD License       | Yes         |       |
| jsonschema       | 4.26.0       | MIT                        | Yes         |       |
| openai           | 2.31.0       | Apache-2.0                 | Yes         |       |
| packaging        | 26.0         | Apache-2.0 OR BSD-2-Clause | Yes         |       |
| pandas           | 3.0.2        | BSD License                | Yes         |       |
| portalocker      | 3.2.0        | BSD-3-Clause               | Yes         | [redis] extra pulls in redis (MIT) |
| pyarrow          | 23.0.1       | Apache-2.0                 | Yes         | NOTICE entry present |
| pydantic         | 2.11.10      | MIT                        | Yes         |       |
| python-dotenv    | 1.2.2        | BSD-3-Clause               | Yes         | Verified correct package (not deprecated dotenv) |
| python-multipart | 0.0.22       | Apache-2.0                 | Yes         |       |
| pytz             | 2026.1.post1 | MIT License                | Yes         |       |
| requests         | 2.33.1       | Apache-2.0                 | Yes         | NOTICE entry present |
| rich             | 14.3.3       | MIT                        | Yes         |       |
| sqlite_database  | 0.7.15       | BSD-3-Clause               | Yes         |       |
| tabulate         | 0.10.0       | MIT                        | Yes         |       |
| tenacity         | 9.1.4        | Apache 2.0                 | Yes         |       |
| toml             | 0.10.2       | MIT                        | Yes         |       |
| tqdm             | 4.67.3       | MPL-2.0 AND MIT            | Yes         | Weak copyleft; also MIT; consumers may elect MIT |
| uvicorn          | 0.44.0       | BSD-3-Clause               | Yes         |       |
| xxhash           | 3.6.0        | BSD                        | Yes         |       |

(*) pip show reports blank or UNKNOWN for the License field; license confirmed via
Classifier: License :: OSI Approved :: BSD License in dist-info metadata.

## Optional Dependencies (Extras)

### Audited

| Extra        | Package             | Version   | License                     | Compatible? | Notes |
|--------------|---------------------|-----------|-----------------------------|-------------|-------|
| rabbitmq     | aio-pika            | 9.6.2     | Apache-2.0                  | Yes         |       |
| rabbitmq     | aiormq              | 6.9.4     | Apache-2.0                  | Yes         |       |
| nats         | nats-py             | 2.14.0    | Apache-2.0                  | Yes         |       |
| postgres     | psycopg2-binary     | 2.9.11    | LGPL with exceptions        | Yes*        | LGPL OK as unmodified library dep |
| k8s          | kubernetes_asyncio  | latest    | Apache-2.0                  | Yes         | Via PyPI metadata |
| ssh          | asyncssh            | 2.22.0    | EPL-2.0 OR GPL-2.0-or-later | Yes*        | EPL-2.0 is ASF Category B; users may elect EPL-2.0 over GPL |
| docker       | docker              | >=7.0.0   | Apache-2.0                  | Yes         | Via PyPI metadata |
| skypilot     | skypilot            | >=0.11.2  | Apache 2.0                  | Yes         |       |
| standalone   | nats-py             | 2.14.0    | Apache-2.0                  | Yes         |       |

### Not Fully Audited (Require IBM Artifactory)

| Extra       | Package                 | Known License | Notes |
|-------------|-------------------------|---------------|-------|
| ibmcloud    | ibm_cloud_sdk_core      | Apache-2.0    | Via PyPI metadata |
| ibmcloud    | ibm_secrets_manager_sdk | Apache-2.0    | Via PyPI metadata |
| lakehouse   | dmf-lib==1.10.2         | IBM Internal  | Not on public PyPI; requires IBM Artifactory |
| lakehouse   | daft                    | Apache-2.0    | Renamed from getdaft (deprecated stub with no declared license) |
| lakehouse   | aspera==v1.1.6          | IBM Internal  | Not on public PyPI; differs from public aspera 0.10.5 (BUSL-1.1) |

## Changes Made in This Audit

1. **getdaft replaced with daft** in [lakehouse] and [all] extras (pyproject.toml):
   - getdaft is a deprecated PyPI stub with no declared license that redirects to daft
   - daft (Apache-2.0) is the correct, actively maintained package

2. **NOTICE file updated** (NOTICE): Added attribution for propcache and yarl,
   which are transitive Apache-2.0 dependencies of aiohttp (a core dep) that ship
   NOTICE files of their own per Apache License Section 4(d).

## Summary

**No GPL-only dependencies exist in the core install path.** All core dependencies
use permissive licenses (MIT, BSD, Apache-2.0) or weak copyleft (MPL-2.0) that are
compatible with the project Apache 2.0 license.

### Notable Findings

- **python-dotenv** (1.2.2, BSD-3-Clause): Verified correct package. Deprecated dotenv wrapper (License: UNKNOWN) is not used.
- **tqdm** (MPL-2.0 AND MIT): MPL-2.0 is weak copyleft (file-level); also offered under MIT.
- **psycopg2-binary** ([postgres] extra, LGPL): Compatible as unmodified library dep; not in core install.
- **asyncssh** ([ssh] extra, EPL-2.0 OR GPL-2.0-or-later): EPL-2.0 is ASF Category B. Users may elect EPL-2.0. Not in core or standalone path.
- **IBM-only extras** (dmf-lib, aspera==v1.1.6, ibm_cloud_sdk_core, ibm_secrets_manager_sdk): Require IBM Artifactory; not for OSS use. Note: aspera==v1.1.6 is IBM-internal, distinct from public aspera 0.10.5 (BUSL-1.1).
- **getdaft replaced with daft**: getdaft was a deprecated stub with no declared license; daft is Apache-2.0.

## License Compatibility Reference

| License                 | Compatible? | Notes |
|-------------------------|-------------|-------|
| MIT                     | Yes         | Permissive |
| BSD (2-Clause/3-Clause) | Yes         | Permissive |
| Apache-2.0              | Yes         | Same license |
| MPL-2.0                 | Yes         | Weak copyleft, file-level; permits combination with Apache 2.0 |
| LGPL                    | Yes*        | Compatible as unmodified library dep |
| EPL-2.0                 | Yes*        | ASF Category B; users may elect EPL-2.0 over GPL secondary |
| BUSL-1.1                | No          | Not open-source; not in any public-PyPI install path |
| GPL-2.0/3.0             | No          | Strong copyleft; not present in any install path |

\* With conditions noted above.

## Source File License Notes

- src/gbserver/utils/logger.py contains # (C) Copyright IBM Corp. 2024. followed by Apache 2.0 text. Valid Apache 2.0 header; not proprietary.
- No source files contain proprietary, confidential, or All Rights Reserved headers.

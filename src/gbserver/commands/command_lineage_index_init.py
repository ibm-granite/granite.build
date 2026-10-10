#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Place the lineage *indexer* checkpoint without starting the indexer.

The indexer counterpart of ``lineage-init``. Outside standalone the indexer
indexes nothing until its checkpoint exists, so a fresh environment is set up
once, deliberately, with this command; a running indexer picks the checkpoint up
on its next scan. In standalone the indexer starts on its own from the
beginning, and this command can still seed, move or show its checkpoint.

The checkpoint is per source (``lineage_index_checkpoint:<provider>``), resolved
from ``GBSERVER_LINEAGE_PROVIDER`` exactly as the indexer resolves it.
"""

import json
from typing import Optional

import click

from gbserver.lineage.indexer import create_indexer, resolve_indexer_source
from gbserver.lineage.lineage_seeding import LineageSeedError
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.context import CliEnvironment, pass_environment


@click.command()
@click.option(
    "--base-timestamp",
    required=False,
    default=None,
    type=str,
    metavar="from-latest|all|ISO-8601",
    help=(
        "Anchor to seed: 'from-latest' starts at the newest job in the source, "
        "'all' indexes the full history (expensive first scan), any other value "
        "is an ISO-8601 timestamp (naive is local time). Jobs at or after it are "
        "indexed."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Replace an existing checkpoint instead of keeping it. Requires "
        "--base-timestamp. Moving it back re-indexes jobs (deduplicated); moving "
        "it forward skips them for good."
    ),
)
@click.option(
    "--show",
    is_flag=True,
    default=False,
    help=(
        "Print the current checkpoint and exit without writing. Must be passed "
        "alone."
    ),
)
@pass_environment
def cli(
    ctx: CliEnvironment,
    base_timestamp: Optional[str],
    force: bool,
    show: bool,
) -> None:
    """Seed the lineage indexer checkpoint (gb_kv_pairs) without running it."""
    if show and (base_timestamp is not None or force):
        conflicting = ", ".join(
            flag
            for flag, given in (
                ("--base-timestamp", base_timestamp is not None),
                ("--force", force),
            )
            if given
        )
        raise click.ClickException(
            f"--show is read-only and cannot be combined with {conflicting}; run "
            "them as separate commands."
        )
    if not show and base_timestamp is None:
        if force:
            raise click.ClickException(
                "--force only applies when seeding; pass --base-timestamp with it."
            )
        raise click.ClickException(
            "Nothing to do: pass --base-timestamp to seed the checkpoint, or --show "
            "to inspect it."
        )
    if base_timestamp is not None and not base_timestamp.strip():
        raise click.ClickException(
            "--base-timestamp was given an empty value; pass 'from-latest', 'all', "
            "or an ISO-8601 timestamp."
        )

    source = resolve_indexer_source()
    indexer = create_indexer(source)
    if indexer is None:
        raise click.ClickException(
            f"The lineage source {source!r} has no indexer, so there is no "
            "checkpoint to seed."
        )
    key = indexer._get_checkpoint_key()  # pylint: disable=protected-access
    storage = get_admin_storage()

    if show:
        existing = storage.kv_pair_storage.get_value(key)
        if not existing:
            click.echo(
                f"No lineage index checkpoint under {key}. Outside standalone the "
                "indexer indexes nothing until one is seeded."
            )
        else:
            click.echo(f"{key} = {json.dumps(existing)}")
        return

    assert base_timestamp is not None
    try:
        wrote = indexer.seed_if_absent(storage, base_timestamp.strip(), force=force)
    except LineageSeedError as exc:
        raise click.ClickException(str(exc)) from exc

    value = storage.kv_pair_storage.get_value(key)
    if wrote:
        click.echo(f"Seeded {key} = {json.dumps(value)}")
    else:
        click.echo(
            f"{key} already set to {json.dumps(value)}; kept it. Pass --force to "
            "replace it."
        )

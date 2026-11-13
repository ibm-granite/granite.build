# Copyright Granite.Build Authors
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

import click


@click.group("dataset")
@click.pass_context
def cli(ctx):
    """
    Operate datasets.
    In this version, please use 'gb artifact' for dataset operations.
    """
    pass


@cli.command()
@click.pass_context
def search(ctx):
    """Search within or across datasets"""
    click.echo("This command is not yet available.")
    ctx.exit(1)


@cli.command()
@click.pass_context
def diff(ctx):
    """Compare documents between datasets or compare individual documents before/after processing"""
    click.echo("This command is not yet available.")
    ctx.exit(1)


@cli.command(name="import")
@click.pass_context
def dataset_import(ctx):
    """Import"""
    click.echo("This command is not yet available.")
    ctx.exit(1)


@cli.command()
@click.pass_context
def list(ctx):
    """List"""
    click.echo("This command is not yet available.")
    ctx.exit(1)


@cli.command()
@click.pass_context
def move(ctx):
    """Move to different artifact store, update content URI"""
    click.echo("This command is not yet available.")
    ctx.exit(1)


@cli.command()
@click.pass_context
def download(ctx):
    """Download"""
    click.echo("This command is not yet available.")
    ctx.exit(1)

import click
import json
import sys
import time
import webbrowser
from tabulate import tabulate
from tqdm import tqdm
from typing import Dict

from gbcli.client.client import GBClient
from gbcli.commands.command_auth import execute_with_spinner, str_exc_chain
from gbcli.utils.gbcredentials import get_user_token
from gbcli.commands.common_options import common_options
from gbcli.services.service_auth import verify_rits_api_key
from gbcli.utils.gbconstants import (
    MODEL_LIST_HEADERS,
    MODEL_LIST_URI_HEADERS,
    PROJECT_NAME,
    RITS_MAX_TOKENS,
    RITS_TEMP,
    RITS_TOP_P,
    RITS_URL,
)
from gbcli.utils.utils import check_runnable_browser, get_standard_model_prompt
from gbcli.utils.versionutil import check_current_and_latest_versions


@click.group("model")
@click.pass_context
def cli(ctx):
    """Work with deployed models"""
    pass


@cli.command()
@click.pass_context
@click.option(
    "--byom",
    is_flag=True,
    default=False,
    help=f"Only BYOM models",
)
@click.option("--uri", is_flag=True, default=False, help=f"List with RITS Base URIs")
@click.option(
    "--format",
    "format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help="Output format: plain (default), json",
)
@common_options
def list(ctx, byom, uri, format, skip_version_check, quiet):
    """List standard and BYOM checkpoints deployed in RITS."""
    if format == "json":
        quiet = True
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if not quiet:
        click.echo(f"(1/3) Looking for user's RITS_API_KEY in variables")
    auth_client = GBClient.Auth()
    rits_api_key = auth_client.rits_user_api_key()

    if rits_api_key is None:
        click.echo(
            f"RITS_API_KEY is not set. Go to {RITS_URL} to obtain the API Key (“RITS_API_KEY”) and set it in your environment."
        )
        if check_runnable_browser():
            answer = click.confirm("Open the browser?", False)
            if answer:
                webbrowser.open(RITS_URL)
        ctx.exit(1)

    if not quiet:
        click.echo(f"(2/3) Verifying RITS_API_KEY")

    def echo_callback(callback_event: str, callback_args: Dict):
        if callback_event == "error":
            reason = callback_args.get("reason", "")
            click.echo(
                f"❌ Models can't be retrieved at this moment... Reason: {reason}",
                err=True,
            )
            ctx.exit(1)  # Exit with a non-zero status
        else:
            pass  # Ignore unknown events

    verified = verify_rits_api_key(rits_api_key, callback=echo_callback)

    if not verified:
        click.echo(
            f"Your API Key is not valid. Please go to {RITS_URL} and obtain a new key.",
            err=True,
        )
        ctx.exit(1)
    if not quiet:
        click.echo("RITS_API_KEY verified successfully!")

    model_client = GBClient.Model(get_user_token())

    try:
        if quiet:
            m = model_client.get_rits_models(rits_api_key, callback=None)
        else:
            with tqdm(
                total=100,
                miniters=1,
                desc=f"(3/3)📝 Listing deployed {'BYOM ' if byom else ''}RITS models",
                bar_format="{desc} [{bar}] {percentage:3.0f}% {postfix}",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    steps = callback_args.get("steps", 0)
                    match callback_event:
                        case "set_total":
                            progress_bar.total = callback_args.get("total", 0)
                        case "listing_models":
                            progress_bar.update(steps)
                        case "listed_models":
                            progress_bar.close()
                            progress_bar.write(
                                f"(3/3) 📝 Listing deployed {'BYOM' if byom else ''}RITS models"
                            )
                        case "error":
                            reason = callback_args.get("reason", "")
                            click.echo(
                                f"\n❌ Models can't be retrieved at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status
                        case _:
                            pass

                m = model_client.get_rits_models(rits_api_key, callback=update_bar)

        if byom:
            m = {key: m[key] for key in m if "byom" in key}

        models = []
        for key in m:
            if uri:
                models.append([key.split(":")[1], m[key]])
            else:
                models.append([key.split(":")[0], key.split(":")[1]])

        if format == "json":
            click.echo(
                json.dumps(
                    [{"name": key.split(":")[1], "url": url} for key, url in m.items()]
                )
            )
        else:
            model_table = tabulate(
                models,
                MODEL_LIST_URI_HEADERS if uri else MODEL_LIST_HEADERS,
                tablefmt="plain",
            )
            click.echo("\n" + model_table)

        return
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo(f"❌ Model list failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status


class MultipleMatchException(Exception):
    pass


@cli.command()
@click.pass_context
@click.argument("msg", required=True)
@click.option(
    "--model", "-m", help="Model name or full model id in RITS", required=True
)
@click.option("--temp", help="Temperature", type=float, required=False)
@click.option("--max", help="Max output tokens", type=int, required=False)
@click.option("--top_p", help="top_p", type=float, required=False)
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@common_options
def prompt(
    ctx,
    msg: str,
    model: str,
    temp: float,
    max: int,
    top_p: float,
    format: str,
    skip_version_check: bool,
    quiet: bool,
):
    """Submit one prompt to a model."""
    if format == "json":
        quiet = True
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if not quiet:
        click.echo(f"(1/3) Looking for user's RITS_API_KEY in variables")
    auth_client = GBClient.Auth()
    rits_api_key = auth_client.rits_user_api_key()

    if rits_api_key is None:
        click.echo(
            f"RITS_API_KEY is not set. Go to {RITS_URL} to obtain the API Key (“RITS_API_KEY”) and set it in your environment."
        )
        if check_runnable_browser():
            answer = click.confirm("Open the browser?", False)
            if answer:
                webbrowser.open(RITS_URL)
        ctx.exit(1)

    if not quiet:
        click.echo(f"(2/3) Verifying RITS_API_KEY")

    def echo_callback(callback_event: str, callback_args: Dict):
        if callback_event == "error":
            reason = callback_args.get("reason", "")
            click.echo(
                f"❌ Model prompt can't be executed at this moment... Reason: {reason}",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status
        else:
            pass  # Ignore unknown events

    verified = verify_rits_api_key(rits_api_key, callback=echo_callback)
    if not verified:
        click.echo(
            f"Your API Key is not valid. Please go to {RITS_URL} and obtain a new key."
        )
        ctx.exit(1)
    if not quiet:
        click.echo("RITS_API_KEY verified successfully!")

    model_client = GBClient.Model(get_user_token())

    try:
        matching_model = model_client.lookup_model(rits_api_key, model)

        # if multiple matches
        if type(matching_model) is type([]):
            raise MultipleMatchException(
                f'"{model}" matches multiple models, listed above. Please specify enough of the unique name for a single match.'
            )

        url, full_model_id = matching_model
        if not quiet:
            click.echo(f"(3/3) Prompting {full_model_id}")

        response = (
            model_client.prompt_model(
                rits_api_key=rits_api_key,
                prompt=msg,
                url=url,
                model_id=full_model_id,
                temp=temp if temp else RITS_TEMP,
                max=max if max else RITS_MAX_TOKENS,
                top_p=top_p if top_p else RITS_TOP_P,
                callback=echo_callback,
            )
            if quiet
            else execute_with_spinner(
                model_client.prompt_model,
                rits_api_key=rits_api_key,
                prompt=msg,
                url=url,
                model_id=full_model_id,
                temp=temp if temp else RITS_TEMP,
                max=max if max else RITS_MAX_TOKENS,
                top_p=top_p if top_p else RITS_TOP_P,
                callback=echo_callback,
            )
        )

        if format == "json":
            click.echo(
                json.dumps(
                    {
                        "model": full_model_id,
                        "response": response.choices[0].text.strip(),
                    }
                )
            )
        else:
            click.echo(
                click.style("\n\n" + response.choices[0].text.strip(), fg="blue")
            )

    except MultipleMatchException as e:
        matches = [
            [match.split(":")[0], match.split(":")[1]] for match in matching_model
        ]
        match_table = tabulate(
            matches,
            ["MODEL NAME", "MODEL ID"],
            tablefmt="plain",
        )
        click.echo("\n" + match_table + "\n", err=True)
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo(f"❌ Model prompt failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.option(
    "--model", "-m", help="Model name or full model id in RITS", required=True
)
@click.option("--system", help="System prompt override", required=False)
@click.option("--temp", help="Temperature", type=float, required=False)
@click.option("--max", help="Max output tokens", type=int, required=False)
@click.option("--top_p", help="top_p", type=float, required=False)
@click.option("--chat_template", help="chat_template", type=str, required=False)
@common_options
def chat(
    ctx,
    model: str,
    system: str,
    temp: float,
    max: int,
    top_p: float,
    chat_template: str,
    skip_version_check: bool,
    quiet: bool,
):
    """Start interactive chat with a model."""
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    click.echo(f"(1/3) Looking for user's RITS_API_KEY in variables")
    auth_client = GBClient.Auth()
    rits_api_key = auth_client.rits_user_api_key()

    if rits_api_key is None:
        click.echo(
            f"RITS_API_KEY is not set. Go to {RITS_URL} to obtain the API Key (“RITS_API_KEY”) and set it in your environment."
        )
        if check_runnable_browser():
            answer = click.confirm("Open the browser?", False)
            if answer:
                webbrowser.open(RITS_URL)
        ctx.exit(1)

    click.echo(f"(2/3) Verifying RITS_API_KEY")

    def echo_callback(callback_event: str, callback_args: Dict):
        if callback_event == "error":
            reason = callback_args.get("reason", "")
            click.echo(
                f"❌ Model prompt can't be executed at this moment... Reason: {reason}",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status
        else:
            pass  # Ignore unknown events

    verified = verify_rits_api_key(rits_api_key, callback=echo_callback)
    if not verified:
        click.echo(
            f"Your API Key is not valid. Please go to {RITS_URL} and obtain a new key."
        )
        ctx.exit(1)
    click.echo("RITS_API_KEY verified successfully!")

    model_client = GBClient.Model(get_user_token())

    try:
        matching_model = model_client.lookup_model(rits_api_key, model)

        # if multiple matches
        if type(matching_model) is type([]):
            raise MultipleMatchException(
                f'"{model}" matches multiple models, listed above. Please specify enough of the unique name for a single match.'
            )

        url, full_model_id = matching_model
        model_id = full_model_id.split(":")[1]

        messages = (
            [{"role": "system", "content": system}]
            if system
            else get_standard_model_prompt()
        )

        click.echo("(3/3) Starting model chat")
        click.echo(
            click.style(
                f"\n💬 {PROJECT_NAME} RITS chat using {full_model_id}",
                fg="blue",
                bold=True,
            )
        )
        click.echo("Enter your message, or press Ctrl+C/'quit'/'exit' to end.")

        while True:
            text = click.style("\nYou", fg="magenta", bold=True)
            user_input = click.prompt(
                text + "\033[1;37;35m", type=str
            )  # User input in bold and in color
            click.echo(f"\033[0m")  # Reset textstyle

            if user_input is None or (
                user_input and user_input.lower() in ["quit", "exit", "bye"]
            ):
                break

            if not user_input or not user_input.strip():
                continue

            messages.append({"role": "user", "content": f"{user_input}"})
            start_time = time.time()
            response = execute_with_spinner(
                model_client.model_chat,
                rits_api_key=rits_api_key,
                url=url,
                model_id=full_model_id,
                messages=messages,
                temp=temp if temp else RITS_TEMP,
                max=max if max else RITS_MAX_TOKENS,
                top_p=top_p if top_p else RITS_TOP_P,
                callback=echo_callback,
                chat_template=chat_template,
            )
            end_time = time.time()
            elapsed_secs = end_time - start_time
            messages.append({"role": "assistant", "content": response})

            click.echo(
                click.style(
                    "\r\r" + f"{elapsed_secs:.2f}secs - Model:", fg="blue", bold=True
                )
                + f" {response}"
            )

    except MultipleMatchException as e:
        matches = [
            [match.split(":")[0], match.split(":")[1]] for match in matching_model
        ]
        match_table = tabulate(
            matches,
            ["MODEL NAME", "MODEL ID"],
            tablefmt="plain",
        )
        click.echo("\n" + match_table + "\n", err=True)
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo(f"❌ Model chat failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status

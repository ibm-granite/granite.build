import itertools
import threading
import time
import webbrowser

import click
from fastapi import HTTPException
from requests.exceptions import ConnectionError

from gbcli.client import GBClient
from gbcli.utils.click_utils import FileOrStringParamType
from gbcli.utils.gbconstants import PROJECT_NAME, is_standalone
from gbcli.utils.gbcredentials import GBCredentials
from gbcli.utils.utils import check_runnable_browser
from gbcommon.types.constants import get_gh_credentials_section

# Maps user-facing provider names to internal credential keys.
# Multiple synonyms may map to the same internal value.
_PROVIDER_TO_INTERNAL = {
    "github": "github",
    "sso": "ibmid",
    "ibmid": "ibmid",
    "gbserver": "apikey",
    "apikey": "apikey",
}
# Canonical display name for each internal value.
_INTERNAL_TO_DISPLAY = {"github": "github", "ibmid": "sso", "apikey": "apikey"}


def str_exc_chain(exc) -> str:
    result = str(exc)
    if exc.__cause__:
        result += " caused by: " + str_exc_chain(exc.__cause__)
        # print('\nThe above exception was the direct cause...')
    elif exc.__context__:
        result += " context: " + str_exc_chain(exc.__context__)
        # print('\nDuring handling of the above exception, ...')
    return result


def execute_with_spinner(fn, *args, **kwargs):
    """
    Runs a function with a spinner while waiting for it to complete.
    - `fn`: The function to execute.
    - `*args`: Positional arguments for `fn`.
    - `**kwargs`: Keyword arguments for `fn`.
    """
    stop_event = threading.Event()

    # Start spinner in a separate thread
    spinner_thread = threading.Thread(target=spinner_running, args=(stop_event,))
    spinner_thread.start()

    try:
        result = fn(*args, **kwargs)  # Call the provided function

    finally:
        stop_event.set()  # Stop spinner thread
        spinner_thread.join()  # Ensure spinner stops
    return result


def spinner_running(stop_event):
    """Displays a spinner until `stop_event` is set."""
    spinner = itertools.cycle(["-", "\\", "|", "/"])  # Spinner characters
    while not stop_event.is_set():
        click.echo(f"\rProcessing... {next(spinner)}", nl=False)
        time.sleep(0.1)


@click.group("auth")
@click.pass_context
def cli(ctx):
    """Authenticate a user"""
    # ctx.log(f"Authenticate a user")
    pass


@cli.command()
@click.option(
    "--token",
    is_flag=False,
    flag_value="flag",
    help="Provide a GitHub token value. If ommitted, a new one will be generated for you.",
    type=FileOrStringParamType(),
)
@click.option(
    "--gbserver",
    is_flag=True,
    default=False,
    help="Login to a standalone gbserver using API credentials (GBSERVER_API_USER / GBSERVER_API_KEY).",
)
@click.option(
    "--sso",
    is_flag=False,
    flag_value="ibm",
    type=click.Choice(["ibm"], case_sensitive=False),
    default=None,
    help="Login via SSO provider (default: ibm). Use 'ibm' for IBMid OIDC authentication.",
)
@click.pass_context
def login(ctx, token, gbserver, sso):
    """Login"""
    # Validate mutually exclusive options
    selected = sum([bool(token), gbserver, bool(sso)])
    if selected > 1:
        click.echo(
            "Error: --token, --gbserver, and --sso are mutually exclusive.",
            err=True,
        )
        ctx.exit(1)
        return

    auth_client = GBClient.Auth()

    # ---- IBMid SSO flow ----
    if sso == "ibm":
        click.echo(f"🔒 {PROJECT_NAME} IBMid SSO login")
        try:
            runnable_browser = check_runnable_browser()

            def _open_browser(url):
                if runnable_browser and click.confirm(
                    "Open the browser?", default=True
                ):
                    webbrowser.open(url)
                else:
                    click.echo(f"Open this link in the browser: {url}")
                click.echo("Waiting for IBMid authentication...")

            user_login = auth_client.login_ibmid(open_browser=_open_browser)
            click.echo(f"Logged in as " + click.style(user_login, bold=True))
            click.echo("✅ IBMid authentication is successful!")
        except Exception as e:
            click.echo(f"\n{str_exc_chain(e)}", err=True)
            click.echo("❌ IBMid authentication failed!", err=True)
            ctx.exit(1)
        return

    # ---- gbserver API key flow ----
    if gbserver:
        click.echo(f"🔒 {PROJECT_NAME} gbserver login")
        try:
            api_user = click.prompt("GBSERVER_API_USER", type=str)
            api_key = click.prompt(
                "GBSERVER_API_KEY", type=str, hide_input=True, confirmation_prompt=True
            )
            user_login = auth_client.login_gbserver(api_user, api_key)
            click.echo(f"Logged in as " + click.style(user_login, bold=True))
            click.echo("✅ gbserver authentication saved!")
        except Exception as e:
            click.echo(f"\n{str_exc_chain(e)}", err=True)
            click.echo("❌ gbserver authentication failed!", err=True)
            ctx.exit(1)
        return

    # ---- GitHub flow (default) ----
    click.echo(f"🔒 {PROJECT_NAME} login")

    try:
        if token and token == "flag":
            new_token = click.prompt(
                "Please enter a GitHub token",
                type=str,
                hide_input=True,
                confirmation_prompt=True,
            )
            click.echo("Initiating IBM GitHub Token verification")
            user_login = auth_client.login_github_with_token(new_token)

            click.echo(f"Logged in as " + click.style(user_login, bold=True))
        elif token and len(token) > 0:
            user_login = auth_client.login_github_with_token(token)

            click.echo(f"Logged in as " + click.style(user_login, bold=True))
        else:
            click.echo("(1/2) Initiating IBM GitHub authorization")
            user_token = auth_client.github_token()

            runnable_browser = check_runnable_browser()

            click.echo(
                f"Open this link in the browser {'on your local environment' if not runnable_browser else ''}: {user_token.verification_uri}"
            )
            click.echo(
                f"and enter this code: " + click.style(user_token.user_code, bold=True)
            )

            if runnable_browser:
                answer = click.confirm("Open the browser?", True)

                if answer:
                    # https://docs.python.org/3/library/webbrowser.html
                    webbrowser.open(user_token.verification_uri)
                else:
                    click.echo(f"Waiting for device code verification...")
            else:
                click.echo(f"Waiting for device code verification...")

            user_login = auth_client.login_github(user_token.device_code)

            click.echo(f"(2/2) Logged in as " + click.style(user_login, bold=True))

        click.echo("✅ IBM GitHub authentication is successful!")
    except HTTPException as e:
        click.echo(f"\n❌ github returned '{e.status_code} {e.detail}'", err=True)
    except ConnectionError:
        click.echo(
            f"\n❌ Error: Unable to connect to network. Please check network connection.",
            err=True,
        )
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo("❌ IBM GitHub authentication failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command("provider")
@click.option(
    "--set",
    "set_provider",
    type=click.Choice(
        ["github", "sso", "ibmid", "gbserver", "apikey"], case_sensitive=False
    ),
    default=None,
    help="Set the default authentication provider (sso/ibmid for IBMid, gbserver/apikey for API key).",
)
@click.pass_context
def provider(ctx, set_provider):
    """Show or change the default authentication provider."""
    creds = GBCredentials()

    if set_provider is None:
        # Show current provider
        if is_standalone():
            raw = "apikey"
        else:
            raw = creds.get("default_provider", section="user") or "github"
        display_name = _INTERNAL_TO_DISPLAY.get(raw, raw)
        login = _get_provider_login(creds, raw)
        click.echo(f"Provider: {display_name}")
        if login:
            click.echo(f"Login:    {login}")
        return

    # --set: switch provider
    internal = _PROVIDER_TO_INTERNAL[set_provider.lower()]

    # Verify credentials exist for the target provider
    if internal == "ibmid" and not creds.check_ibmid_values():
        click.echo(
            "Error: No IBMid credentials found. Run 'auth login --sso' first.",
            err=True,
        )
        ctx.exit(1)
        return
    if internal == "github" and not _has_github_credentials(creds):
        click.echo(
            "Error: No GitHub credentials found. Run 'auth login' first.",
            err=True,
        )
        ctx.exit(1)
        return
    if internal == "apikey" and not creds.check_gbserver_values():
        click.echo(
            "Error: No gbserver credentials found. Run 'auth login --gbserver' first.",
            err=True,
        )
        ctx.exit(1)
        return

    creds.set("default_provider", internal, section="user")
    creds.save()
    display_name = _INTERNAL_TO_DISPLAY.get(internal, internal)
    click.echo(f"Default provider set to: {display_name}")


def _get_provider_login(creds: GBCredentials, internal_provider: str) -> str:
    """Return the login name for the given internal provider, or empty string."""
    if internal_provider == "ibmid":
        return creds.get("login", section="user.ibmid") or ""
    if internal_provider == "github":
        return creds.get("login", section=get_gh_credentials_section()) or ""
    if internal_provider == "apikey":
        return creds.get("login", section="user.gbserver") or ""
    return ""


def _has_github_credentials(creds: GBCredentials) -> bool:
    """Check if GitHub credentials exist (without network validation)."""
    gh_section = get_gh_credentials_section()
    fields = [
        creds.get("token", section=gh_section),
        creds.get("login", section=gh_section),
        creds.get("email", section=gh_section),
    ]
    return all(val is not None for val in fields)

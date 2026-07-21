import subprocess

from gbmcp.utils.auth import get_github_token


def run_gbcli(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a gbcli command authenticated with the current request's GitHub token."""
    token = get_github_token()
    if token is not None:
        subprocess.run(
            ["gbcli", "auth", "login", "--token", token],
            check=True,
            capture_output=True,
        )
    return subprocess.run(cmd, check=True, capture_output=True, text=True)

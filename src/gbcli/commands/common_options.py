import click
from functools import wraps


def common_options(f):
    @wraps(f)
    @click.option(
        "--skip-version-check",
        is_flag=True,
        default=False,
        help="Skip current version check.",
    )
    @click.option(
        "--quiet",
        "-q",
        is_flag=True,
        default=False,
        help="Enables quiet mode.",
    )
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper

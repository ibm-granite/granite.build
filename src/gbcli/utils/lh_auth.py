import logging
import os

from gbcli.utils.gbconstants import LAKEHOUSE_ENVIRONMENT


class AuthException(Exception):
    """Custom exception for authentication errors."""

    pass


def getLH(token):
    # remove user set environment variable
    if "LAKEHOUSE_ENVIRONMENT" in os.environ:
        os.environ.pop("LAKEHOUSE_ENVIRONMENT")
    try:
        from lakehouse import LakehouseIceberg  # type: ignore
        from lakehouse.api import ConfigMap  # type: ignore

        # Reduce verbose DMF-library warnings
        logging.getLogger("lakehouse").setLevel(logging.ERROR)

        return LakehouseIceberg(
            config="map",
            conf_map=ConfigMap(
                token=token,
                environment=LAKEHOUSE_ENVIRONMENT,
            ),
        )

    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access.",
        )
    except Exception as e:
        message = str(e)
        if "Token is expired" in message:
            raise AuthException(
                f"Error: Lakehouse token is expired! Please, provide a new one."
            ) from None
        else:
            raise e

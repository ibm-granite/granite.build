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

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


"""
Send metrics to a backend for analytics and display in a  dashboard.
"""

from enum import StrEnum, auto
from typing import Any, List, Optional, Self, Union

import requests

from gbserver.types.constants import (
    GBSERVER_METRICS_AUTH_TOKEN,
    GBSERVER_METRICS_ENDPOINT,
    PUSH_METRICS_TIMEOUT,
)
from gbserver.types.metrics import Metric, Metrics
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsClient:
    """
    A client to connect to the metrics endpoint.
    """

    metrics_endpoint: str
    token: str

    def __init__(self: Self, metrics_endpoint: str, token: str = "") -> None:
        self.metrics_endpoint = metrics_endpoint
        self.token = token
        assert self.metrics_endpoint, f"invalid metrics_endpoint: {metrics_endpoint}"

    def push_metrics(self: Self, metrics: Union[Metrics, List[Metric]]) -> None:
        """Push one or more metrics to the metrics endpoint."""
        if isinstance(metrics, list):
            metrics = Metrics(metrics=metrics)
        data = metrics.model_dump(exclude_none=True)
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = requests.post(
            url=self.metrics_endpoint,
            json=data,
            headers=headers,
            timeout=PUSH_METRICS_TIMEOUT,
        )
        resp.raise_for_status()

    def push_metric(self: Self, metric: Metric) -> None:
        """Push a single metric to the metrics endpoint."""
        self.push_metrics(metrics=[metric])


_SINGLETON_METRICS_CLIENT: Optional[MetricsClient] = None


def get_metrics_client() -> Optional[MetricsClient]:
    """Get the singleton metrics client"""
    global _SINGLETON_METRICS_CLIENT
    if not _SINGLETON_METRICS_CLIENT:
        if GBSERVER_METRICS_ENDPOINT:
            _SINGLETON_METRICS_CLIENT = MetricsClient(
                metrics_endpoint=GBSERVER_METRICS_ENDPOINT,
                token=GBSERVER_METRICS_AUTH_TOKEN,
            )
    return _SINGLETON_METRICS_CLIENT


def push_metrics(metrics: List[Metric]) -> None:
    """Push some metrics to the endpoint."""
    metrics_client = get_metrics_client()
    if not metrics_client:
        logger.info("no metrics endpoint configured, skip pushing: %s", metrics)
        return
    try:
        metrics_client.push_metrics(metrics=metrics)
    except Exception as e:
        logger.error("failed to push metrics, error: %s", e)

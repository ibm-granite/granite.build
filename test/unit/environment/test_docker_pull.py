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

"""Docker._pull_image surfaces in-stream pull errors and retries transient ones.

``client.images.pull`` drains the pull stream without checking it, so a
registry error mid-pull used to surface only as ImageNotFound from its
follow-up lookup.
"""

from unittest.mock import MagicMock

import pytest
import requests
import urllib3

docker = pytest.importorskip("docker")

from gbserver.environment import docker as docker_env  # noqa: E402
from gbserver.environment.docker import Docker, ImagePullError  # noqa: E402

_IMAGE = "registry.example/library/alpine:3.22"


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(docker_env, "DOCKER_PULL_RETRY_BASE_DELAY", 0)
    monkeypatch.setattr(docker_env, "DOCKER_PULL_RETRY_MAX_DELAY", 0)


def _client(*pull_results, lands=True):
    """Mock client whose api.pull yields *pull_results*, one per attempt.

    Each result is a list of stream chunks, or an exception to raise. The image
    is absent until a stream finishes cleanly, then present if *lands*.
    """
    present = False

    def _stream(chunks):
        nonlocal present
        yield from chunks
        present = lands

    def _get(_image):
        if not present:
            raise docker.errors.ImageNotFound("absent")
        return MagicMock()

    client = MagicMock()
    client.images.get.side_effect = _get
    client.api.pull.side_effect = [
        r if isinstance(r, BaseException) else _stream(r) for r in pull_results
    ]
    return client


def _pull(client, policy="if-not-present"):
    env = object.__new__(Docker)
    env._pull_image(client, docker, _IMAGE, policy)


_OK = [{"status": "Pulling"}, {"status": "Downloaded newer image"}]


def test_clean_stream_pulls_once():
    client = _client(_OK)
    _pull(client)
    client.api.pull.assert_called_once()
    args, kwargs = client.api.pull.call_args
    assert args == (_IMAGE,)
    assert kwargs["stream"] and kwargs["decode"]


def test_always_policy_pulls_present_image():
    client = _client(_OK)
    _pull(client, policy="always")
    client.api.pull.assert_called_once()


@pytest.mark.parametrize(
    "message",
    [
        "manifest for alpine:3.22 not found: manifest unknown",
        "no matching manifest for linux/arm64 in the manifest list entries",
        "pull access denied for alpine",
    ],
)
def test_permanent_in_stream_error_is_not_retried(message):
    client = _client([{"status": "Pulling"}, {"error": message}])
    with pytest.raises(ImagePullError, match=message):
        _pull(client)
    client.api.pull.assert_called_once()


def test_transient_in_stream_error_is_retried():
    client = _client(
        [{"errorDetail": {"message": "toomanyrequests: Rate exceeded"}}], _OK
    )
    _pull(client)
    assert client.api.pull.call_count == 2


def test_transient_errors_exhaust_attempts():
    rate_limited = [{"error": "toomanyrequests: Rate exceeded"}]
    client = _client(*([rate_limited] * docker_env.DOCKER_PULL_MAX_ATTEMPTS))
    with pytest.raises(ImagePullError, match="toomanyrequests"):
        _pull(client)
    assert client.api.pull.call_count == docker_env.DOCKER_PULL_MAX_ATTEMPTS


def test_stream_ending_without_image_is_retried():
    attempts = docker_env.DOCKER_PULL_MAX_ATTEMPTS
    client = _client(*([[]] * attempts), lands=False)
    with pytest.raises(ImagePullError, match="image is missing"):
        _pull(client)
    assert client.api.pull.call_count == attempts


@pytest.mark.parametrize(
    "exc",
    [
        docker.errors.APIError("server error", response=MagicMock(status_code=500)),
        docker.errors.APIError("rate", response=MagicMock(status_code=429)),
        docker.errors.APIError("blob not found", response=MagicMock(status_code=502)),
        requests.exceptions.ConnectionError("reset"),
        urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead"),
    ],
)
def test_transient_request_errors_are_retried(exc):
    client = _client(exc, _OK)
    _pull(client)
    assert client.api.pull.call_count == 2


def test_client_error_status_is_not_retried():
    exc = docker.errors.APIError("denied", response=MagicMock(status_code=403))
    client = _client(exc)
    with pytest.raises(docker.errors.APIError):
        _pull(client)
    client.api.pull.assert_called_once()


def test_present_image_is_not_pulled():
    client = MagicMock()
    _pull(client)
    client.api.pull.assert_not_called()

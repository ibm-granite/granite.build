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

"""Docker image names the tests use, read from the test-data image manifests."""

from functools import cache
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "test-data" / "ci-docker-images.env"
NIGHTLY_MANIFEST = MANIFEST.with_name("nightly-docker-images.env")


@cache
def ci_docker_images(manifest: Path = MANIFEST) -> dict[str, str]:
    """Return a manifest as {NAME: image}; raise ValueError on a malformed line."""
    images = {}
    for raw in manifest.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, image = (part.strip() for part in line.partition("="))
        if not (sep and name and image):
            raise ValueError(f"{manifest}: expected NAME=image, got {raw!r}")
        images[name] = image
    return images


def ci_docker_image(name: str) -> str:
    """Return the image for *name*, e.g. ``ci_docker_image("ALPINE_IMAGE")``."""
    return ci_docker_images()[name]

"""Docker image names the tests use, read from the test-data image manifests.

``ci-docker-images.env`` lists the images CI preloads; ``nightly-docker-images.env``
the extra ones only the nightly extended tests use.
"""

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

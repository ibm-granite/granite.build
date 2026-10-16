"""Docker image names the tests use, read from test-data/ci-docker-images.env."""

from functools import cache
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "test-data" / "ci-docker-images.env"


@cache
def ci_docker_images() -> dict[str, str]:
    """Return the manifest as {NAME: image}; raise ValueError on a malformed line."""
    images = {}
    for raw in MANIFEST.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, image = (part.strip() for part in line.partition("="))
        if not (sep and name and image):
            raise ValueError(f"{MANIFEST}: expected NAME=image, got {raw!r}")
        images[name] = image
    return images


def ci_docker_image(name: str) -> str:
    """Return the image for *name*, e.g. ``ci_docker_image("ALPINE_IMAGE")``."""
    return ci_docker_images()[name]

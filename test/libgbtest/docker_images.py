"""Docker image names the tests use, read from test-data/ci-docker-images.env."""

from functools import cache
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "test-data" / "ci-docker-images.env"


@cache
def ci_docker_images() -> dict[str, str]:
    """Return the manifest as a {NAME: image} dict, skipping comments and blanks."""
    images = {}
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, image = line.partition("=")
        images[name.strip()] = image.strip()
    return images


def ci_docker_image(name: str) -> str:
    """Return the image for *name*, e.g. ``ci_docker_image("ALPINE_IMAGE")``."""
    return ci_docker_images()[name]

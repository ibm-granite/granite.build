"""Check the user-facing copies of test image names against the manifest.

The Dockerfile and step.yaml that demo-standalone.sh uses spell out the image so
they work without the test harness. CI preloads only manifest images, so a copy
that drifts would silently go back to pulling from the registry.
"""

import yaml
from libgbtest.docker_images import MANIFEST, ci_docker_image, ci_docker_images

SPACE = MANIFEST.parent / "standalone-environments"


def test_manifest_parses():
    assert ci_docker_images()


def test_dockerfile_cpu_test_from_matches_manifest():
    dockerfile = SPACE / "docker" / "Dockerfile.cpu.test"
    froms = [
        line.split()[1]
        for line in dockerfile.read_text().splitlines()
        if line.strip().upper().startswith("FROM ")
    ]
    assert froms == [ci_docker_image("PYTHON_IMAGE")]


def test_inference_cpu_launcher_image_matches_manifest():
    step = yaml.safe_load((SPACE / "steps" / "inference" / "step.yaml").read_text())
    launcher = step["environment_configs"]["Docker"]["launchers"]["inference-cpu"]
    assert launcher["config"]["image"] == ci_docker_image("PYTHON_IMAGE")

"""Keep user-facing copies of the test image names in sync with the manifest.

The Dockerfile and step.yaml that scripts/demo-standalone.sh uses must work
without the test harness, so they spell out the image instead of reading
test-data/ci-docker-images.env. CI only preloads what the manifest lists, so a
copy that drifts would quietly go back to pulling from the registry.
"""

import yaml
from libgbtest.docker_images import MANIFEST, ci_docker_image

SPACE = MANIFEST.parent / "standalone-environments"


def test_manifest_entries_are_well_formed():
    for raw in MANIFEST.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            name, sep, image = line.partition("=")
            assert sep and name.strip() and image.strip(), f"bad line: {raw!r}"


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

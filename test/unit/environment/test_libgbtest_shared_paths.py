# SPDX-License-Identifier: Apache-2.0

"""Unit tests for libgbtest.shared_paths, which guards the build tests' ``rm -rf``."""

import pytest
import yaml
from libgbtest.shared_paths import require_below_shared, step_output_dir


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/shared/mergekit-test/kube", "/shared/mergekit-test/kube"),
        ("/shared/x/", "/shared/x"),
    ],
)
def test_paths_below_shared_are_accepted(path, expected):
    assert require_below_shared(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "/shared",
        "/shared/",
        "/shared/.",
        "/shared/a/..",
        "/shared/../etc",
        "/shared//x",
        "//shared/x",
        "shared/x",
        "/sharedx/y",
        "/",
        "",
    ],
)
def test_anything_else_is_refused(path):
    """``/shared`` itself, paths leaving it, relative or oddly spelled paths."""
    with pytest.raises(ValueError, match="strictly below /shared/"):
        require_below_shared(path)


def _build_yaml(tmp_path, output_path):
    doc = {
        "granite.build": {
            "targets": {
                "merge": {
                    "steps": [
                        {"config": {"mergekit_config": {"output_path": output_path}}}
                    ]
                }
            }
        }
    }
    path = tmp_path / "build.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_step_output_dir_is_the_outputs_parent(tmp_path):
    build = _build_yaml(tmp_path, "/shared/mergekit-test/kube/model")
    got = step_output_dir(build, "merge", 0, "mergekit_config", "output_path")
    assert got == "/shared/mergekit-test/kube"


def test_step_output_dir_refuses_an_output_directly_in_shared(tmp_path):
    """Its parent would be /shared itself: never cleared wholesale."""
    build = _build_yaml(tmp_path, "/shared/model")
    with pytest.raises(ValueError):
        step_output_dir(build, "merge", 0, "mergekit_config", "output_path")

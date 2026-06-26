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

"""Unit tests for the inline SkyPilot config materialization helpers.

Pure-filesystem: ``home``/``tmp_root`` point at ``tmp_path`` so the real home
directory is never touched, and no ``sky`` SDK import is required.
"""

import configparser
import threading

import pytest
import yaml

from gbserver.environment import skypilot_config as sc
from gbserver.types.environmentconfig import (
    AwsCredentialProfile,
    ClusterSshConfigs,
    ClusterSshHost,
)
from gbserver.types.errors import SkypilotConfigCollisionError


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset module-level cloud_config accumulation around each test."""
    sc._reset_for_tests()
    yield
    sc._reset_for_tests()


def _host(host="clusterA", **kw):
    return ClusterSshHost(host=host, **kw)


def _read(path):
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Secret resolution + rendering
# --------------------------------------------------------------------------- #
class TestSecretResolution:
    def test_secret_name_resolves_literal_falls_back(self):
        secrets = {"LSF_HOSTNAME": "login.example.com"}
        blocks = sc.render_ssh_hosts(
            [_host(hostname="LSF_HOSTNAME", user="root", port=2222)], secrets
        )
        block = blocks["clusterA"]
        assert "HostName login.example.com" in block  # resolved from secret
        assert "User root" in block  # no such secret -> literal
        assert "Port 2222" in block

    def test_host_alias_never_resolved(self):
        # A secret named like the alias must NOT change the Host line.
        blocks = sc.render_ssh_hosts(
            [_host(host="clusterA", hostname="h")], {"clusterA": "SHOULD_NOT_APPLY"}
        )
        assert blocks["clusterA"].splitlines()[0] == "Host clusterA"

    def test_options_resolved(self):
        blocks = sc.render_ssh_hosts(
            [_host(options={"StrictHostKeyChecking": "no", "Port": "PORTSECRET"})],
            {"PORTSECRET": "2200"},
        )
        assert "StrictHostKeyChecking no" in blocks["clusterA"]
        assert "Port 2200" in blocks["clusterA"]

    def test_resolved_values_not_logged(self, caplog):
        import logging

        with caplog.at_level(logging.DEBUG):
            sc.render_ssh_hosts(
                [_host(hostname="SECRET_HOST")], {"SECRET_HOST": "secret-value"}
            )
        assert "secret-value" not in caplog.text
        assert "from-secret" in caplog.text


# --------------------------------------------------------------------------- #
# SSH merge / idempotency / multi-cluster / collision
# --------------------------------------------------------------------------- #
class TestSshMerge:
    def test_writes_managed_block(self, tmp_path):
        sc.merge_ssh_blocks(
            "slurm",
            sc.render_ssh_hosts([_host(hostname="h", port=2222)], {}),
            "envA",
            home=tmp_path,
        )
        text = _read(tmp_path / ".slurm" / "config")
        assert sc.MANAGED_BEGIN in text and sc.MANAGED_END in text
        assert "Host clusterA" in text and "HostName h" in text

    def test_idempotent_same_body(self, tmp_path):
        blocks = sc.render_ssh_hosts([_host(hostname="h", port=2222)], {})
        sc.merge_ssh_blocks("slurm", blocks, "envA", home=tmp_path)
        first = _read(tmp_path / ".slurm" / "config")
        sc.merge_ssh_blocks(
            "slurm", blocks, "envB", home=tmp_path
        )  # different env, same body
        assert _read(tmp_path / ".slurm" / "config") == first

    def test_multi_cluster_coexist(self, tmp_path):
        sc.merge_ssh_blocks(
            "slurm",
            sc.render_ssh_hosts([_host("clusterA", hostname="a")], {}),
            "envA",
            home=tmp_path,
        )
        sc.merge_ssh_blocks(
            "slurm",
            sc.render_ssh_hosts([_host("clusterB", hostname="b")], {}),
            "envB",
            home=tmp_path,
        )
        text = _read(tmp_path / ".slurm" / "config")
        assert "Host clusterA" in text and "Host clusterB" in text

    def test_collision_same_alias_different_body(self, tmp_path):
        sc.merge_ssh_blocks(
            "slurm",
            sc.render_ssh_hosts([_host("clusterA", hostname="a")], {}),
            "envA",
            home=tmp_path,
        )
        with pytest.raises(SkypilotConfigCollisionError) as exc:
            sc.merge_ssh_blocks(
                "slurm",
                sc.render_ssh_hosts([_host("clusterA", hostname="DIFFERENT")], {}),
                "envB",
                home=tmp_path,
            )
        msg = str(exc.value)
        assert "clusterA" in msg and "envB" in msg and "envA" in msg

    def test_foreign_content_preserved_and_foreign_alias_conflicts(self, tmp_path):
        dest = tmp_path / ".slurm" / "config"
        dest.parent.mkdir(parents=True)
        dest.write_text("Host other\n    HostName x\n", encoding="utf-8")
        # Adding an unrelated alias preserves the foreign entry.
        sc.merge_ssh_blocks(
            "slurm",
            sc.render_ssh_hosts([_host("clusterA", hostname="a")], {}),
            "envA",
            home=tmp_path,
        )
        assert "Host other" in _read(dest)
        # A foreign (non-gbserver) entry for the same alias is a conflict — refuse.
        with pytest.raises(SkypilotConfigCollisionError):
            sc.merge_ssh_blocks(
                "slurm",
                sc.render_ssh_hosts([_host("other", hostname="a")], {}),
                "envA",
                home=tmp_path,
            )


# --------------------------------------------------------------------------- #
# cloud_config deep-merge
# --------------------------------------------------------------------------- #
class TestCloudConfig:
    def test_disjoint_keys_merge(self, tmp_path):
        sc.merge_cloud_config({"lsf": {"a": 1}}, "envA", tmp_root=tmp_path)
        sc.merge_cloud_config({"lsf": {"b": 2}}, "envB", tmp_root=tmp_path)
        import os

        data = yaml.safe_load(
            _read(__import__("pathlib").Path(os.environ[sc.ENV_VAR_PROJECT_CONFIG]))
        )
        assert data == {"lsf": {"a": 1, "b": 2}}

    def test_identical_idempotent(self, tmp_path):
        sc.merge_cloud_config({"lsf": {"a": 1}}, "envA", tmp_root=tmp_path)
        sc.merge_cloud_config({"lsf": {"a": 1}}, "envB", tmp_root=tmp_path)  # no raise

    def test_conflict_same_key_different_value(self, tmp_path):
        sc.merge_cloud_config({"lsf": {"a": 1}}, "envA", tmp_root=tmp_path)
        with pytest.raises(SkypilotConfigCollisionError) as exc:
            sc.merge_cloud_config({"lsf": {"a": 2}}, "envB", tmp_root=tmp_path)
        assert "lsf.a" in str(exc.value)


# --------------------------------------------------------------------------- #
# AWS credentials
# --------------------------------------------------------------------------- #
class TestAwsCredentials:
    def test_renders_resolved_profile_mode_0600(self, tmp_path):
        profiles = [
            AwsCredentialProfile(
                profile="default",
                aws_access_key_id="AWS_KEY",
                aws_secret_access_key="AWS_SECRET",
            )
        ]
        sc.merge_aws_credentials(
            profiles, {"AWS_KEY": "AKIA", "AWS_SECRET": "shh"}, "envA", home=tmp_path
        )
        dest = tmp_path / ".aws" / "credentials"
        cp = configparser.ConfigParser()
        cp.read(dest)
        assert cp["default"]["aws_access_key_id"] == "AKIA"
        assert cp["default"]["aws_secret_access_key"] == "shh"
        assert (dest.stat().st_mode & 0o777) == 0o600

    def test_foreign_profile_preserved(self, tmp_path):
        dest = tmp_path / ".aws" / "credentials"
        dest.parent.mkdir(parents=True)
        dest.write_text("[other]\naws_access_key_id = keep\n", encoding="utf-8")
        sc.merge_aws_credentials(
            [AwsCredentialProfile(profile="default", aws_access_key_id="X")],
            {},
            "envA",
            home=tmp_path,
        )
        cp = configparser.ConfigParser()
        cp.read(dest)
        assert cp["other"]["aws_access_key_id"] == "keep"
        assert cp["default"]["aws_access_key_id"] == "X"

    def test_collision_same_profile_different_values(self, tmp_path):
        # An existing profile with different values is a conflict — refuse rather
        # than clobber it (never overwrites a user's real credentials).
        sc.merge_aws_credentials(
            [AwsCredentialProfile(profile="default", aws_access_key_id="X")],
            {},
            "envA",
            home=tmp_path,
        )
        with pytest.raises(SkypilotConfigCollisionError) as exc:
            sc.merge_aws_credentials(
                [AwsCredentialProfile(profile="default", aws_access_key_id="Y")],
                {},
                "envB",
                home=tmp_path,
            )
        assert "default" in str(exc.value)


# --------------------------------------------------------------------------- #
# No teardown + concurrency
# --------------------------------------------------------------------------- #
class TestNoTeardownAndConcurrency:
    def test_module_exposes_no_release(self):
        assert not hasattr(sc, "release")

    def test_materialize_all_sections(self, tmp_path):
        ssh = ClusterSshConfigs(slurm=[_host(hostname="h")])
        aws = [AwsCredentialProfile(profile="default", aws_access_key_id="K")]
        sc.materialize(
            "envA", ssh, {"lsf": {"q": 1}}, aws, {}, home=tmp_path, tmp_root=tmp_path
        )
        assert (tmp_path / ".slurm" / "config").exists()
        assert (tmp_path / ".aws" / "credentials").exists()

    def test_concurrent_distinct_aliases(self, tmp_path):
        def worker(alias):
            sc.merge_ssh_blocks(
                "slurm",
                sc.render_ssh_hosts([_host(alias, hostname=alias)], {}),
                alias,
                home=tmp_path,
            )

        threads = [threading.Thread(target=worker, args=(f"c{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        text = _read(tmp_path / ".slurm" / "config")
        for i in range(8):
            assert f"Host c{i}" in text

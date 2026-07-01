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

Pure-filesystem: ``home`` points at ``tmp_path`` so the real home directory is
never touched, and no ``sky`` SDK import is required.
"""

import configparser
import threading

import pytest
import yaml

from gbserver.environment import skypilot_config as sc
from gbserver.types.environmentconfig import (
    AwsCredentialProfile,
    ClusterSshConfigs,
)
from gbserver.types.errors import SkypilotConfigCollisionError


def _host(alias="clusterA", **directives):
    """Build a host mapping using exact OpenSSH directive keys (e.g. HostName=...)."""
    return {"Host": alias, **directives}


def _read(path):
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Secret resolution + rendering
# --------------------------------------------------------------------------- #
class TestSecretResolution:
    def test_secret_name_resolves_literal_falls_back(self):
        secrets = {"LSF_HOSTNAME": "login.example.com"}
        blocks = sc.render_ssh_hosts(
            [_host(HostName="LSF_HOSTNAME", User="root", Port=2222)], secrets
        )
        block = blocks["clusterA"]
        assert "HostName login.example.com" in block  # resolved from secret
        assert "User root" in block  # no such secret -> literal
        assert "Port 2222" in block

    def test_host_alias_never_resolved(self):
        # A secret named like the alias must NOT change the Host line.
        blocks = sc.render_ssh_hosts(
            [_host("clusterA", HostName="h")], {"clusterA": "SHOULD_NOT_APPLY"}
        )
        assert blocks["clusterA"].splitlines()[0] == "Host clusterA"

    def test_directives_rendered_verbatim_and_resolved(self):
        blocks = sc.render_ssh_hosts(
            [_host(StrictHostKeyChecking="no", Port="PORTSECRET")],
            {"PORTSECRET": "2200"},
        )
        assert "StrictHostKeyChecking no" in blocks["clusterA"]
        assert "Port 2200" in blocks["clusterA"]

    def test_bool_directive_renders_yes_no(self):
        blocks = sc.render_ssh_hosts([_host(IdentitiesOnly=True)], {})
        assert "IdentitiesOnly yes" in blocks["clusterA"]

    def test_resolved_values_not_logged(self, caplog):
        import logging

        with caplog.at_level(logging.DEBUG):
            sc.render_ssh_hosts(
                [_host(HostName="SECRET_HOST")], {"SECRET_HOST": "secret-value"}
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
            sc.render_ssh_hosts([_host(HostName="h", Port=2222)], {}),
            "envA",
            home=tmp_path,
        )
        text = _read(tmp_path / ".slurm" / "config")
        assert sc.MANAGED_BEGIN in text and sc.MANAGED_END in text
        assert "Host clusterA" in text and "HostName h" in text

    def test_idempotent_same_body(self, tmp_path):
        blocks = sc.render_ssh_hosts([_host(HostName="h", Port=2222)], {})
        sc.merge_ssh_blocks("slurm", blocks, "envA", home=tmp_path)
        first = _read(tmp_path / ".slurm" / "config")
        sc.merge_ssh_blocks(
            "slurm", blocks, "envB", home=tmp_path
        )  # different env, same body
        assert _read(tmp_path / ".slurm" / "config") == first

    def test_multi_cluster_coexist(self, tmp_path):
        sc.merge_ssh_blocks(
            "slurm",
            sc.render_ssh_hosts([_host("clusterA", HostName="a")], {}),
            "envA",
            home=tmp_path,
        )
        sc.merge_ssh_blocks(
            "slurm",
            sc.render_ssh_hosts([_host("clusterB", HostName="b")], {}),
            "envB",
            home=tmp_path,
        )
        text = _read(tmp_path / ".slurm" / "config")
        assert "Host clusterA" in text and "Host clusterB" in text

    def test_collision_same_alias_different_body(self, tmp_path):
        sc.merge_ssh_blocks(
            "slurm",
            sc.render_ssh_hosts([_host("clusterA", HostName="a")], {}),
            "envA",
            home=tmp_path,
        )
        with pytest.raises(SkypilotConfigCollisionError) as exc:
            sc.merge_ssh_blocks(
                "slurm",
                sc.render_ssh_hosts([_host("clusterA", HostName="DIFFERENT")], {}),
                "envB",
                home=tmp_path,
            )
        msg = str(exc.value)
        assert "clusterA" in msg and "envB" in msg and "envA" in msg

    def test_foreign_content_preserved_and_differing_alias_conflicts(self, tmp_path):
        dest = tmp_path / ".slurm" / "config"
        dest.parent.mkdir(parents=True)
        dest.write_text("Host other\n    HostName x\n", encoding="utf-8")
        # Adding an unrelated alias preserves the foreign entry.
        sc.merge_ssh_blocks(
            "slurm",
            sc.render_ssh_hosts([_host("clusterA", HostName="a")], {}),
            "envA",
            home=tmp_path,
        )
        assert "Host other" in _read(dest)
        # A foreign entry for the same alias with DIFFERENT content is a conflict.
        with pytest.raises(SkypilotConfigCollisionError):
            sc.merge_ssh_blocks(
                "slurm",
                sc.render_ssh_hosts([_host("other", HostName="a")], {}),
                "envA",
                home=tmp_path,
            )

    def test_foreign_alias_identical_is_noop(self, tmp_path):
        # The common case: ~/.<cloud>/config already has a matching entry. An
        # identical foreign entry is NOT a conflict — it is left untouched and no
        # gbserver-managed block is added (content-aware, order-independent).
        dest = tmp_path / ".lsf" / "config"
        dest.parent.mkdir(parents=True)
        original = (
            "Host bluevela\n"
            "    HostName login3.example.com\n"
            "    User svc\n"
            "    IdentityFile ~/.ssh/k\n"
            "    IdentitiesOnly yes\n"
        )
        dest.write_text(original, encoding="utf-8")
        # Same directives, different field order in the env — still equivalent.
        host = _host(
            "bluevela",
            IdentitiesOnly="yes",
            User="svc",
            HostName="login3.example.com",
            IdentityFile="~/.ssh/k",
        )
        sc.merge_ssh_blocks(
            "lsf", sc.render_ssh_hosts([host], {}), "sky-lsf", home=tmp_path
        )
        text = _read(dest)
        assert sc.MANAGED_BEGIN not in text  # no managed block added
        assert text == original  # foreign file left byte-for-byte unchanged


# --------------------------------------------------------------------------- #
# cloud_config -> ~/.sky/config.yaml (written from the env, env wins)
# --------------------------------------------------------------------------- #
class TestCloudConfig:
    def _sky_config(self, tmp_path):
        return yaml.safe_load((tmp_path / ".sky" / "config.yaml").read_text())

    def test_writes_into_sky_config(self, tmp_path):
        sc.merge_cloud_config({"lsf": {"a": 1}}, "envA", home=tmp_path)
        assert self._sky_config(tmp_path) == {"lsf": {"a": 1}}

    def test_deep_merge_preserves_unrelated_keys(self, tmp_path):
        dest = tmp_path / ".sky" / "config.yaml"
        dest.parent.mkdir(parents=True)
        dest.write_text("kubernetes:\n  remote_identity: sa\nlsf:\n  x: 1\n")
        sc.merge_cloud_config({"lsf": {"y": 2}}, "envA", home=tmp_path)
        cfg = self._sky_config(tmp_path)
        assert cfg["kubernetes"] == {"remote_identity": "sa"}  # untouched
        assert cfg["lsf"] == {"x": 1, "y": 2}  # merged

    def test_env_value_wins_on_conflict(self, tmp_path):
        dest = tmp_path / ".sky" / "config.yaml"
        dest.parent.mkdir(parents=True)
        dest.write_text("lsf:\n  cluster_configs:\n    bv:\n      queue: g\n")
        sc.merge_cloud_config(
            {"lsf": {"cluster_configs": {"bv": {"queue": "grp_preemptable"}}}},
            "envA",
            home=tmp_path,
        )
        cfg = self._sky_config(tmp_path)
        assert cfg["lsf"]["cluster_configs"]["bv"]["queue"] == "grp_preemptable"


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

    def test_secret_resolving_to_none_is_skipped(self, tmp_path):
        # A field naming a secret whose value is None must be omitted, not crash
        # configparser (which rejects non-string option values).
        profiles = [
            AwsCredentialProfile(
                profile="default",
                aws_access_key_id="AWS_KEY",
                aws_secret_access_key="AWS_SECRET",
            )
        ]
        sc.merge_aws_credentials(
            profiles, {"AWS_KEY": "AKIA", "AWS_SECRET": None}, "envA", home=tmp_path
        )
        cp = configparser.ConfigParser()
        cp.read(tmp_path / ".aws" / "credentials")
        assert cp["default"]["aws_access_key_id"] == "AKIA"
        assert "aws_secret_access_key" not in cp["default"]


# --------------------------------------------------------------------------- #
# IdentityKey: inline private-key material -> managed 0600 key file
# --------------------------------------------------------------------------- #
class TestIdentityKey:
    _PEM = "-----BEGIN OPENSSH PRIVATE KEY-----\nabc123\n-----END-----"

    def _materialize(self, tmp_path, secrets, **directives):
        ssh = ClusterSshConfigs(lsf=[_host("bluevela", **directives)])
        sc.materialize("sky-lsf", ssh, None, None, secrets, home=tmp_path)
        return tmp_path / ".lsf" / "config"

    def test_writes_keyfile_and_rewrites_identityfile(self, tmp_path):
        dest = self._materialize(
            tmp_path, {"BV_KEY": self._PEM}, HostName="h", IdentityKey="BV_KEY"
        )
        text = _read(dest)
        assert "IdentityKey" not in text  # directive replaced
        # The rewritten IdentityFile points at a managed 0600 key file with the key.
        key_dir = tmp_path / ".sky" / "keys"
        key_files = list(key_dir.glob("*.key"))
        assert len(key_files) == 1
        assert f"IdentityFile {key_files[0]}" in text
        assert key_files[0].read_text(encoding="utf-8") == self._PEM + "\n"
        assert (key_files[0].stat().st_mode & 0o777) == 0o600

    def test_idempotent_same_key(self, tmp_path):
        dest = self._materialize(
            tmp_path, {"BV_KEY": self._PEM}, HostName="h", IdentityKey="BV_KEY"
        )
        first = _read(dest)
        self._materialize(
            tmp_path, {"BV_KEY": self._PEM}, HostName="h", IdentityKey="BV_KEY"
        )
        assert _read(dest) == first  # stable content-addressed path, no churn

    def test_same_alias_different_key_collides(self, tmp_path):
        self._materialize(
            tmp_path, {"BV_KEY": self._PEM}, HostName="h", IdentityKey="BV_KEY"
        )
        with pytest.raises(SkypilotConfigCollisionError):
            self._materialize(
                tmp_path,
                {"BV_KEY": "-----DIFFERENT KEY-----"},
                HostName="h",
                IdentityKey="BV_KEY",
            )

    def test_both_identityfile_and_identitykey_raises(self, tmp_path):
        with pytest.raises(ValueError):
            self._materialize(
                tmp_path,
                {"BV_KEY": self._PEM},
                IdentityFile="~/.ssh/k",
                IdentityKey="BV_KEY",
            )

    def test_empty_resolved_key_raises(self, tmp_path):
        with pytest.raises(ValueError):
            self._materialize(tmp_path, {"BV_KEY": ""}, IdentityKey="BV_KEY")

    def test_unresolved_secret_name_warns(self, tmp_path, caplog):
        import logging

        # No matching secret: IdentityKey falls back to the literal name and is
        # almost certainly a misconfiguration — warn loudly.
        with caplog.at_level(logging.WARNING):
            sc._materialize_identity_keys(
                [_host("bluevela", IdentityKey="BV_SSH_PRIVATE_KEY")],
                "lsf",
                {},
                tmp_path,
            )
        assert "looks like a secret name" in caplog.text

    def test_key_contents_not_logged(self, tmp_path, caplog):
        import logging

        with caplog.at_level(logging.DEBUG):
            self._materialize(
                tmp_path, {"BV_KEY": self._PEM}, HostName="h", IdentityKey="BV_KEY"
            )
        assert "abc123" not in caplog.text


# --------------------------------------------------------------------------- #
# No teardown + concurrency
# --------------------------------------------------------------------------- #
class TestNoTeardownAndConcurrency:
    def test_module_exposes_no_release(self):
        assert not hasattr(sc, "release")

    def test_materialize_all_sections(self, tmp_path):
        ssh = ClusterSshConfigs(slurm=[_host(HostName="h")])
        aws = [AwsCredentialProfile(profile="default", aws_access_key_id="K")]
        sc.materialize("envA", ssh, {"lsf": {"q": 1}}, aws, {}, home=tmp_path)
        assert (tmp_path / ".slurm" / "config").exists()
        assert (tmp_path / ".aws" / "credentials").exists()
        assert (tmp_path / ".sky" / "config.yaml").exists()

    def test_concurrent_distinct_aliases(self, tmp_path):
        def worker(alias):
            sc.merge_ssh_blocks(
                "slurm",
                sc.render_ssh_hosts([_host(alias, HostName=alias)], {}),
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

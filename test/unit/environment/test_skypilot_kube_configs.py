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

"""``kube_configs``: an environment supplies its Kubernetes credentials (#505)."""

import asyncio
import os
import stat
from unittest.mock import patch

import pytest
import yaml

from gbserver.environment import skypilot_config as sc
from gbserver.types.environmentconfig import KubeConfigEntry
from gbserver.types.errors import SkypilotConfigCollisionError


def _kubeconfig(context, server="https://a:6443", token="t1"):
    return yaml.safe_dump(
        {
            "apiVersion": "v1",
            "kind": "Config",
            "clusters": [{"name": f"{context}-c", "cluster": {"server": server}}],
            "users": [{"name": f"{context}-u", "user": {"token": token}}],
            "contexts": [
                {
                    "name": context,
                    "context": {"cluster": f"{context}-c", "user": f"{context}-u"},
                }
            ],
            "current-context": context,
        }
    )


@pytest.fixture(autouse=True)
def _isolated_kubeconfig(tmp_path, monkeypatch):
    """No test reads the developer's real kubeconfig."""
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "user-kubeconfig"))


def _managed(tmp_path):
    """The managed kubeconfig, minus the placeholder default entries."""
    doc = yaml.safe_load(sc.kubeconfig_path(tmp_path).read_text())
    for section in ("contexts", "clusters", "users"):
        doc[section] = [
            e for e in doc.get(section) or [] if e["name"] != sc._KUBE_PLACEHOLDER
        ]
    return doc


def _raw_managed(tmp_path):
    return yaml.safe_load(sc.kubeconfig_path(tmp_path).read_text())


def _entry(context="remote", secret="remote-kc"):
    return KubeConfigEntry(context=context, kubeconfig=secret)


class TestMerge:
    def test_writes_context_cluster_user_0600(self, tmp_path):
        assert sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        doc = _managed(tmp_path)
        assert [c["name"] for c in doc["contexts"]] == ["remote"]
        # The secret's own cluster/user names are replaced (see the collision
        # tests below for why).
        assert [c["name"] for c in doc["clusters"]] == ["gbserver/remote"]
        assert [u["name"] for u in doc["users"]] == ["gbserver/remote"]
        assert doc["contexts"][0]["context"] == {
            "cluster": "gbserver/remote",
            "user": "gbserver/remote",
        }
        # The placeholder, not a real context, so a merged KUBECONFIG is valid
        # without making an env's cluster the process-wide default.
        assert doc["current-context"] == sc._KUBE_PLACEHOLDER
        mode = stat.S_IMODE(os.stat(sc.kubeconfig_path(tmp_path)).st_mode)
        assert mode == 0o600

    def test_same_entry_again_is_a_noop(self, tmp_path):
        secrets = {"remote-kc": _kubeconfig("remote")}
        sc.merge_kube_configs([_entry()], secrets, "envA", home=tmp_path)
        assert not sc.merge_kube_configs([_entry()], secrets, "envA", home=tmp_path)

    def test_same_env_replaces_its_rotated_token(self, tmp_path):
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        sc.merge_kube_configs(
            [_entry()],
            {"remote-kc": _kubeconfig("remote", token="t2")},
            "envA",
            home=tmp_path,
        )
        users = _managed(tmp_path)["users"]
        assert [u["user"]["token"] for u in users] == ["t2"]

    def test_other_env_conflicting_context_is_refused(self, tmp_path):
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        with pytest.raises(SkypilotConfigCollisionError, match="'envA'"):
            sc.merge_kube_configs(
                [_entry()],
                {"remote-kc": _kubeconfig("remote", token="other")},
                "envB",
                home=tmp_path,
            )
        # envA's credentials are untouched.
        assert _managed(tmp_path)["users"][0]["user"]["token"] == "t1"

    def test_other_env_different_contexts_coexist(self, tmp_path):
        sc.merge_kube_configs(
            [_entry("a", "a-kc")], {"a-kc": _kubeconfig("a")}, "envA", home=tmp_path
        )
        sc.merge_kube_configs(
            [_entry("b", "b-kc")], {"b-kc": _kubeconfig("b")}, "envB", home=tmp_path
        )
        assert sorted(c["name"] for c in _managed(tmp_path)["contexts"]) == ["a", "b"]

    def test_users_own_differing_context_is_refused(self, tmp_path, monkeypatch):
        user = tmp_path / "user-kubeconfig"
        user.write_text(_kubeconfig("remote", server="https://mine:6443"))
        with pytest.raises(SkypilotConfigCollisionError, match="your own kubeconfig"):
            sc.merge_kube_configs(
                [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
            )
        assert not sc.kubeconfig_path(tmp_path).exists()

    def test_missing_secret_names_the_secret_not_its_value(self, tmp_path):
        pasted = _kubeconfig("remote", token="PASTED-TOKEN")
        with pytest.raises(ValueError, match="not in the space's secrets") as info:
            sc.merge_kube_configs([_entry(secret=pasted)], {}, "envA", home=tmp_path)
        # A kubeconfig pasted where the secret *name* belongs is not echoed.
        assert "PASTED-TOKEN" not in str(info.value)

    @pytest.mark.parametrize(
        "text, match",
        [
            (": not yaml: [", "not valid YAML"),
            ("- a list", "not a mapping"),
            (_kubeconfig("other"), "does not define context 'remote'"),
        ],
    )
    def test_bad_secret_content(self, tmp_path, text, match):
        with pytest.raises(ValueError, match=match):
            sc.merge_kube_configs(
                [_entry()], {"remote-kc": text}, "envA", home=tmp_path
            )

    def test_secret_values_never_logged(self, tmp_path, caplog):
        caplog.set_level("DEBUG")
        sc.merge_kube_configs(
            [_entry()],
            {"remote-kc": _kubeconfig("remote", token="S3CR3T")},
            "envA",
            home=tmp_path,
        )
        assert "S3CR3T" not in caplog.text

    @pytest.mark.parametrize("section", ["clusters", "users"])
    def test_users_cluster_or_user_named_like_ours_is_refused(self, tmp_path, section):
        """A user entry named gbserver/<context> would shadow ours in one tool."""
        user = yaml.safe_load(_kubeconfig("mine"))
        user[section][0]["name"] = "gbserver/remote"
        key = "cluster" if section == "clusters" else "user"
        user["contexts"][0]["context"][key] = "gbserver/remote"
        (tmp_path / "user-kubeconfig").write_text(yaml.safe_dump(user))
        with pytest.raises(SkypilotConfigCollisionError, match=f"kube {key} "):
            sc.merge_kube_configs(
                [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
            )

    def test_generic_cluster_user_names_cannot_hijack_users_context(self, tmp_path):
        """A secret reusing the user's cluster/user names leaves theirs intact.

        kubernetes-client keeps the first definition of each cluster/user name
        across KUBECONFIG, so a secret's `default` cluster must not shadow the
        user's `default`.
        """
        kube_config = pytest.importorskip("kubernetes.config.kube_config")

        def config(context, server, token):
            return yaml.safe_dump(
                {
                    "apiVersion": "v1",
                    "kind": "Config",
                    "clusters": [{"name": "default", "cluster": {"server": server}}],
                    "users": [{"name": "default", "user": {"token": token}}],
                    "contexts": [
                        {
                            "name": context,
                            "context": {"cluster": "default", "user": "default"},
                        }
                    ],
                    "current-context": context,
                }
            )

        (tmp_path / "user-kubeconfig").write_text(
            config("local", "https://mine", "MINE")
        )
        sc.merge_kube_configs(
            [_entry()],
            {"remote-kc": config("remote", "https://remote", "REMOTE")},
            "envA",
            home=tmp_path,
        )
        sc.activate_kubeconfig(home=tmp_path)
        for ctx, server, token in (
            ("local", "https://mine", "MINE"),
            ("remote", "https://remote", "REMOTE"),
        ):
            loader = kube_config._get_kube_config_loader(
                filename=os.environ["KUBECONFIG"], active_context=ctx
            )
            assert loader._cluster["server"] == server
            assert loader._user["token"] == token

    def test_two_contexts_from_one_env_sharing_a_cluster_name(self, tmp_path):
        """Different secrets naming their cluster alike no longer clobber."""
        sc.merge_kube_configs(
            [_entry("a", "a-kc"), _entry("b", "b-kc")],
            {
                "a-kc": _kubeconfig("a", server="https://a"),
                "b-kc": yaml.safe_dump(
                    {
                        "clusters": [
                            {"name": "a-c", "cluster": {"server": "https://b"}}
                        ],
                        "users": [{"name": "a-u", "user": {"token": "tb"}}],
                        "contexts": [
                            {"name": "b", "context": {"cluster": "a-c", "user": "a-u"}}
                        ],
                    }
                ),
            },
            "envA",
            home=tmp_path,
        )
        servers = {
            c["name"]: c["cluster"]["server"] for c in _managed(tmp_path)["clusters"]
        }
        assert servers == {"gbserver/a": "https://a", "gbserver/b": "https://b"}

    def test_same_context_listed_twice_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="more than once"):
            sc.merge_kube_configs(
                [_entry(), _entry()],
                {"remote-kc": _kubeconfig("remote")},
                "envA",
                home=tmp_path,
            )

    def test_identical_shared_context_is_co_owned(self, tmp_path):
        """envB using envA's exact context becomes a co-owner (review note 7)."""
        secrets = {"remote-kc": _kubeconfig("remote")}
        sc.merge_kube_configs([_entry()], secrets, "envA", home=tmp_path)
        sc.merge_kube_configs([_entry()], secrets, "envB", home=tmp_path)
        assert _managed(tmp_path)["x-gbserver-owners"] == {"remote": ["envA", "envB"]}
        # A co-owned context cannot be changed by either: it would change the
        # credentials the other one uses.
        with pytest.raises(SkypilotConfigCollisionError, match="'envB'"):
            sc.merge_kube_configs(
                [_entry()],
                {"remote-kc": _kubeconfig("remote", token="t2")},
                "envA",
                home=tmp_path,
            )
        # Releasing one keeps the context for the other...
        assert sc.release_kube_configs("envA", home=tmp_path)
        assert [c["name"] for c in _managed(tmp_path)["contexts"]] == ["remote"]
        assert _managed(tmp_path)["x-gbserver-owners"] == {"remote": ["envB"]}
        # ...which, now sole owner, can rotate it; releasing it removes it.
        sc.merge_kube_configs(
            [_entry()],
            {"remote-kc": _kubeconfig("remote", token="t2")},
            "envB",
            home=tmp_path,
        )
        assert _managed(tmp_path)["users"][0]["user"]["token"] == "t2"
        assert sc.release_kube_configs("envB", home=tmp_path)
        assert _managed(tmp_path)["contexts"] == []

    def test_legacy_single_owner_string_is_read_as_a_set(self, tmp_path):
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        raw = _raw_managed(tmp_path)
        raw["x-gbserver-owners"] = {"remote": "envA"}
        sc.kubeconfig_path(tmp_path).write_text(yaml.safe_dump(raw))
        assert sc.managed_kube_contexts(home=tmp_path) == {"remote": {"envA"}}

    def test_yaml_error_does_not_chain_the_secret_text(self, tmp_path):
        """PyYAML's error echoes the offending line; it must not ride along."""
        bad = "users:\n- user: {token: S3CR3T: x}\n"
        with pytest.raises(ValueError) as info:
            sc.merge_kube_configs([_entry()], {"remote-kc": bad}, "envA", home=tmp_path)
        assert info.value.__cause__ is None and info.value.__suppress_context__
        assert "S3CR3T" not in str(info.value)

    def test_release_removes_only_that_envs_entries(self, tmp_path):
        sc.merge_kube_configs(
            [_entry("a", "a-kc")], {"a-kc": _kubeconfig("a")}, "envA", home=tmp_path
        )
        sc.merge_kube_configs(
            [_entry("b", "b-kc")], {"b-kc": _kubeconfig("b")}, "envB", home=tmp_path
        )
        assert sc.release_kube_configs("envA", home=tmp_path)
        doc = _managed(tmp_path)
        assert [c["name"] for c in doc["contexts"]] == ["b"]
        assert doc["current-context"] == sc._KUBE_PLACEHOLDER
        assert not sc.release_kube_configs("envA", home=tmp_path)
        # Releasing the last owner removes the placeholder too.
        assert sc.release_kube_configs("envB", home=tmp_path)
        raw = _raw_managed(tmp_path)
        assert raw["contexts"] == [] and "current-context" not in raw


class TestRobustness:
    def test_tmp_file_is_created_0600(self, tmp_path, monkeypatch):
        """The temp file never exists with looser permissions, even briefly."""
        seen = []
        real_open = os.open

        def spy(path, flags, mode=0o777, *a, **k):
            if str(path).endswith(".gbtmp"):
                seen.append(mode)
            return real_open(path, flags, mode, *a, **k)

        monkeypatch.setattr(os, "open", spy)
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        assert seen == [0o600]

    def test_corrupt_managed_file_is_a_valueerror_without_its_text(self, tmp_path):
        dest = sc.kubeconfig_path(tmp_path)
        dest.parent.mkdir(parents=True)
        dest.write_text("users:\n- user: {token: S3CR3T: x}\n")
        with pytest.raises(ValueError, match="not valid YAML") as info:
            sc.merge_kube_configs(
                [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
            )
        assert "S3CR3T" not in str(info.value) and info.value.__cause__ is None
        with pytest.raises(ValueError):
            sc.release_kube_configs("envA", home=tmp_path)

    def test_malformed_context_entry_is_a_valueerror(self, tmp_path):
        bad = yaml.safe_dump({"contexts": [{"name": "remote", "context": "oops"}]})
        with pytest.raises(ValueError, match="malformed"):
            sc.merge_kube_configs([_entry()], {"remote-kc": bad}, "envA", home=tmp_path)

    def test_placeholder_name_is_reserved(self, tmp_path):
        with pytest.raises(ValueError, match="reserved"):
            sc.merge_kube_configs(
                [_entry(sc._KUBE_PLACEHOLDER, "x")],
                {"x": _kubeconfig(sc._KUBE_PLACEHOLDER)},
                "envA",
                home=tmp_path,
            )


class TestActivate:
    def test_appends_managed_last_with_absolute_paths(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KUBECONFIG", "/x/one" + os.pathsep + "/x/two")
        sc.activate_kubeconfig(home=tmp_path)
        managed = os.path.abspath(str(sc.kubeconfig_path(tmp_path)))
        assert os.environ["KUBECONFIG"].split(os.pathsep) == [
            "/x/one",
            "/x/two",
            managed,
        ]
        sc.activate_kubeconfig(home=tmp_path)  # idempotent
        assert os.environ["KUBECONFIG"].split(os.pathsep).count(managed) == 1

    def test_tilde_is_expanded_for_kubectl(self, tmp_path, monkeypatch):
        """kubectl does not expand ~ in KUBECONFIG, so no part may keep one."""
        monkeypatch.delenv("KUBECONFIG")
        monkeypatch.setenv("HOME", str(tmp_path / "h"))
        sc.activate_kubeconfig(home=tmp_path)
        parts = os.environ["KUBECONFIG"].split(os.pathsep)
        assert parts[0] == str(tmp_path / "h" / ".kube" / "config")
        assert all(os.path.isabs(p) and "~" not in p for p in parts)

        monkeypatch.setenv("KUBECONFIG", "~/a" + os.pathsep + "rel/b")
        sc.activate_kubeconfig(home=tmp_path)
        parts = os.environ["KUBECONFIG"].split(os.pathsep)
        assert parts[:2] == [str(tmp_path / "h" / "a"), os.path.abspath("rel/b")]

    def test_merged_kubeconfig_lists_both_and_keeps_users_current(
        self, tmp_path, monkeypatch
    ):
        """The kubernetes client SkyPilot uses accepts the merged list."""
        kube_config = pytest.importorskip("kubernetes.config")
        user = tmp_path / "user-kubeconfig"
        user.write_text(_kubeconfig("mine", server="https://mine:6443"))
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        sc.activate_kubeconfig(home=tmp_path)
        contexts, current = kube_config.list_kube_config_contexts(
            os.environ["KUBECONFIG"]
        )
        names = {c["name"] for c in contexts} - {sc._KUBE_PLACEHOLDER}
        assert names == {"mine", "remote"}
        assert current["name"] == "mine"

    def test_merged_kubeconfig_valid_when_user_has_none(self, tmp_path, monkeypatch):
        kube_config = pytest.importorskip("kubernetes.config")
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        sc.activate_kubeconfig(home=tmp_path)
        contexts, current = kube_config.list_kube_config_contexts(
            os.environ["KUBECONFIG"]
        )
        assert {c["name"] for c in contexts} == {"remote", sc._KUBE_PLACEHOLDER}
        # The default is the unreachable placeholder, never an env's cluster.
        assert current["name"] == sc._KUBE_PLACEHOLDER
        from kubernetes.config import kube_config as kc

        loader = kc._get_kube_config_loader(filename=os.environ["KUBECONFIG"])
        assert loader._cluster["server"] == "https://192.0.2.1:1"
        assert not loader._user


class TestDefaultContextBothTools:
    """kubectl takes the *first* current-context in KUBECONFIG, the Python
    client the *last*; the user's must win in both, and with none set both
    must get the credential-less placeholder, never an environment's cluster."""

    def _user(self, tmp_path, current):
        doc = yaml.safe_load(_kubeconfig("mine", server="https://mine:6443"))
        if current is None:
            doc.pop("current-context")
        else:
            doc["current-context"] = current
        (tmp_path / "user-kubeconfig").write_text(yaml.safe_dump(doc))

    def _defaults(self):
        import shutil
        import subprocess

        from kubernetes.config import kube_config as kc

        contexts, current = kc.list_kube_config_contexts(os.environ["KUBECONFIG"])
        py = (current["name"], {c["name"] for c in contexts})
        kubectl = shutil.which("kubectl")
        if kubectl is None:
            return py, None
        run = lambda *a: subprocess.run(  # noqa: E731
            [kubectl, "config", *a], capture_output=True, text=True, timeout=30
        )
        k_current = run("current-context").stdout.strip()
        k_names = set(run("get-contexts", "-o", "name").stdout.split())
        return py, (k_current, k_names)

    @pytest.mark.parametrize(
        "user_current, expected",
        [
            ("mine", "mine"),
            (None, sc._KUBE_PLACEHOLDER),
            ("", sc._KUBE_PLACEHOLDER),
            ("gone", sc._KUBE_PLACEHOLDER),
        ],
    )
    def test_default_context(self, tmp_path, user_current, expected):
        pytest.importorskip("kubernetes.config")
        self._user(tmp_path, user_current)
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        sc.activate_kubeconfig(home=tmp_path)
        if user_current == "gone":
            # A dangling user current-context breaks both tools on its own,
            # before and without kube_configs; nothing to assert beyond the
            # managed file not supplying a real default.
            assert _raw_managed(tmp_path)["current-context"] == sc._KUBE_PLACEHOLDER
            return
        py, kubectl = self._defaults()
        assert py[0] == expected
        assert {"mine", "remote"} <= py[1]
        if kubectl is not None:
            assert kubectl[0] == expected
            assert {"mine", "remote"} <= kubectl[1]

    @pytest.mark.parametrize(
        "first_current, second_current",
        [
            # The reviewer's case: the second user file's current-context names
            # a context only the managed file defines. kubectl (first set) gets
            # "mine"; the Python client (last set) would get the secret one.
            ("mine", "remote"),
            # The mirror image: kubectl would get the secret one.
            ("remote", "mine"),
            # Second file sets a different, valid user context.
            ("mine", "other"),
            # Second file sets a current-context nobody defines.
            ("mine", "ghost"),
        ],
    )
    def test_several_user_files(
        self, tmp_path, monkeypatch, first_current, second_current
    ):
        pytest.importorskip("kubernetes.config")
        a = yaml.safe_load(_kubeconfig("mine", server="https://mine:6443"))
        a["current-context"] = first_current
        b = yaml.safe_load(_kubeconfig("other", server="https://other:6443"))
        b["current-context"] = second_current
        (tmp_path / "a.yaml").write_text(yaml.safe_dump(a))
        (tmp_path / "b.yaml").write_text(yaml.safe_dump(b))
        monkeypatch.setenv(
            "KUBECONFIG",
            os.pathsep.join([str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")]),
        )
        if "remote" in (first_current, second_current):
            # A user default naming a secret context: kubectl would use it no
            # matter what follows on KUBECONFIG, so the merge is refused.
            with pytest.raises(SkypilotConfigCollisionError, match="current-context"):
                sc.merge_kube_configs(
                    [_entry()],
                    {"remote-kc": _kubeconfig("remote")},
                    "envA",
                    home=tmp_path,
                )
            return
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        sc.activate_kubeconfig(home=tmp_path)
        py, kubectl = self._defaults()
        user_contexts = {"mine", "other"}
        both_valid = {first_current, second_current} <= user_contexts
        # Never the secret context, in either tool.
        assert py[0] != "remote"
        if kubectl is not None:
            assert kubectl[0] != "remote"
        if both_valid:
            assert py[0] == second_current  # last set
            if kubectl is not None:
                assert kubectl[0] == first_current  # first set
        else:
            assert _raw_managed(tmp_path)["current-context"] == sc._KUBE_PLACEHOLDER

    def test_user_context_of_same_name_that_is_also_selected(self, tmp_path):
        """Reported as a same-name clash, not as 'your kubeconfig lacks it'."""
        user = yaml.safe_load(_kubeconfig("remote", server="https://mine:6443"))
        (tmp_path / "user-kubeconfig").write_text(yaml.safe_dump(user))
        with pytest.raises(
            SkypilotConfigCollisionError, match="your own kubeconfig"
        ) as e:
            sc.merge_kube_configs(
                [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
            )
        assert "current-context" not in str(e.value)

    def test_use_context_on_a_managed_context_is_refused_on_remerge(self, tmp_path):
        """`kubectl config use-context <managed>` makes the env's next setup fail.

        The merged file supplies the context, so use-context succeeds and writes
        it as the user's current-context; the next merge (a new gbserver process
        or environment instance) must refuse rather than leave it the default.
        """
        import shutil
        import subprocess

        kubectl = shutil.which("kubectl")
        if kubectl is None:
            pytest.skip("kubectl not installed")
        self._user(tmp_path, "mine")
        secrets = {"remote-kc": _kubeconfig("remote")}
        sc.merge_kube_configs([_entry()], secrets, "envA", home=tmp_path)
        sc.activate_kubeconfig(home=tmp_path)
        subprocess.run(
            [kubectl, "config", "use-context", "remote"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        user = yaml.safe_load((tmp_path / "user-kubeconfig").read_text())
        assert user["current-context"] == "remote"  # written to the user's file
        with pytest.raises(SkypilotConfigCollisionError, match="current-context"):
            sc.merge_kube_configs([_entry()], secrets, "envA", home=tmp_path)

    def test_default_reevaluated_after_release_and_remerge(self, tmp_path):
        secrets = {"remote-kc": _kubeconfig("remote")}
        self._user(tmp_path, "mine")
        sc.merge_kube_configs([_entry()], secrets, "envA", home=tmp_path)
        assert "current-context" not in _raw_managed(tmp_path)
        # The user drops their default; the next merge adds the placeholder.
        self._user(tmp_path, None)
        sc.release_kube_configs("envA", home=tmp_path)
        sc.merge_kube_configs([_entry()], secrets, "envA", home=tmp_path)
        assert _raw_managed(tmp_path)["current-context"] == sc._KUBE_PLACEHOLDER
        # And restores it: the placeholder goes away again.
        self._user(tmp_path, "mine")
        sc.merge_kube_configs(
            [_entry()],
            {"remote-kc": _kubeconfig("remote", token="t2")},
            "envA",
            home=tmp_path,
        )
        raw = _raw_managed(tmp_path)
        assert "current-context" not in raw
        assert sc._KUBE_PLACEHOLDER not in {c["name"] for c in raw["contexts"]}

    def test_placeholder_only_when_user_has_no_default(self, tmp_path):
        self._user(tmp_path, "mine")
        sc.merge_kube_configs(
            [_entry()], {"remote-kc": _kubeconfig("remote")}, "envA", home=tmp_path
        )
        raw = _raw_managed(tmp_path)
        assert "current-context" not in raw
        assert sc._KUBE_PLACEHOLDER not in {c["name"] for c in raw["contexts"]}


class TestEnvironmentWiring:
    def _env(self, config, secrets=None):
        from gbserver.environment.skypilot import Skypilot
        from gbserver.types.environmentconfig import EnvironmentConfig

        cfg = EnvironmentConfig(name="env-k", type="Skypilot", config=config)
        return Skypilot(
            event_q=asyncio.Queue(), environment_config=cfg, secrets=secrets or {}
        )

    def _config(self, allowed):
        return {
            "default_cloud": "kubernetes",
            "kube_configs": [{"context": "remote", "kubeconfig": "remote-kc"}],
            "cloud_config": {"kubernetes": {"allowed_contexts": allowed}},
        }

    def test_context_must_be_in_allowed_contexts(self):
        env = self._env(self._config(["remot"]))
        with pytest.raises(ValueError, match="allowed_contexts"):
            env._kube_config_entries()

    def test_no_allowed_contexts_at_all_is_refused(self):
        cfg = self._config([])
        del cfg["cloud_config"]
        with pytest.raises(ValueError, match="allowed_contexts"):
            self._env(cfg)._kube_config_entries()

    @pytest.mark.parametrize(
        "cloud_config, expected",
        [
            ({"kubernetes": {"allowed_contexts": ["a"]}}, ["a"]),
            (
                {
                    "kubernetes": {"allowed_contexts": ["a"]},
                    "workspaces": {
                        "default": {"kubernetes": {"allowed_contexts": ["w"]}}
                    },
                },
                ["w"],
            ),
            (
                {
                    "active_workspace": "team",
                    "kubernetes": {"allowed_contexts": ["a"]},
                    "workspaces": {"team": {"kubernetes": {"allowed_contexts": ["t"]}}},
                },
                ["t"],
            ),
            (
                {
                    "active_workspace": "team",
                    "kubernetes": {"allowed_contexts": ["a"]},
                    "workspaces": {
                        "other": {"kubernetes": {"allowed_contexts": ["o"]}}
                    },
                },
                ["a"],
            ),
            ({"kubernetes": {"allowed_contexts": "all"}}, []),
            ({}, []),
        ],
    )
    def test_allowed_contexts_resolved_like_skypilot(self, cloud_config, expected):
        from gbserver.environment.skypilot import _kube_allowed_contexts

        assert _kube_allowed_contexts(cloud_config) == expected

    def test_workspace_pinned_context_is_accepted(self):
        cfg = {
            "default_cloud": "kubernetes",
            "kube_configs": [{"context": "remote", "kubeconfig": "remote-kc"}],
            "cloud_config": {
                "workspaces": {
                    "default": {"kubernetes": {"allowed_contexts": ["remote"]}}
                }
            },
        }
        assert [e.context for e in self._env(cfg)._kube_config_entries()] == ["remote"]

    def test_teardown_paths_log_instead_of_raising(self, caplog):
        """A bad kube_configs fails setup; it must not also block cleanup."""
        env = self._env(self._config(["typo"]))
        with pytest.raises(ValueError):
            env._ensure_kube_configs()
        env._ensure_kube_configs(strict=False)
        assert "kube_configs not applied" in caplog.text

    def test_every_cluster_path_activates(self):
        """Every Skypilot method that reaches SkyPilot ensures the credentials.

        Scans the code (AST, so comments and docstrings don't count) of every
        Skypilot method and module-level helper for SkyPilot request calls. A
        method sending one directly must ensure the credentials itself; a
        module-level helper must only be called from methods that do. A new
        request path that forgets it fails here.
        """
        import ast
        import inspect

        from gbserver.environment import skypilot as sp

        tree = ast.parse(inspect.getsource(sp))
        request_attrs = {
            "launch", "down", "job_status", "status", "queue", "tail_logs",
            "cancel", "stop", "exec", "get", "stream_and_get", "download_logs",
            "api_cancel",
        }  # fmt: skip
        ensure_calls = {
            "_ensure_kube_configs",
            "_ensure_kube_configs_async",
            "_ensure_inline_configs_materialized",
            "_kube_configs_held",  # async-with bracket: merge, then release
        }

        def calls(node):
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    yield n.func

        def sends_request(node):
            # sky.<verb>(...), sky.jobs.<verb>(...), or a sky.<verb> passed as
            # a callable (to_thread(sky.down, ...)).
            for n in ast.walk(node):
                if (
                    isinstance(n, ast.Attribute)
                    and n.attr in request_attrs
                    and isinstance(n.value, (ast.Name, ast.Attribute))
                    and "sky" in ast.unparse(n.value).split(".")
                ):
                    return True
            return False

        def called_names(node):
            out = set()
            for f in calls(node):
                if isinstance(f, ast.Attribute):
                    out.add(f.attr)
                elif isinstance(f, ast.Name):
                    out.add(f.id)
            for n in ast.walk(node):  # callables passed as arguments
                if isinstance(n, ast.Call):
                    for arg in n.args:
                        if isinstance(arg, ast.Name):
                            out.add(arg.id)
                        elif isinstance(arg, ast.Attribute):
                            out.add(arg.attr)
            return out

        funcs = (ast.FunctionDef, ast.AsyncFunctionDef)
        helpers = {n.name: n for n in tree.body if isinstance(n, funcs)}
        skypilot_cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Skypilot"
        )
        methods = {n.name: n for n in skypilot_cls.body if isinstance(n, funcs)}

        requesting_helpers = {
            name for name, node in helpers.items() if sends_request(node)
        }
        ensuring = {
            name for name, node in methods.items() if called_names(node) & ensure_calls
        }
        problems = []
        for name, node in methods.items():
            if name in ensuring:
                continue
            direct = sends_request(node)
            via = called_names(node) & requesting_helpers
            if direct or via:
                problems.append(name)
        # Private methods only ever reached from methods that ensure it first.
        reached_after_ensure = {
            "_abort_provision",  # from _provision_with_retry (launch path)
            "_await_cleanup_job_status",  # from teardown_skypilot
            "_provision_with_retry",  # from _launch_skypilot_inner
            "_launch_cleanup_vm",  # from _reap_per_run_workdir
            "_reap_per_run_workdir",  # from teardown_skypilot
            "_download_and_parse_logs",  # from _poll_skypilot_job
        }
        for name in reached_after_ensure:
            callers = [m for m, n in methods.items() if name in called_names(n)]
            assert callers and all(
                c in ensuring or c in reached_after_ensure for c in callers
            ), (
                name,
                callers,
            )
        assert sorted(set(problems) - reached_after_ensure) == []

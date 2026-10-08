"""The residency preflight, tested against the two readings it has to separate.

Ported alongside src/check_weight_residency.py from this repo's standalone audit script,
whose self-test these first cases are: the mmlsattr output below is VERBATIM from the
incident of 2026-09-21, one shard of a migrated teacher and one of a resident student. They
are kept as text rather than reduced to `{"ARCHIVE", "OFFLINE"}` because the parser's job is
to survive the real format, and the real format is the thing that would change.

The rest tests the asymmetry the module's docstring argues for and nothing else enforces:
mmlsattr may clear a file or condemn it, while allocation arithmetic may only ever condemn.
A test that let `st_blocks` clear a file would pass happily and put us back where the
incident started -- a 91.6%-allocated shard that GPFS still called OFFLINE.

No test here reads a weight file's contents, for the same reason the module does not: on a
GPFS mount that would stage the data in, which is the one way this check is worse than none.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import check_weight_residency as cwr  # noqa: E402

# Verbatim `mmlsattr -L` output, one shard of the migrated teacher on fileset data-eng-cos.
FIXTURE_OFFLINE = """file name:            /proj/data-eng/archive/issei/models/granite-5.0-20b-sft/model-00008-of-00009.safetensors
metadata replication: 1 max 2
data replication:     1 max 2
immutable:            no
appendOnly:           no
flags:
storage pool name:    system
fileset name:         data-eng-cos
snapshot name:
creation time:        Thu Aug 13 02:09:44 2026
Misc attributes:      ARCHIVE OFFLINE
Encrypted:            no
"""

# Verbatim, a resident shard. Note ARCHIVE with no OFFLINE: every file on this cluster
# carries ARCHIVE, which is why a check keying on it would condemn the whole filesystem.
FIXTURE_RESIDENT = """file name:            .../granite-4.1-3b-base_retagged_v2/model-00001-of-00002.safetensors
metadata replication: 1 max 2
data replication:     1 max 2
immutable:            no
appendOnly:           no
flags:
storage pool name:    system
fileset name:         data-eng
snapshot name:
creation time:        Wed Aug 26 11:02:11 2026
Misc attributes:      ARCHIVE
Encrypted:            no
"""


def _fake_mmlsattr(tmp_path):
    """An executable stand-in that answers OFFLINE for any path containing "offline".

    A script rather than a monkeypatched `attrs_of`, so the subprocess call, the `-L`
    argument position and the stdout parsing are all exercised -- those are where a version
    difference would actually bite. It never touches the file it is asked about, which is
    also true of the real thing.
    """
    path = tmp_path / "fake-mmlsattr"
    path.write_text(
        "#!/bin/sh\n"
        'case "$2" in\n'
        "  *offline*) cat <<'XEOF'\n"
        f"{FIXTURE_OFFLINE}"
        "XEOF\n"
        "  ;;\n"
        "  *) cat <<'XEOF'\n"
        f"{FIXTURE_RESIDENT}"
        "XEOF\n"
        "  ;;\n"
        "esac\n"
    )
    path.chmod(0o755)
    return str(path)


def _shard(directory, name, size=64 * 1024 * 1024):
    """A weight file of plausible size, written sparsely so the test costs no disk.

    truncate() rather than writing bytes: this is metadata the module reads, and a real
    64 MiB of zeros per case would make the suite slow for nothing.
    """
    directory.mkdir(parents=True, exist_ok=True)
    f = directory / name
    with open(f, "wb") as fh:
        fh.truncate(size)
    return f


class TestTheTwoReadings:
    """The parser, against the real format."""

    def test_the_migrated_shard_parses_as_archive_offline(self):
        assert cwr.parse_misc_attributes(FIXTURE_OFFLINE) == {"ARCHIVE", "OFFLINE"}

    def test_the_resident_shard_parses_as_archive_alone(self):
        assert cwr.parse_misc_attributes(FIXTURE_RESIDENT) == {"ARCHIVE"}

    def test_offline_tokens_separates_the_two(self):
        """The trap the module's docstring names: ARCHIVE is universal here, OFFLINE is the
        signal. Keying on ARCHIVE would refuse every run on this cluster."""
        assert cwr.parse_misc_attributes(FIXTURE_OFFLINE) & cwr.OFFLINE_TOKENS
        assert not cwr.parse_misc_attributes(FIXTURE_RESIDENT) & cwr.OFFLINE_TOKENS
        assert "ARCHIVE" not in cwr.OFFLINE_TOKENS

    def test_a_missing_field_is_none_and_not_an_empty_set(self):
        """None means UNMEASURED (another mmlsattr version, a non-GPFS path); set() would
        mean a real reading of "no attributes". Conflating them turns "I could not tell"
        into "it is fine"."""
        assert cwr.parse_misc_attributes("file name: x\nEncrypted: no\n") is None
        assert cwr.parse_misc_attributes("Misc attributes:\n") == set()

    def test_attrs_of_reports_unmeasured_when_there_is_no_mmlsattr(self):
        toks, detail = cwr.attrs_of("/nonexistent", None)
        assert toks is None
        assert "not available" in detail

    def test_attrs_of_reports_unmeasured_rather_than_raising_on_a_broken_tool(
        self, tmp_path
    ):
        """A missing or non-executable mmlsattr path must degrade to UNMEASURED, not kill
        the step with an OSError before training starts."""
        toks, detail = cwr.attrs_of("/nonexistent", str(tmp_path / "not-a-program"))
        assert toks is None
        assert "failed" in detail


class TestTheToolCanLoadInsideAContainer:
    """With /usr/lpp/mmfs bind-mounted into the step container, mmlsattr is FOUND and then
    fails to load libgpfs.so (rc 127): the host resolves it through /lib64/libgpfs.so, which
    the image does not have. Measured on BlueVela 2026-09-25 -- the preflight then ran on the
    heuristic alone and still said OK, so nothing but this test notices."""

    def _gpfs_tree(self, tmp_path):
        """/usr/lpp/mmfs's shape: bin/mmlsattr a symlink to bin/tslsattr, lib/ beside bin/."""
        root = tmp_path / "mmfs"
        (root / "bin").mkdir(parents=True)
        (root / "lib").mkdir()
        real = root / "bin" / "tslsattr"
        real.write_text('#!/bin/sh\necho "Misc attributes: LD=$LD_LIBRARY_PATH"\n')
        real.chmod(0o755)
        (root / "bin" / "mmlsattr").symlink_to("tslsattr")
        return root

    def test_gpfs_lib_is_put_first_on_the_library_path(self, tmp_path, monkeypatch):
        root = self._gpfs_tree(tmp_path)
        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
        env = cwr.mmlsattr_env(str(root / "bin" / "mmlsattr"))
        assert (
            env["LD_LIBRARY_PATH"] == f"{root.resolve() / 'lib'}:/usr/local/cuda/lib64"
        )

    def test_an_unset_library_path_gets_no_empty_entry(self, tmp_path, monkeypatch):
        """A trailing ':' is an empty entry, which ld.so reads as the current directory."""
        root = self._gpfs_tree(tmp_path)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        env = cwr.mmlsattr_env(str(root / "bin" / "mmlsattr"))
        assert env["LD_LIBRARY_PATH"] == str(root.resolve() / "lib")

    def test_no_lib_beside_the_tool_leaves_the_environment_alone(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        env = cwr.mmlsattr_env(_fake_mmlsattr(tmp_path))
        assert "LD_LIBRARY_PATH" not in env

    def test_the_subprocess_actually_receives_it(self, tmp_path, monkeypatch):
        root = self._gpfs_tree(tmp_path)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        toks, _ = cwr.attrs_of("/anything", str(root / "bin" / "mmlsattr"))
        assert toks == {f"LD={root.resolve() / 'lib'}"}


class TestMmlsattrIsAuthoritativeInBothDirections:
    def test_offline_is_a_problem(self, tmp_path):
        f = _shard(tmp_path / "offline-teacher", "model-00001-of-00001.safetensors")
        verdict, detail = cwr.probe(f, _fake_mmlsattr(tmp_path), {})
        assert verdict == "offline"
        assert "OFFLINE" in detail

    def test_resident_clears_the_file_even_when_barely_allocated(self, tmp_path):
        """The direction that matters for false alarms. This shard is sparse -- the
        allocation heuristic would condemn it -- and GPFS says it is here, so it is here.
        """
        f = _shard(tmp_path / "student", "model-00001-of-00001.safetensors")
        verdict, _ = cwr.probe(f, _fake_mmlsattr(tmp_path), {})
        assert verdict == "ok"

    def test_a_shared_realpath_is_measured_once(self, tmp_path, monkeypatch):
        """Two roles naming one checkpoint (the common case: teacher == student init) must
        not double the mmlsattr calls, and the symlink must be resolved -- asking about the
        link tells you nothing, since the incident's shards WERE links into an archive.
        """
        real = _shard(tmp_path / "real", "model-00001-of-00001.safetensors")
        link = tmp_path / "link.safetensors"
        link.symlink_to(real)
        calls = []
        seen = {}

        def counting(path, mm):
            calls.append(path)
            return {"ARCHIVE"}, "ARCHIVE"

        monkeypatch.setattr(cwr, "attrs_of", counting)
        assert cwr.probe(real, "mm", seen)[0] == "ok"
        assert cwr.probe(link, "mm", seen)[0] == "ok"
        assert calls == [os.path.realpath(real)]


class TestAllocationMayOnlyAccuse:
    def test_it_cannot_clear_a_file(self, tmp_path):
        """A fully allocated file with no mmlsattr is UNMEASURED, never "ok" -- the measured
        counterexample was 91.6% allocated and still OFFLINE, so allocation cannot rule out
        a partial recall."""
        f = _shard(tmp_path / "m", "model-00001-of-00001.safetensors", size=1024)
        with open(f, "wb") as fh:
            fh.write(b"\0" * 1024)
        assert cwr.probe(f, None, {})[0] == "unmeasured"

    def test_it_does_accuse_a_large_barely_allocated_file(self, tmp_path):
        """The one thing the weak instrument is allowed to do. Skipped rather than fudged if
        the filesystem refuses to keep the file sparse: what is under test is the decision,
        and without a sparse file there is no low ratio to decide on."""
        f = _shard(tmp_path / "m", "model-00001-of-00001.safetensors")
        st = os.stat(f)
        if st.st_blocks * 512 >= cwr.ALLOC_ACCUSE_BELOW * st.st_size:
            pytest.skip(
                f"filesystem pre-allocated the sparse file ({st.st_blocks} blocks)"
            )
        verdict, detail = cwr.probe(f, None, {})
        assert verdict == "offline"
        assert "heuristic" in detail

    def test_a_small_file_is_never_accused(self, tmp_path):
        """Below MIN_INTERESTING the arithmetic is noise -- block granularity and inlined
        data swamp it -- and nothing that small is a recall worth gating on."""
        f = _shard(tmp_path / "m", "tiny.bin", size=cwr.MIN_INTERESTING - 1)
        assert cwr.probe(f, None, {})[0] == "unmeasured"

    def test_a_missing_file_is_condemned(self, tmp_path):
        """A path that cannot be stat'ed is a problem, not an UNMEASURED: for a real
        filesystem path, "gone" is a finding. Reading a hub id as a path is what
        looks_like_a_local_path exists to prevent. It is UNREADABLE rather than OFFLINE so
        the refusal does not tell its reader to recall it from tape."""
        assert cwr.probe(tmp_path / "nope.safetensors", None, {})[0] == "unreadable"


class TestWhichFilesAreEvenLookedAt:
    def test_shards_one_level_down_are_found(self, tmp_path):
        _shard(tmp_path / "m", "model-00001-of-00002.safetensors")
        _shard(tmp_path / "m" / "nested", "model-00002-of-00002.safetensors")
        assert len(cwr.weight_files(tmp_path / "m")) == 2

    def test_dot_directories_are_skipped(self, tmp_path):
        """`.git` is the case that matters, not tidiness: the granite checkpoints under
        /proj are git-lfs clones, so each holds a SECOND copy of every shard that no
        trainer will ever open. Walking it doubles every count."""
        _shard(tmp_path / "m", "model-00001-of-00001.safetensors")
        _shard(tmp_path / "m" / ".git" / "lfs", "model-00001-of-00001.safetensors")
        found = cwr.weight_files(tmp_path / "m")
        assert [f.name for f in found] == ["model-00001-of-00001.safetensors"]
        assert ".git" not in str(found[0])

    def test_non_weight_files_are_ignored(self, tmp_path):
        _shard(tmp_path / "m", "config.json", size=64)
        _shard(tmp_path / "m", "tokenizer.model", size=64)
        assert cwr.weight_files(tmp_path / "m") == []


class TestAHubIdIsNotAPath:
    """Without this the preflight would abort every recipe that names a model by repo id --
    a check whose failure mode is breaking correct runs is worse than no check."""

    @pytest.mark.parametrize("value", ["ibm-granite/granite-4.0-tiny", "meta-llama/x"])
    def test_a_repo_id_is_not_a_path(self, value):
        assert not cwr.looks_like_a_local_path(value)

    @pytest.mark.parametrize("value", ["/proj/x", "./x", "../x", "~/x"])
    def test_an_explicit_path_is_a_path(self, value):
        assert cwr.looks_like_a_local_path(value)

    def test_an_existing_relative_path_is_a_path(self, tmp_path, monkeypatch):
        (tmp_path / "models").mkdir()
        monkeypatch.chdir(tmp_path)
        assert cwr.looks_like_a_local_path("models")


class TestAuditKeepsProblemsAndNotesApart:
    """Only `problems` can stop a run, so what lands in which list IS the behaviour."""

    def test_an_empty_role_is_a_note(self):
        problems, unreadable, notes, measured = cwr.audit([("student", "")], None)
        assert problems == unreadable == []
        assert measured == 0
        assert "empty" in notes[0]

    def test_a_hub_id_is_a_note(self):
        problems, unreadable, notes, _ = cwr.audit(
            [("student", "ibm-granite/granite-4.0-tiny")], None
        )
        assert problems == unreadable == []
        assert "not a local path" in notes[0]

    def test_a_nonexistent_local_path_is_unreadable_not_offline(self, tmp_path):
        problems, unreadable, _, _ = cwr.audit(
            [("student", str(tmp_path / "gone"))], None
        )
        assert problems == []
        assert "does not exist" in unreadable[0]

    def test_a_tilde_path_is_expanded(self, tmp_path, monkeypatch):
        """looks_like_a_local_path accepts `~`, so audit has to expand it: an unexpanded
        Path("~/x") is neither a file nor a directory, and would refuse a real model."""
        monkeypatch.setenv("HOME", str(tmp_path))
        _shard(tmp_path / "m", "model-00001-of-00001.safetensors")
        problems, unreadable, _, measured = cwr.audit(
            [("student", "~/m")], _fake_mmlsattr(tmp_path)
        )
        assert problems == unreadable == []
        assert measured == 1

    def test_a_directory_with_no_weights_is_a_note(self, tmp_path):
        """A tokenizer-only overlay is a legitimate value for these flags and has no
        shards. Refusing on it would block the retagging arms."""
        (tmp_path / "tok").mkdir()
        problems, unreadable, notes, _ = cwr.audit(
            [("tokenizer", str(tmp_path / "tok"))], None
        )
        assert problems == unreadable == []
        assert "no weight files" in notes[0]

    def test_a_file_may_be_named_directly(self, tmp_path):
        f = _shard(tmp_path / "m", "model-00001-of-00001.safetensors")
        _, _, _, measured = cwr.audit([("student", str(f))], _fake_mmlsattr(tmp_path))
        assert measured == 1


class TestExitCodes:
    """These differ from the standalone audit script on purpose: there, UNMEASURED is rc 2
    for a reviewer to read; here a non-zero rc aborts the step, so only an authoritative
    OFFLINE earns one."""

    def _run(self, tmp_path, *args, mm=None):
        env = dict(os.environ)
        if mm:
            env["PATH"] = f"{Path(mm).parent}:{env['PATH']}"
        return subprocess.run(
            [sys.executable, str(SRC_DIR / "check_weight_residency.py"), *args],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

    def _mm_on_path(self, tmp_path):
        """The fake, installed under mmlsattr's own name so find_mmlsattr picks it up from
        PATH -- which is also how the real one is found on a login shell."""
        fake = Path(_fake_mmlsattr(tmp_path))
        target = tmp_path / "bin" / "mmlsattr"
        target.parent.mkdir(exist_ok=True)
        target.write_text(fake.read_text())
        target.chmod(0o755)
        return str(target)

    def test_a_migrated_teacher_refuses(self, tmp_path):
        d = tmp_path / "offline-teacher"
        _shard(d, "model-00001-of-00001.safetensors")
        r = self._run(tmp_path, f"teacher={d}", mm=self._mm_on_path(tmp_path))
        assert r.returncode == 1
        assert "REFUSING" in r.stderr
        # The remedy is named: a refusal that does not say what to do gets overridden.
        assert "mmrestripefile" in r.stderr

    def test_allow_offline_proceeds_and_says_so(self, tmp_path):
        d = tmp_path / "offline-teacher"
        _shard(d, "model-00001-of-00001.safetensors")
        r = self._run(
            tmp_path, "--allow-offline", f"teacher={d}", mm=self._mm_on_path(tmp_path)
        )
        assert r.returncode == 0
        assert "OVERRIDDEN" in r.stderr

    def test_a_missing_path_refuses_without_the_tape_remedy(self, tmp_path):
        """Still rc 1 -- the run cannot load it -- but the message must not send its reader
        to stage in from tape, and --allow-offline does not wave it through: there is no
        slow first read to accept for a file that is not there."""
        for args in ([], ["--allow-offline"]):
            r = self._run(tmp_path, *args, f"student={tmp_path / 'gone'}")
            assert r.returncode == 1
            assert "NOT a tape recall" in r.stderr
            assert "mmrestripefile" not in r.stderr

    def test_a_resident_student_is_rc_zero(self, tmp_path):
        d = tmp_path / "student"
        _shard(d, "model-00001-of-00001.safetensors")
        r = self._run(tmp_path, f"student={d}", mm=self._mm_on_path(tmp_path))
        assert r.returncode == 0
        assert "OK" in r.stdout

    def test_unmeasurable_is_rc_zero_with_a_warning(
        self, tmp_path, monkeypatch, capsys
    ):
        """No mmlsattr is the normal case off this cluster, and must not stop a run that
        would have been fine. In-process, because the absolute-path fallback in
        find_mmlsattr means a subprocess on a GPFS host would find the real tool no matter
        what PATH says -- and it is find_mmlsattr returning None that is under test."""
        d = tmp_path / "student"
        _shard(d, "model-00001-of-00001.safetensors", size=1024)
        monkeypatch.setattr(cwr, "find_mmlsattr", lambda: None)
        monkeypatch.setattr(sys, "argv", ["check_weight_residency.py", f"student={d}"])
        assert cwr.main() == 0
        assert "WARN" in capsys.readouterr().out

    def test_no_roles_at_all_is_rc_zero(self, tmp_path):
        """A recipe that leaves every guarded path empty renders to this. It is not an
        error -- the step's own required-value checks are what catch a missing model."""
        assert self._run(tmp_path).returncode == 0

    def test_a_malformed_argument_is_rc_two(self, tmp_path):
        """Distinct from the refusal: rc 2 says the CALLER is wrong, which is a template
        bug, not a storage finding."""
        r = self._run(tmp_path, "/proj/some/path")
        assert r.returncode == 2
        assert "ROLE=PATH" in r.stderr

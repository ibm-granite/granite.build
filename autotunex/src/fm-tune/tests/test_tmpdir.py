"""Scheduler-agnostic Ray temp-dir resolution.

The failure being prevented, observed on an LSF node::

    drwxrwxr-x 3 otheruser otheruser /tmp/_ray   # created by another user
    PermissionError: [Errno 13] Permission denied: '/tmp/_ray/job_1823184'

``/tmp`` itself was healthy (``drwxrwxrwt``); only the shared intermediate level was
owned by a group the running user did not belong to. Jobs then failed or succeeded
purely on which node the scheduler happened to pick.
"""

import os
import shutil
import stat
import sys

import pytest

from autotune.tmpdir import (
    FALLBACK_ROOT,
    JOB_ID_ENV_VARS,
    MAX_TEMP_DIR_LEN,
    RAY_TMPDIR_ENV,
    SCRATCH_ENV_VARS,
    candidate_roots,
    resolve_job_id,
    resolve_ray_temp_dir,
)

ALL_ENV = (*JOB_ID_ENV_VARS, *SCRATCH_ENV_VARS, RAY_TMPDIR_ENV)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No scheduler variables leak in from the real environment."""
    for var in ALL_ENV:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def short_tmp():
    """A short-path scratch dir.

    pytest's ``tmp_path`` is ~100 chars on macOS — beyond
    :data:`MAX_TEMP_DIR_LEN`, so every candidate under it would be rejected for
    length and mask what these tests are actually checking. Ray needs short paths
    for the same reason, so exercise the code the way it runs in production.
    """
    import shutil
    import uuid

    root = os.path.join(FALLBACK_ROOT, f"fmt_t{uuid.uuid4().hex[:6]}")
    os.makedirs(root, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class TestResolveJobId:
    @pytest.mark.parametrize(
        "var",
        ["AUTOTUNE_JOB_ID", "LSB_JOBID", "SLURM_JOB_ID", "PBS_JOBID", "JOB_ID"],
    )
    def test_reads_each_scheduler_variable(self, var, monkeypatch):
        monkeypatch.setenv(var, "12345")
        assert resolve_job_id() == "12345"

    def test_precedence_is_most_specific_first(self, monkeypatch):
        monkeypatch.setenv("JOB_ID", "generic")
        monkeypatch.setenv("LSB_JOBID", "lsf")
        monkeypatch.setenv("AUTOTUNE_JOB_ID", "explicit")
        assert resolve_job_id() == "explicit"

    def test_falls_back_to_pid_not_a_constant(self):
        # The old code defaulted to "0", so two concurrent off-LSF jobs on one node
        # would share a temp dir and collide on the plasma socket.
        assert resolve_job_id() == str(os.getpid())

    def test_explicit_default_honoured(self):
        assert resolve_job_id(default="fallback") == "fallback"

    def test_blank_value_is_ignored(self, monkeypatch):
        monkeypatch.setenv("LSB_JOBID", "   ")
        monkeypatch.setenv("SLURM_JOB_ID", "real")
        assert resolve_job_id() == "real"


class TestCandidateRoots:
    def test_parent_is_per_uid(self):
        # The whole point: no shared intermediate directory another user can own.
        for root in candidate_roots():
            assert f"_ray-{os.getuid()}" in root

    def test_override_comes_first(self, monkeypatch, tmp_path):
        monkeypatch.setenv(RAY_TMPDIR_ENV, str(tmp_path / "override"))
        monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
        assert candidate_roots()[0] == str(tmp_path / "override")

    def test_override_used_verbatim(self, monkeypatch, tmp_path):
        # An operator naming an exact path gets it, without a uid level appended.
        monkeypatch.setenv(RAY_TMPDIR_ENV, str(tmp_path / "exact"))
        assert candidate_roots()[0] == str(tmp_path / "exact")

    def test_scratch_dirs_preferred_over_system_tmp(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
        roots = candidate_roots()
        assert str(tmp_path / "scratch") in roots[0]

    def test_hardcoded_fallback_always_present(self, monkeypatch, tmp_path):
        # Regression: tempfile.gettempdir() honours TMPDIR, so without a hardcoded
        # root a poisoned TMPDIR left NO fallback candidate at all.
        monkeypatch.setenv("TMPDIR", str(tmp_path / "poisoned"))
        roots = candidate_roots()
        assert any(r.startswith(FALLBACK_ROOT) for r in roots), roots
        assert len(roots) >= 2

    def test_duplicates_collapsed(self, monkeypatch):
        monkeypatch.setenv("TMPDIR", FALLBACK_ROOT)
        roots = candidate_roots()
        assert len(roots) == len(set(roots))


class TestResolveRayTempDir:
    def test_creates_a_writable_dir(self, monkeypatch, short_tmp):
        monkeypatch.setenv(RAY_TMPDIR_ENV, os.path.join(short_tmp, "r"))
        path = resolve_ray_temp_dir("job_1")
        assert os.path.isdir(path)
        assert os.access(path, os.W_OK | os.X_OK)

    def test_parent_created_private(self, monkeypatch, short_tmp):
        monkeypatch.setenv("TMPDIR", short_tmp)
        path = resolve_ray_temp_dir("job_1")
        # 0700 so this directory can never become another user's blocker.
        mode = stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode)
        assert mode == 0o700, oct(mode)

    def test_idempotent(self, monkeypatch, short_tmp):
        monkeypatch.setenv(RAY_TMPDIR_ENV, os.path.join(short_tmp, "r"))
        assert resolve_ray_temp_dir("job_1") == resolve_ray_temp_dir("job_1")

    def test_falls_back_past_unwritable_parent(self, monkeypatch, short_tmp):
        """The otheruser scenario: parent exists, owned elsewhere, not writable."""
        poisoned = os.path.join(short_tmp, "p")
        blocked = os.path.join(poisoned, f"_ray-{os.getuid()}")
        os.makedirs(blocked)
        good = os.path.join(short_tmp, "g")
        try:
            os.chmod(blocked, 0o555)  # r-xr-xr-x: cannot create children
            monkeypatch.setenv("TMPDIR", poisoned)
            monkeypatch.setattr("autotune.tmpdir.FALLBACK_ROOT", good)
            monkeypatch.setattr("autotune.tmpdir.tempfile.gettempdir", lambda: good)
            path = resolve_ray_temp_dir("job_1823184")
            assert not path.startswith(poisoned), path
            assert os.access(path, os.W_OK)
        finally:
            os.chmod(blocked, 0o755)

    def test_skips_path_over_socket_length_cap(self, monkeypatch, short_tmp):
        # A too-long dir would break Ray's AF_UNIX plasma socket far more
        # confusingly than a permission error, so it is skipped like one.
        long_root = os.path.join(short_tmp, "x" * (MAX_TEMP_DIR_LEN + 20))
        os.makedirs(long_root)
        good = os.path.join(short_tmp, "g")
        monkeypatch.setenv(RAY_TMPDIR_ENV, long_root)
        monkeypatch.setattr("autotune.tmpdir.FALLBACK_ROOT", good)
        monkeypatch.setattr("autotune.tmpdir.tempfile.gettempdir", lambda: good)
        path = resolve_ray_temp_dir("job_1")
        assert not path.startswith(long_root)
        assert len(path) <= MAX_TEMP_DIR_LEN

    def test_resolved_path_leaves_room_for_plasma_socket(self, monkeypatch, tmp_path):
        monkeypatch.setattr("autotune.tmpdir.FALLBACK_ROOT", FALLBACK_ROOT)
        # Realistic worst case from a real run: long job id, hostname, pid.
        path = resolve_ray_temp_dir("w_1823184_p2-r28-n3_3728440")
        socket_path = os.path.join(path, "session_2026-09-21_13-19-26_123456_3728440", "sockets", "plasma_store")
        assert len(socket_path) < 108, f"{len(socket_path)} bytes exceeds AF_UNIX sun_path"

    def test_uuid_job_id_socket_path_fits_platform_limit(self, monkeypatch):
        """A UUID job id (what AutoTuneX passes) overflowed macOS's 103-byte sun_path."""
        monkeypatch.setattr("autotune.tmpdir.FALLBACK_ROOT", FALLBACK_ROOT)
        path = resolve_ray_temp_dir("job_79e3bb73-cd4f-42b4-bbca-6bf360256e34")
        try:
            # Worst-case Ray suffix: 26-char timestamp and a 7-digit pid.
            socket_path = os.path.join(path, "session_2026-10-21_20-43-24_404935_4194304", "sockets", "plasma_store")
            limit = 103 if sys.platform == "darwin" else 107  # AF_UNIX sun_path: 104 / 108 bytes incl. NUL
            assert len(socket_path) <= limit, f"{len(socket_path)} bytes > {limit}: {socket_path}"
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_long_suffix_is_shortened_not_rejected(self, monkeypatch, tmp_path):
        """An LSF worker suffix under /tmp/_ray-<uid> overflowed the old cap by a byte."""
        from autotune.tmpdir import _fit_suffix

        root = "/tmp/_ray-700454"
        suffix = "w_1823184_p2-r28-n3_3728440"
        path = _fit_suffix(root, suffix)
        assert len(path) <= MAX_TEMP_DIR_LEN
        assert path.startswith(root + "/w_1823184")
        assert _fit_suffix(root, suffix) == path  # deterministic
        assert _fit_suffix(root, suffix.replace("3728440", "3728441")) != path  # distinct stay distinct
        assert _fit_suffix(root, "job_42") == os.path.join(root, "job_42")  # short paths untouched

    @staticmethod
    def _fallback_to(monkeypatch, good):
        monkeypatch.setattr("autotune.tmpdir.FALLBACK_ROOT", good)
        monkeypatch.setattr("autotune.tmpdir.tempfile.gettempdir", lambda: good)

    def test_symlinked_parent_rejected(self, monkeypatch, short_tmp):
        """Another local user can pre-plant /tmp/_ray-<uid> as a symlink to a dir they control."""
        target = os.path.join(short_tmp, "t")
        os.makedirs(target)
        planted = os.path.join(short_tmp, "s")
        os.makedirs(planted)
        os.symlink(target, os.path.join(planted, f"_ray-{os.getuid()}"))
        good = os.path.join(short_tmp, "g")
        monkeypatch.setenv("TMPDIR", planted)
        self._fallback_to(monkeypatch, good)
        path = resolve_ray_temp_dir("job_1")
        assert path.startswith(good), path

    def test_symlinked_leaf_rejected(self, monkeypatch, short_tmp):
        target = os.path.join(short_tmp, "t")
        os.makedirs(target)
        root = os.path.join(short_tmp, "r")
        os.makedirs(root)
        os.symlink(target, os.path.join(root, "job_1"))
        good = os.path.join(short_tmp, "g")
        monkeypatch.setenv(RAY_TMPDIR_ENV, root)
        self._fallback_to(monkeypatch, good)
        path = resolve_ray_temp_dir("job_1")
        assert path.startswith(good), path

    def test_dir_owned_by_another_uid_rejected(self, monkeypatch, short_tmp):
        """A pre-created leaf owned by someone else (but usable by us) must not be trusted."""
        root = os.path.join(short_tmp, "o")
        foreign = os.path.join(root, "job_1")
        os.makedirs(foreign)
        real_lstat = os.lstat

        def fake_lstat(path, *args, **kwargs):
            st = real_lstat(path, *args, **kwargs)
            if os.path.normpath(path) == foreign:
                fields = list(st)
                fields[stat.ST_UID] = os.getuid() + 1
                return os.stat_result(fields)
            return st

        monkeypatch.setattr("autotune.tmpdir.os.lstat", fake_lstat)
        good = os.path.join(short_tmp, "g")
        monkeypatch.setenv(RAY_TMPDIR_ENV, root)
        self._fallback_to(monkeypatch, good)
        path = resolve_ray_temp_dir("job_1")
        assert path.startswith(good), path

    def test_world_open_own_dirs_are_tightened(self, monkeypatch, short_tmp):
        # Our own pre-existing 0777 dirs (e.g. from an older run under a loose
        # umask) are tightened to 0700 and then accepted.
        parent = os.path.join(short_tmp, f"_ray-{os.getuid()}")
        leaf = os.path.join(parent, "job_1")
        os.makedirs(leaf)
        os.chmod(parent, 0o777)
        os.chmod(leaf, 0o777)
        monkeypatch.setenv("TMPDIR", short_tmp)
        path = resolve_ray_temp_dir("job_1")
        assert path == leaf
        for p in (parent, leaf):
            mode = stat.S_IMODE(os.lstat(p).st_mode)
            assert mode == 0o700, (p, oct(mode))

    def test_failed_chmod_rejects_candidate(self, monkeypatch, short_tmp):
        """A chmod failure is not swallowed: the dir may still be group/world-open."""
        root = os.path.join(short_tmp, "c")
        leaf = os.path.join(root, "job_1")
        os.makedirs(leaf)
        os.chmod(leaf, 0o777)
        real_chmod = os.chmod

        def failing_chmod(path, *args, **kwargs):
            if os.path.normpath(path) == leaf:
                raise PermissionError(1, "Operation not permitted", path)
            return real_chmod(path, *args, **kwargs)

        monkeypatch.setattr("autotune.tmpdir.os.chmod", failing_chmod)
        good = os.path.join(short_tmp, "g")
        monkeypatch.setenv(RAY_TMPDIR_ENV, root)
        self._fallback_to(monkeypatch, good)
        path = resolve_ray_temp_dir("job_1")
        assert path.startswith(good), path

    def test_normal_path_accepted_private(self, monkeypatch, short_tmp):
        monkeypatch.setenv("TMPDIR", short_tmp)
        path = resolve_ray_temp_dir("job_1")
        assert path == os.path.join(short_tmp, f"_ray-{os.getuid()}", "job_1")
        st = os.lstat(path)
        assert not stat.S_ISLNK(st.st_mode)
        assert st.st_uid == os.getuid()
        assert stat.S_IMODE(st.st_mode) == 0o700

    def test_raises_actionable_error_when_all_fail(self, monkeypatch, short_tmp):
        blocked = os.path.join(short_tmp, "b")
        os.makedirs(blocked)
        try:
            os.chmod(blocked, 0o555)
            monkeypatch.setenv(RAY_TMPDIR_ENV, os.path.join(blocked, "s1"))
            monkeypatch.setattr("autotune.tmpdir.FALLBACK_ROOT", os.path.join(blocked, "s2"))
            monkeypatch.setattr("autotune.tmpdir.tempfile.gettempdir", lambda: os.path.join(blocked, "s3"))
            with pytest.raises(RuntimeError) as exc:
                resolve_ray_temp_dir("job_1")
            msg = str(exc.value)
            # The message must be usable without reading the source.
            assert RAY_TMPDIR_ENV in msg, "should name the override variable"
            assert "administrator" in msg, "should suggest removing the stale dir"
            assert blocked in msg, "should name the offending path"
        finally:
            os.chmod(blocked, 0o755)


class TestCallSitesUseTheResolver:
    """All Ray temp dirs must route through the shared resolver."""

    PATHS = [
        "autotune/cluster.py",
        "autotune/lsf/ray_up.py",
        "autotune/lsf/ray_up_blaunch.py",
        "autotune/lsf/worker_entry.py",
        "autotune/slurm/ray_up_srun.py",
    ]

    @pytest.mark.parametrize("path", PATHS)
    def test_no_hardcoded_shared_tmp_parent(self, path):
        src = open(path).read()
        for bad in ('"/tmp/ray', "'/tmp/ray", '"/tmp/_ray', "'/tmp/_ray", 'f"/tmp/ray', 'f"/tmp/_ray'):
            assert bad not in src, f"{path}: hardcoded shared Ray tmp parent {bad!r}"

    @pytest.mark.parametrize("path", PATHS)
    def test_imports_resolver(self, path):
        assert "from autotune.tmpdir import" in open(path).read()

    @pytest.mark.parametrize("path", PATHS)
    def test_no_bare_lsb_jobid_for_temp_dir(self, path):
        # LSB_JOBID alone is LSF-only; off-LSF it collapsed to a shared "0".
        src = open(path).read()
        assert 'os.environ.get("LSB_JOBID", "0")' not in src, f"{path}: LSF-only job id"

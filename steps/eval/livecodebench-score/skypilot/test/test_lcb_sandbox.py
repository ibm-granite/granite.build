# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``src/lcb_sandbox.py``, the Landlock wrapper around the scorer.

A fake libc stands in for the kernel, so these run anywhere. The real kernel is
exercised by ``TestSandboxForReal`` in the render tests, where the host has Landlock.
"""

import ctypes
import errno
import importlib.util
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
_spec = importlib.util.spec_from_file_location("lcb_sandbox", _SRC / "lcb_sandbox.py")
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)


class FakeLibc:
    """Records the Landlock syscalls; ``abi`` < 1 fails the version query with ``err``."""

    def __init__(self, abi=6, err=0, fail=None):
        self.abi, self.err, self.fail = abi, err, fail
        self.ruleset = None  # (handled_fs, handled_net, scoped, size)
        self.rules = []  # (path, rights)
        self.no_new_privs = False
        self.restricted = False

    def _fails(self, step):
        if self.fail == step:
            ctypes.set_errno(errno.EPERM)
            return True
        return False

    def syscall(self, nr, *args):
        if nr == sb._SYS_CREATE_RULESET:
            attr, size, flags = args
            if flags == sb._CREATE_RULESET_VERSION:
                if self.abi < 1:
                    ctypes.set_errno(self.err)
                    return -1
                return self.abi
            if self._fails("create"):
                return -1
            a = ctypes.cast(attr, ctypes.POINTER(sb._RulesetAttr)).contents
            self.ruleset = (a.handled_access_fs, a.handled_access_net, a.scoped, size)
            return 99
        if nr == sb._SYS_ADD_RULE:
            _, _, attr, _ = args
            if self._fails("add"):
                return -1
            a = ctypes.cast(attr, ctypes.POINTER(sb._PathBeneathAttr)).contents
            self.rules.append((_fd_paths[a.parent_fd], a.allowed_access))
            return 0
        if nr == sb._SYS_RESTRICT_SELF:
            if self._fails("restrict"):
                return -1
            self.restricted = True
            return 0
        raise AssertionError(nr)

    def prctl(self, option, *_):
        assert option == sb._PR_SET_NO_NEW_PRIVS
        if self._fails("prctl"):
            return -1
        self.no_new_privs = True
        return 0


#: Which path each fake descriptor stands for.
_fd_paths: dict = {}


@pytest.fixture(autouse=True)
def _fake_descriptors(monkeypatch):
    """Rule paths get fake descriptors, so the tests need neither Linux's O_PATH nor
    read access to the paths; the fake ruleset fd 99 isn't real either."""
    _fd_paths.clear()

    def open_path(path):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        fd = 1000 + len(_fd_paths)
        _fd_paths[fd] = Path(path)
        return fd

    monkeypatch.setattr(sb, "_open_path", open_path)
    monkeypatch.setattr(sb.os, "close", lambda fd: None)


class TestAbi:
    def test_returns_the_kernels_version(self):
        assert sb.abi_version(FakeLibc(abi=6)) == 6

    @pytest.mark.parametrize(
        "err, why",
        [
            (errno.ENOSYS, "no Landlock"),
            (errno.EOPNOTSUPP, "Landlock is disabled"),
        ],
    )
    def test_missing_or_disabled_landlock_is_an_error(self, err, why):
        with pytest.raises(sb.SandboxError, match=why):
            sb.abi_version(FakeLibc(abi=-1, err=err))

    def test_handled_fs_grows_with_the_abi(self):
        assert sb.handled_fs(1) == sb._FS_ABI1
        assert sb.handled_fs(3) & sb._FS_TRUNCATE
        assert not sb.handled_fs(4) & sb._FS_IOCTL_DEV
        assert sb.handled_fs(5) & sb._FS_IOCTL_DEV


class TestRestrict:
    def test_refuses_an_abi_that_cant_deny_tcp(self):
        libc = FakeLibc(abi=3)
        with pytest.raises(sb.SandboxError, match="can't deny TCP"):
            sb.restrict([], [], libc)
        assert not libc.restricted

    def test_abi_6_denies_tcp_and_scopes_signals_and_sockets(self, tmp_path):
        libc = FakeLibc(abi=6)
        assert sb.restrict([], [str(tmp_path)], libc) == 6
        fs, net, scoped, size = libc.ruleset
        assert fs == sb.handled_fs(6)
        assert net == sb._NET_TCP
        assert scoped == sb._SCOPE_ALL and size == ctypes.sizeof(sb._RulesetAttr)
        assert libc.no_new_privs and libc.restricted

    def test_abi_4_leaves_out_the_scoped_field(self):
        """An ABI 4 kernel rejects a ruleset size that includes ``scoped``."""
        libc = FakeLibc(abi=4)
        sb.restrict([], [], libc)
        *_, scoped, size = libc.ruleset
        assert scoped == 0 and size == sb._RulesetAttr.scoped.offset

    def test_read_paths_are_read_only_and_write_paths_get_everything(self, tmp_path):
        ro_dir, rw_dir = tmp_path / "ro", tmp_path / "rw"
        ro_dir.mkdir(), rw_dir.mkdir()
        libc = FakeLibc(abi=6)
        sb.restrict([str(ro_dir)], [str(rw_dir)], libc)
        rights = dict(libc.rules)
        ro = sb._FS_EXECUTE | sb._FS_READ_FILE | sb._FS_READ_DIR
        assert rights[ro_dir] == ro
        assert rights[rw_dir] == sb.handled_fs(6)
        if Path("/dev/shm").exists():  # Linux; macOS has none
            assert rights[Path("/dev/shm")] == sb.handled_fs(6)

    def test_a_file_rule_keeps_only_file_rights(self, tmp_path):
        f = tmp_path / "prog.py"
        f.write_text("")
        libc = FakeLibc(abi=6)
        sb.restrict([str(f)], [], libc)
        (rights,) = [r for p, r in libc.rules if p == f]
        assert rights == sb._FS_EXECUTE | sb._FS_READ_FILE

    def test_missing_paths_are_skipped(self, tmp_path):
        libc = FakeLibc(abi=6)
        sb.restrict([str(tmp_path / "nope")], [], libc)
        assert libc.restricted
        assert tmp_path / "nope" not in dict(libc.rules)

    @pytest.mark.parametrize("step", ["create", "add", "prctl", "restrict"])
    def test_any_refused_step_is_an_error(self, step, tmp_path):
        libc = FakeLibc(abi=6, fail=step)
        with pytest.raises(sb.SandboxError):
            sb.restrict([str(tmp_path)], [], libc)
        assert not libc.restricted


class TestMain:
    def test_needs_a_program_after_the_separator(self, capsys):
        with pytest.raises(SystemExit):
            sb.parse_args(["--read", "/x", "--"])
        assert "no program to run" in capsys.readouterr().err

    def test_parses_paths_and_the_command(self):
        args = sb.parse_args(["--read", "/a", "--write", "/b", "--", "/bin/prog", "-x"])
        assert args.read == ["/a"] and args.write == ["/b"]
        assert args.command == ["/bin/prog", "-x"]

    def test_fails_closed_without_running_the_program(self, monkeypatch, capsys):
        def refuse(*_):
            raise sb.SandboxError("this kernel has no Landlock")

        ran = []
        monkeypatch.setattr(sb, "restrict", refuse)
        monkeypatch.setattr(sb.os, "execv", lambda *a: ran.append(a))
        assert sb.main(["--", "/bin/prog"]) == 1
        assert not ran
        err = capsys.readouterr().err
        assert "can't apply it (this kernel has no Landlock)" in err
        assert "sandbox: false" in err

    def test_restricts_then_execs_the_program(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            sb,
            "restrict",
            lambda read, write: calls.append(("restrict", read, write)) or 6,
        )
        monkeypatch.setattr(
            sb.os, "execv", lambda prog, argv: calls.append(("exec", prog, argv))
        )
        sb.main(
            ["--read", "/data", "--write", "/out", "--", "/venv/bin/python", "s.py"]
        )
        (_, read, write), exec_call = calls
        # The program itself and the running Python's own paths are readable too.
        assert "/data" in read and "/venv/bin/python" in read
        assert set(sb.python_paths()) <= set(read)
        assert write == ["/out"]
        assert exec_call == ("exec", "/venv/bin/python", ["/venv/bin/python", "s.py"])

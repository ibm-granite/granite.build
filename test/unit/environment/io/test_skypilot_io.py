from gbserver.environment.io.base import EnvironmentIO
from gbserver.environment.io.descriptors import HfInputIO
from gbserver.environment.io.skypilot import SkypilotIO


def test_environmentio_default_prologue_is_empty():
    class NoopIO(EnvironmentIO):
        pass

    io = NoopIO()
    assert io.render_prologue([]) == ""


def test_render_prologue_emits_hf_download():
    sh = SkypilotIO().render_prologue(
        [
            HfInputIO(
                repo="ns/model",
                revision="v1",
                type="model",
                dest="/cache/ns/model/v1",
                token="tok",
            )
        ]
    )
    assert 'hf download "ns/model" --local-dir "/cache/ns/model/v1"' in sh
    assert '--revision "v1"' in sh
    assert "--repo-type model" in sh


def test_render_prologue_empty_for_no_inputs():
    assert SkypilotIO().render_prologue([]) == ""


def test_build_env_io_returns_skypilot_io_for_skypilot_env():
    from gbserver.environment.io import build_env_io
    from gbserver.environment.io.skypilot import SkypilotIO as _SkypilotIO

    # An Environment instance's ``.type`` is ``self.__class__.__name__``
    # (environment.py L294), and EnvironmentConfig.type is the same class
    # identifier -- both are ``"Skypilot"`` (capitalized), NOT ``"skypilot"``.
    class _FakeSkypilotEnv:
        type = "Skypilot"

    assert isinstance(build_env_io(_FakeSkypilotEnv()), _SkypilotIO)


def test_build_env_io_returns_none_for_other_env():
    from gbserver.environment.io import build_env_io

    class _FakeK8sEnv:
        type = "K8s"

    assert build_env_io(_FakeK8sEnv()) is None

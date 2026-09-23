from gbserver.environment.io.base import EnvironmentIO


def test_environmentio_defaults_are_empty():
    class NoopIO(EnvironmentIO):
        pass

    io = NoopIO()
    assert io.render_prologue([]) == ""
    assert io.render_epilogue([], "cap") == ""
